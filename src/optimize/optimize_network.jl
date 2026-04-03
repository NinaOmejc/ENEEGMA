

function optimize_network(
    net::Network,
    data::TargetPSD,
    settings::Settings;
    hyperparam_combo::Union{Nothing, Tuple}=nothing,
    hyperparam_idx::Union{Nothing, Int}=nothing,
    hyperparam_keys::Union{Nothing, Vector{String}}=nothing
    )

    vprint("--- STARTING OPTIMIZATION ")
    ss = settings.simulation_settings
    os = settings.optimization_settings
    ls = os.loss_settings

    tspan = ss.tspan
    tspan === nothing && error("SimulationSettings.tspan must be specified for optimization.")

    # get some optimization components
    solver = ENMEEG.get_solver(net.problem, ss)
    solver_kwargs = ENMEEG.get_solver_kwargs(net.problem, ss)
    loss_fun = ENMEEG.get_loss_function(os.loss)
    optimizer = ENMEEG.get_optimizer(os)
    
    blocks = ENMEEG.prepare_optimization_blocks(net, os)
    tunable_params_symbols = blocks.tunable_params_symbols
    tunable_params_lb = blocks.tunable_params_lb
    tunable_params_ub = blocks.tunable_params_ub
    initial_values_native = blocks.initial_values_native
    inits_lb = blocks.init_lb
    inits_ub = blocks.init_ub

    param_spec = blocks.param_spec
    init_spec = blocks.init_spec
    use_reparam = os.reparametrize && os.reparam_strategy != :none
    loss_impl = wrap_loss_for_reparam(loss_fun, param_spec, init_spec; active=use_reparam)
    optfun = OptimizationFunction(loss_impl)
        
    param_lb, param_ub = reparam_bounds(param_spec, tunable_params_lb, tunable_params_ub)
    init_lb, init_ub = reparam_bounds(init_spec, inits_lb, inits_ub)
    tunables_lb = vcat(param_lb, init_lb)
    tunables_ub = vcat(param_ub, init_ub)
    ensure_population_size!(os, length(tunables_lb))

    # Note: tscale parameters are now proper Param objects with defaults,
    # so they're automatically included in net.problem.p
    all_params = net.problem.p

    # create setter function for updating NamedTuple of **all params** (not just tunables) inside Problem. Nothing to do with inits here.
    setter = make_namedtuple_setter(Tuple(tunable_params_symbols))
    brain_source_idx = ENMEEG.get_brain_source_idx(net)

    args = (
        prob=net.problem, data=data, setter=setter,
        all_params=all_params,
        tspan=tspan, loss_settings=ls,
        brain_source_idx=brain_source_idx,
        solver=solver, solver_kwargs=solver_kwargs,
    )

    best_optsol = nothing
    best_loss = Inf
    optlogger = OptLogEntry[]
    failure_reasons = String[]

    for irestart = 1:os.n_restarts

        start_time = now()
        
        optsol, runlog, failure_reason = ENMEEG.singlerun_optimization(irestart, optfun, optimizer, args, 
            tunable_params_symbols, tunables_lb, tunables_ub,
            os, start_time, net, param_spec, init_spec, initial_values_native, inits_lb, inits_ub
        )

        append!(optlogger, runlog)
        if failure_reason !== nothing
            push!(failure_reasons, failure_reason)
        end

        if optsol !== nothing && optsol.minimum < best_loss
            best_loss = optsol.minimum
            best_optsol = optsol
        end
        
        if settings.optimization_settings.save_all_optim_restarts_results && optsol !== nothing
            ENMEEG.save_optimization_results(optsol, runlog, setter, net, data, settings; blocks=blocks, restart_idx=irestart,
                                              hyperparam_combo=hyperparam_combo, hyperparam_idx=hyperparam_idx, hyperparam_keys=hyperparam_keys)
        end

        vprint("Completed restart $irestart/$(os.n_restarts), current best loss: $best_loss. \n")
        flush(stdout)

        if best_loss <= os.abs_target_loss
            vprint("Target loss $(os.abs_target_loss) reached. Ending optimization early.")
            break
        end
    end

    if best_optsol === nothing
        detail = isempty(failure_reasons) ? "" : "\nLast failure:\n" * failure_reasons[end]
        max_len = 1200
        detail = length(detail) > max_len ? string(detail[1:max_len], " … [truncated]") : detail
        error("All optimization attempts failed. Check solver settings and model stability." * detail)
    end

    # NOTE: best_optsol.u contains the final optimizer state, which may not be the actual best.
    # The true best parameters are tracked in optlogger. check_optimization_results() will
    # extract min_loss.params from optlogger for plotting and results export.
    
    return best_optsol, optlogger, setter, blocks
end

function singlerun_optimization(
    irestart::Int,
    optfun::OptimizationFunction,
    optimizer::Evolutionary.AbstractOptimizer,
    args::NamedTuple,
    tunable_params_symbols::Vector{Symbol},
    tunables_lb::Vector{Float64},
    tunables_ub::Vector{Float64},
    os::OptimizationSettings,
    start_time::Dates.DateTime,
    net::Network,
    param_spec::ReparamSpec,
    init_spec::ReparamSpec,
    initial_values_native::Vector{Float64},
    inits_lb::Vector{Float64},
    inits_ub::Vector{Float64}
)

    vprint("\n[Restart $irestart/$(os.n_restarts)]")

    optlogger = OptLogEntry[]
    failure_reason = nothing
    callback_fun = ENMEEG.create_callback(start_time, irestart, optlogger, os;
                                   param_spec=param_spec,
                                   init_spec=init_spec)

    # Sample fresh parameter and initial values for this restart
    tunable_params_guess = sample_param_values(net.params; p_subset=tunable_params_symbols, return_type="vector")
    tunable_params_guess = map_to_shared_space(tunable_params_guess, param_spec)
    initial_values_guess_native = ENMEEG.sample_inits(net.vars; return_type="vector", sort=true)
    initial_values_guess = map_to_shared_space(initial_values_guess_native, init_spec)
    tunables_guess = vcat(tunable_params_guess, initial_values_guess)

    optprob = Optimization.OptimizationProblem(
        optfun,
        tunables_guess,
        args,
        lb = tunables_lb, 
        ub = tunables_ub
    )

    local current_optsol = nothing
    try
        current_optsol = Optimization.solve(optprob,
                                            optimizer;
                                            callback=callback_fun,
                                            maxiters=os.maxiters
        )

        u_phys = materialize_logged_params(current_optsol.u, param_spec, init_spec)
        vprint("Optimization completed with:")
        vprint("  Return code: $(current_optsol.retcode)")
        vprint("  Final loss: $(current_optsol.minimum)")
        vprint("  Iterations: $(length(optlogger) > 0 ? optlogger[end].iter : 0)")
        flush(stdout)
        
        # Check if final loss differs from best logged loss
        if !isempty(optlogger)
            best_logged_loss = minimum(entry.loss for entry in optlogger)
            if abs(current_optsol.minimum - best_logged_loss) > 1e-6
                vprint("  ⚠ Best logged loss: $(round(best_logged_loss, digits=6)) (optimizer drifted from best)")
            end
        end
        
        if !isempty(initial_values_native)
            init_values_str = join(round.(initial_values_native; digits=3), ", ")
            vprint("  Initial values: [$init_values_str]")
        end
        if u_phys !== nothing && !isempty(u_phys)
            params_str = join(round.(u_phys; digits=3), ", ")
            vprint("  Native-space params: [$params_str]")
        end

    catch e
        bt = catch_backtrace()
        failure_reason = sprint(showerror, e, bt)
        vprint("Error during optimization: $(typeof(e))")
        if hasfield(typeof(e), :msg)
            vprint(e.msg)
        else
            vprint(e)
        end
        last_loss = length(optlogger) > 0 ? optlogger[end].loss : 1e9
        logged_params = current_optsol === nothing ? nothing : materialize_logged_params(current_optsol.u, param_spec, init_spec)
        push!(optlogger, OptLogEntry(irestart,
                                     length(optlogger) > 0 ? optlogger[end].iter : 0,
                                     last_loss,
                                     Second(round(Dates.value((now() - start_time)) / 1000)),
                                     logged_params))
        vprint("Optimization failed with error. Continuing with next restart.")
        current_optsol = nothing
    end

    return current_optsol, optlogger, failure_reason
end

## -------------
## CALLBACKS
## -------------
"""
    create_callback(start_time, optlogger, os; kwargs...)

Create callback function for optimization progress tracking, with support for both single and multi-restart scenarios.

# Arguments
- `start_time`: Start time of overall optimization
- `optlogger`: Logger to track optimization progress
- `os`: OptimizationSettings containing optimization settings
- `log_every`: Iteration interval for logging state snapshots
- `patience`: Iteration window for early stopping heuristics
- `param_spec` / `init_spec`: Reparameterization specs for tunable parameters and state initials

# Returns
- `callback_fun`: Callback function for optimization
"""
function create_callback(start_time::Dates.DateTime, 
                         irestart::Int,
                         optlogger::Vector{OptLogEntry}, 
                         os::OptimizationSettings;
                         log_every::Int=1,
                         print_every::Int=10,
                         param_spec::ReparamSpec=ReparamSpec(0),
                         init_spec::ReparamSpec=ReparamSpec(0))::Function
        
    # Create a closure to track the last seen iteration value and maintain a running count
    last_logged_iter = 0
    iter_offset = 0
    
    # your global penalty value
    penalty_loss = 1e9
    penalty_tol  = 0.99         
      
    function callback_fun(state, l)   
   
        # time limit
        elapsed_time = Second(round(Dates.value((now() - start_time)) / 1000))
        if elapsed_time > Minute(os.time_limit_minutes)
            vprint("Time limit reached: $(elapsed_time) minutes. Ending optimization.")
            return true
        end

        # Handle non-monotonicity in iteration counter - do this for every call
        if state.iter < last_logged_iter
            iter_offset += last_logged_iter
        end
        true_iter = state.iter + iter_offset

        # only log occasionally
        if rem(true_iter, log_every) == 0
            logged_params = state.u === nothing ? nothing : materialize_logged_params(state.u, param_spec, init_spec)
            push!(optlogger, OptLogEntry(irestart, true_iter, l, elapsed_time, logged_params))
            last_logged_iter = state.iter
        end
        
        # print occasionally
        if rem(true_iter, print_every) == 0
            vprint("Restart $irestart | Iteration $true_iter | Loss: $l | Elapsed time: $(elapsed_time)")
        end
        
        return false
    end
    return callback_fun
end

## -------------
## OPTMIZATION MODULE UTILS
## -------------


# Utility: create a setter for a NamedTuple and a list of parameter names
function make_namedtuple_setter(tunable_params_symbols::Tuple)
    function setter(nt::NamedTuple, values::AbstractVector)
        nt_new = (; nt...)  # start with all original fields
        for (i, name) in enumerate(tunable_params_symbols)
            nt_new = merge(nt_new, NamedTuple{(name,)}((values[i],)))
        end
        return nt_new
    end
    return setter
end


## -------------
## OPTIMIZERS
## -------------

"""
    get_optimizer(os, oz)

Create and configure an optimizer based on the specified optimization method and options.
For LBFGS, handles multi-restart configuration if enabled.

# Arguments
- `os`: Dictionary containing optimization settings including the method
- `oz`: Dictionary containing method-specific options

# Returns
- `optimizer`: The configured optimizer object
- `use_restarts`: Boolean flag indicating if restarts should be used
- `restart_options`: Dictionary with restart-specific settings (n_restarts, etc.)
"""

function get_optimizer(os::OptimizationSettings)::Evolutionary.AbstractOptimizer
    oz = os.optimizer_settings

    if os.method != "CMAES"
        error("Optimization method not supported: $(os.method). Only CMAES is currently available.")
    end

    optimizer_kwargs = NamedTuple()
    if oz.population_size != -1
        optimizer_kwargs = merge(optimizer_kwargs, (λ = oz.population_size,))
    end
    if hasproperty(oz, :sigma0) && oz.sigma0 != -1
        optimizer_kwargs = merge(optimizer_kwargs, (sigma0 = oz.sigma0,))
    end
    
    return Evolutionary.CMAES(; optimizer_kwargs..., metrics = [Evolutionary.RelDiff(os.loss_reltol)])
end

