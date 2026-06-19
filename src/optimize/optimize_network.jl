
# using Evolutionary, Optimization, Statistics, Distributions, Interpolations, Dates

const OPTIMIZATION_PENALTY_LOSS = 1e9
const DYNAMIC_N_RESTARTS_MAX_BATCHES = 5

restart_rng(seed::Union{Nothing, Int}, irestart::Int) =
    seed === nothing ? Random.default_rng() : MersenneTwister(seed + irestart)

function restart_found_nonpenalty_loss(
    optsol::Union{Nothing, SciMLBase.OptimizationSolution},
    runlog::Vector{OptLogEntry},
    expected_params_len::Int;
    penalty_loss::Float64=OPTIMIZATION_PENALTY_LOSS,
)::Bool
    optsol === nothing && return false
    effective_loss = ENEEGMA.effective_optimization_loss(
        optsol,
        runlog;
        expected_params_len=expected_params_len,
    )
    return isfinite(effective_loss) && effective_loss < penalty_loss
end

function optimize_network(
    net::Network,
    data::Data,
    settings::Settings;
    hyperparam_combo::Union{Nothing, Tuple}=nothing,
    hyperparam_idx::Union{Nothing, Int}=nothing,
    hyperparam_keys::Union{Nothing, Vector{String}}=nothing
    )
    # Validate settings before proceeding
    check_settings(settings; for_optimization=true)
    
    gs = settings.general_settings
    ns = settings.network_settings
    ss = settings.simulation_settings
    os = settings.optimization_settings
    ls = os.loss_settings
    original_output_dir = os.output_dir
    
    # Create numbered optimization folder ONCE if not already set
    if os.output_dir === nothing
        base_output_dir = ENEEGMA.construct_output_dir(gs, ns)
        optimization_output_dir = ENEEGMA.find_next_numbered_folder(base_output_dir, "optimization")
        mkpath(optimization_output_dir)
        os.output_dir = optimization_output_dir
        vinfo("Created optimization folder: $optimization_output_dir"; level=2)
    end
    
    vinfo("Starting optimization of $(gs.exp_name) - $(net.name)"; level=1)

    # get some optimization components
    solver = ENEEGMA.get_solver(net.problem, ss)
    solver_kwargs = ENEEGMA.get_solver_kwargs(net.problem, ss)
    loss_fun = ENEEGMA.get_loss_function(os.loss_settings) 
    optimizer = ENEEGMA.get_optimizer(os)
    
    blocks = ENEEGMA.prepare_optimization_blocks(net, os)
    tunable_params_symbols = blocks.tunable_params_symbols
    tunable_params_lb = blocks.tunable_params_lb
    tunable_params_ub = blocks.tunable_params_ub
    initial_values_native = blocks.initial_values_native
    inits_lb = blocks.init_lb
    inits_ub = blocks.init_ub
    param_spec = blocks.param_spec
    init_spec = blocks.init_spec
    use_reparam = os.reparametrize && os.reparam_strategy != :none
    loss_impl = ENEEGMA.wrap_loss_for_reparam(loss_fun, param_spec, init_spec; active=use_reparam)
    optfun = Optimization.OptimizationFunction(loss_impl)
    
    all_params = net.problem.p
    param_lb, param_ub = ENEEGMA.reparam_bounds(param_spec, tunable_params_lb, tunable_params_ub)
    init_lb, init_ub = ENEEGMA.reparam_bounds(init_spec, inits_lb, inits_ub)
    tunables_lb = vcat(param_lb, init_lb)
    tunables_ub = vcat(param_ub, init_ub)

    # create setter function for updating NamedTuple of **all params** (not just tunables) inside Problem. Nothing to do with inits here.
    setter = ENEEGMA.make_namedtuple_setter(all_params, Tuple(tunable_params_symbols))
    args = (
        net=net,
        settings=settings,
        prob=net.problem, data=data, setter=setter,
        all_params=all_params,
        tspan=ss.tspan, loss_settings=ls,
        solver=solver, solver_kwargs=solver_kwargs,
        data_settings=settings.data_settings,
    )

    best_optsol = nothing
    best_loss = Inf
    optlogger = OptLogEntry[]
    failure_reasons = String[]
    expected_tunables_len = length(tunables_lb)
    dynamic_restart_mode = os.dynamically_increase_n_restarts_upon_unsuccess
    base_restart_count = os.n_restarts
    total_restart_limit = dynamic_restart_mode ? base_restart_count * DYNAMIC_N_RESTARTS_MAX_BATCHES : base_restart_count
    irestart = 1
    batch_idx = 1

    while irestart <= total_restart_limit
        batch_end = min(irestart + base_restart_count - 1, total_restart_limit)
        batch_found_nonpenalty = false

        while irestart <= batch_end
            start_time = Dates.now()
            restart_rng = ENEEGMA.restart_rng(settings.general_settings.seed, irestart)
            
            optsol, runlog, failure_reason = ENEEGMA.singlerun_optimization(irestart, batch_end, optfun, optimizer, args, 
                tunable_params_symbols, tunables_lb, tunables_ub,
                os, start_time, net, param_spec, init_spec, initial_values_native, inits_lb, inits_ub,
                restart_rng
            );

            append!(optlogger, runlog)
            if failure_reason !== nothing
                push!(failure_reasons, failure_reason)
            end

            if optsol !== nothing && optsol.objective < best_loss
                best_loss = optsol.objective
                best_optsol = optsol
            end
            
            # Optionally save results for this restart
            if optsol !== nothing
                save_optimization_results(optsol, runlog, setter, net, data, settings; 
                                        blocks=blocks, 
                                        restart_idx=irestart,
                                        hyperparam_combo=hyperparam_combo, 
                                        hyperparam_idx=hyperparam_idx, 
                                        hyperparam_keys=hyperparam_keys)
            end

            # vinfo("Completed restart $irestart/$(os.n_restarts), current best loss: $best_loss. \n"; level=1)
            # flush(stdout)

            if best_loss <= os.loss_settings.abs_target_loss
                vinfo("Target loss $(os.loss_settings.abs_target_loss) reached. Ending optimization early."; level=1)
                break
            end

            batch_found_nonpenalty |= ENEEGMA.restart_found_nonpenalty_loss(
                optsol,
                runlog,
                expected_tunables_len,
            )

            irestart += 1
        end

        if best_loss <= os.loss_settings.abs_target_loss
            break
        end

        if !dynamic_restart_mode
            break
        end

        if batch_found_nonpenalty
            if batch_idx > 1
                vwarn(
                    "Dynamic restart search found a non-penalty loss in batch $batch_idx. Ending after $(irestart - 1) total restart(s).";
                    level=1,
                )
            end
            break
        elseif batch_idx < DYNAMIC_N_RESTARTS_MAX_BATCHES
            next_batch_end = min(irestart + base_restart_count - 1, total_restart_limit)
            vwarn(
                "Batch $batch_idx finished with only penalty/non-finite losses. " *
                "Trying restarts $(irestart):$next_batch_end.";
                level=1,
            )
            batch_idx += 1
        else
            vwarn(
                "Dynamic restart expansion stopped after $(irestart - 1) restart(s) " *
                "across $(DYNAMIC_N_RESTARTS_MAX_BATCHES) batch(es) without a non-penalty loss.";
                level=1,
            )
            break
        end
    end

    # Validate that at least one optimization succeeded
    if best_optsol === nothing
        detail = isempty(failure_reasons) ? "" : "\nLast failure:\n" * failure_reasons[end]
        max_len = 1200
        detail = length(detail) > max_len ? string(detail[1:max_len], " … [truncated]") : detail
        os.output_dir = original_output_dir
        error("All optimization attempts failed. Check solver settings and model stability." * detail)
    end

    os.output_dir = original_output_dir
    return best_optsol, optlogger, setter, blocks
end

function singlerun_optimization(
    irestart::Int,
    restart_limit::Int,
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
    inits_ub::Vector{Float64},
    restart_rng::AbstractRNG
)

    vinfo("\n[Restart $irestart/$restart_limit]"; level=1)

    optlogger = OptLogEntry[]
    failure_reason = nothing
    callback_fun = ENEEGMA.create_callback(start_time, irestart, optlogger, os;
                                   param_spec=param_spec,
                                   init_spec=init_spec)

    # Sample fresh parameter and initial values for this restart
    tunable_params_guess = ENEEGMA.sample_param_values(net.params; p_subset=tunable_params_symbols, return_type="vector", rng=restart_rng)
    tunable_params_guess = ENEEGMA.map_to_shared_space(tunable_params_guess, param_spec)
    initial_values_guess_native = ENEEGMA.sample_inits(net.vars; return_type="vector", sort=true, rng=restart_rng)
    initial_values_guess = ENEEGMA.map_to_shared_space(initial_values_guess_native, init_spec)
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
                                            maxiters=os.maxiters,
                                            rng=restart_rng
        )

        u_phys = materialize_logged_params(current_optsol.u, param_spec, init_spec)
        vinfo("Optimization completed with:"; level=1)
        vinfo("  Return code: $(current_optsol.retcode)"; level=1)
        vinfo("  Final loss: $(current_optsol.objective)"; level=1)
        flush(stdout)
        
        # Check if final loss differs from best logged loss
        best_logged_entry = ENEEGMA.best_optlogger_entry(
            optlogger;
            expected_params_len=length(tunables_guess),
        )
        if best_logged_entry !== nothing
            best_logged_loss = best_logged_entry.loss
            if !isfinite(current_optsol.objective) ||
               abs(current_optsol.objective - best_logged_loss) > 1e-6
                vinfo("  ⚠ Best logged loss: $(round(best_logged_loss, digits=6)) (optimizer drifted from best)"; level=2)
            end
        end
        
        if !isempty(initial_values_native)
            init_values_str = join(round.(initial_values_native; digits=3), ", ")
            vinfo("  Initial values: [$init_values_str]"; level=2)
        end
        if u_phys !== nothing && !isempty(u_phys)
            params_str = join(round.(u_phys; digits=3), ", ")
            vinfo("  Native-space params: [$params_str]"; level=2)
        end

    catch e
        bt = catch_backtrace()
        failure_reason = sprint(showerror, e, bt)
        vwarn("Error during optimization: $(typeof(e))"; level=2)
        if hasfield(typeof(e), :msg)
            vwarn("$(e.msg)"; level=2)
        else
            vwarn("$(e)"; level=2)
        end
        last_loss = length(optlogger) > 0 ? optlogger[end].loss : 1e9
        logged_params = current_optsol === nothing ? nothing : materialize_logged_params(current_optsol.u, param_spec, init_spec)
        push!(optlogger, OptLogEntry(irestart,
                                     length(optlogger) > 0 ? optlogger[end].iter : 0,
                                     last_loss,
                                     Second(round(Dates.value((now() - start_time)) / 1000)),
                                     logged_params))
        vwarn("Optimization failed with error. Continuing with next restart."; level=2)
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
            vwarn("Time limit reached: $(elapsed_time) minutes. Ending optimization."; level=2)
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
            vinfo("Restart $irestart | Iteration $true_iter | Loss: $l | Elapsed time: $(elapsed_time)"; level=1)
        end
        
        return false
    end
    return callback_fun
end

## -------------
## OPTMIZATION MODULE UTILS
## -------------


# Utility: create a setter for a NamedTuple and a list of parameter names
function make_namedtuple_setter(template_nt::NamedTuple, tunable_params_symbols::Tuple)
    field_names = propertynames(template_nt)
    field_index_map = Dict{Symbol, Int}(name => idx for (idx, name) in enumerate(field_names))
    update_indices = Int[get(field_index_map, name, 0) for name in tunable_params_symbols]
    any(iszero, update_indices) && error("Some tunable parameters are missing from the parameter NamedTuple template.")
    namedtuple_type = NamedTuple{field_names}

    function setter(nt::NamedTuple, values::AbstractVector)
        length(values) < length(update_indices) && error("Setter received fewer values than tunable parameters.")
        fields = Any[Tuple(nt)...]
        @inbounds for i in eachindex(update_indices)
            fields[update_indices[i]] = values[i]
        end
        return namedtuple_type(Tuple(fields))
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
    
    return Evolutionary.CMAES(; optimizer_kwargs..., metrics = [Evolutionary.RelDiff(os.loss_settings.loss_reltol)])
end

