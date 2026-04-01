function simulate_network(net::Network)
    
    gs = net.settings.general_settings
    ss = net.settings.simulation_settings
    default_prob = net.problem;

    solver = get_solver(default_prob, ss)
    solver_kwargs = get_solver_kwargs(default_prob, ss)

    dfs = Vector{DataFrame}()
    for irun = 1:ss.n_runs

        new_inits = sample_inits(net.vars; return_type="named_tuple")
        save_params_and_inits(new_inits, net, gs, irun)
        
        if isa(new_inits, NamedTuple)
            new_inits = collect(values(new_inits))
        end

        sol = simulate_problem(default_prob; new_inits=new_inits, solver=solver, solver_kwargs=solver_kwargs)

        if sol.retcode != :Success
            @warn "Simulation failed for run $(irun) with return code: $(sol.retcode). Skipping to next run."
            push!(dfs, DataFrame())  # Push an empty DataFrame to maintain run count
            continue
        end

        df = sol2df(sol, net)
        push!(dfs, df)
        save_ts_data(df, net, gs, irun)

        if gs.verbose
            println("Run $(irun) completed for network $(net.name).")
        end
    end

    println("--- SIMULATION COMPLETED ---")

    return dfs;
end


function simulate_problem(prob::SciMLBase.AbstractDEProblem; 
                          new_params::Union{Nothing, NamedTuple}=nothing, 
                          new_inits::Union{Nothing, Vector{Float64}}=nothing,
                          new_tspan::Union{Nothing, Tuple{Float64, Float64}}=nothing,
                          solver::SciMLBase.AbstractDEAlgorithm=Rodas5(),
                          solver_kwargs::NamedTuple=NamedTuple())::SciMLBase.AbstractODESolution

    # Start with empty NamedTuple
    remake_kwargs = NamedTuple()

    # Helper to add a field
    add(nt, k, v) = merge(nt, NamedTuple{(k,)}((v,)))

    # Add only fields that are present
    if new_params !== nothing
        remake_kwargs = add(remake_kwargs, :p, new_params)
    end
    if new_inits !== nothing
        remake_kwargs = add(remake_kwargs, :u0, new_inits)
    end
    if new_tspan !== nothing
        remake_kwargs = add(remake_kwargs, :tspan, new_tspan)
    end

    # Remake problem with all specified parameters (or return original if none specified)
    prob_remade = isempty(remake_kwargs) ? prob : remake(prob; remake_kwargs...)

    # Solve the problem safely
    sol = safe_solve(prob_remade, solver; solver_kwargs=solver_kwargs)

    return sol
end


function safe_solve(prob::SciMLBase.AbstractDEProblem, solver::SciMLBase.AbstractDEAlgorithm; solver_kwargs::NamedTuple=NamedTuple())::SciMLBase.AbstractODESolution
    
    # Create callbacks for unstable detection
    unstable_condition(u, t, integ) =
        any(isnan, u) || any(isinf, u) || any(abs.(u) .> 1e6)
    cb_unstable = DiscreteCallback(unstable_condition, terminate!)

    badstep_condition(u, t, integ) = integ.dt < 1e-10
    cb_badstep = DiscreteCallback(badstep_condition, terminate!)
  
    return Logging.with_logger(QUIET_SOLVER_LOGGER) do
        solve(prob, solver; callback=CallbackSet(cb_unstable, cb_badstep), solver_kwargs..., maxiters=1_000_000)
    end
end

function get_solver_kwargs(prob::SciMLBase.AbstractDEProblem, ss::SimulationSettings)
    kws = NamedTuple()   # start with empty NamedTuple

    # helper to add a key/value pair
    # (creates a new NamedTuple, but this is very cheap)
    add(kws, name, value) = merge(kws, NamedTuple{(name,)}((value,)))

    # populate kws if fields exist and are non-nothing
    if haspropnn(ss, :tspan)
        kws = add(kws, :tspan, ss.tspan)
    end
    if haspropnn(ss, :saveat)
        kws = add(kws, :saveat, ss.saveat)
    end
    if haspropnn(ss, :reltol)
        kws = add(kws, :reltol, ss.reltol)
    end
    if haspropnn(ss, :abstol)
        kws = add(kws, :abstol, ss.abstol)
    end
    if haspropnn(ss, :maxiters) && ss.maxiters !== nothing && ss.maxiters > 0
        kws = add(kws, :maxiters, ss.maxiters)
    end

    # dt handling
    prob_is_stochastic = prob isa SDEProblem || prob isa SDDEProblem

    if haspropnn(ss, :dt)
        kws = add(kws, :dt, ss.dt)
    elseif solver_needs_dt(ss.solver, prob_is_stochastic)
        @warn "Solver requires a dt parameter, but ss.dt is not defined. Using default dt=1e-4."
        kws = add(kws, :dt, 1e-4)
    end

    return kws
end

function get_solver(prob::SciMLBase.AbstractDEProblem, ss::SimulationSettings)::SciMLBase.AbstractDEAlgorithm

    # Define solvers by problem type
    if prob isa ODEProblem
        solvers = Dict{String, SciMLBase.AbstractDEAlgorithm}(
            # Non-stiff solvers
            "Tsit5" => Tsit5(),           # Excellent general-purpose solver
            "RK4" => RK4(),               # Classic 4th order Runge-Kutta
            "BS3" => BS3(),               # Bogacki-Shampine 3/2 method
            "DP5" => DP5(),               # Dormand-Prince 5th order
            "Vern6" => Vern6(),           # 6th order Verner method
            "Vern7" => Vern7(),           # 7th order Verner method
            "Vern8" => Vern8(),           # 8th order Verner method
            "Vern9" => Vern9(),           # 9th order Verner method
            
            # Stiff solvers
            "Rosenbrock23" => Rosenbrock23(), # Good for mild stiffness
            "Rodas4" => Rodas4(),         # 4th order Rosenbrock for stiff problems
            "Rodas5" => Rodas5(),         # 5th order Rosenbrock for stiff problems
            "TRBDF2" => TRBDF2(),         # TR-BDF2 multistep method for stiff problems
            "KenCarp4" => KenCarp4(),     # 4th order ESDIRK method
            "QNDF" => QNDF(),             # High order QNDF for extremely stiff problems
            "Rodas4P" => Rodas4P(),       # Parallel Rodas4
            "Rodas5P" => Rodas5P(),       # Parallel Rodas5
            
            # Adaptive algorithm selection
            "AutoTsit5(Rosenbrock23)" => AutoTsit5(Rosenbrock23()),
            "AutoTsit5(Rosenbrock23())" => AutoTsit5(Rosenbrock23()),
            "AutoTsit5(Rodas5)" => AutoTsit5(Rodas5()),
            "AutoTsit5(Rodas5())" => AutoTsit5(Rodas5()),
            "AutoVern7(Rodas5)" => AutoVern7(Rodas5()),
            "AutoVern7(Rodas5())" => AutoVern7(Rodas5()),
            "AutoVern9(Rodas5)" => AutoVern9(Rodas5()),
            "AutoVern9(Rodas5())" => AutoVern9(Rodas5())
        )
        default_solver_name = "AutoTsit5(Rodas5())"

    
    elseif prob isa SDEProblem

        solvers = Dict{String, SciMLBase.AbstractDEAlgorithm}(
            "EM" => EM(),                # Euler-Maruyama
            "SOSRI" => SOSRI(),          # Default - Adaptive strong order 1.5 solver
            "EulerHeun" => EulerHeun(),  # Heun's method
            "SRA1" => SRA1(),            # Strong order 1.5 adaptive SRA algorithm
            "SRA3" => SRA3(),            # Strong order 1.5 adaptive SRA algorithm
            "SRIW1" => SRIW1(),          # Strong order 1.5 adaptive algorithm
            "SRIW2" => SRIW2(),          # Strong order 1.5 adaptive algorithm
            "SRI" => SRI(),              # Strong order 1.5 adaptive algorithm
            "RKMil" => RKMil(),          # Milstein scheme
            "ImplicitEM" => ImplicitEM(), # Implicit Euler-Maruyama (for stiff SDEs)
            "ISSEM" => ISSEM(),          # Implicit strong order 1.5 method (for stiff SDEs)
            "ISSEulerHeun" => ISSEulerHeun(), # Implicit Heun's method (for stiff SDEs)
            "ImplicitRKMil" => ImplicitRKMil(), # Implicit Milstein scheme (for stiff SDEs)
        )
        default_solver_name = "EM"

    elseif prob isa DDEProblem
        solvers = Dict{String, SciMLBase.AbstractDEAlgorithm}(
            "MethodOfSteps(Tsit5())" => MethodOfSteps(Tsit5()), # Default
            "MethodOfSteps(RK4())" => MethodOfSteps(RK4()),
            "MethodOfSteps(Vern7())" => MethodOfSteps(Vern7()),
            "MethodOfSteps(Vern9())" => MethodOfSteps(Vern9()),
            "MethodOfSteps(Rosenbrock23())" => MethodOfSteps(Rosenbrock23()),
            "MethodOfSteps(TRBDF2())" => MethodOfSteps(TRBDF2()),
            "MethodOfSteps(Rodas5())" => MethodOfSteps(Rodas5())
        )
        default_solver_name = "MethodOfSteps(Tsit5())"

    elseif prob isa SDDEProblem
    
        solvers = Dict{String, SciMLBase.AbstractDEAlgorithm}(
            "ImplicitEM" => ImplicitEM(), # Implicit Euler-Maruyama (for stiff SDEs)
            "LambaEM" => LambaEM(),          # Default - Adaptive strong order 1.5 solver for SDDEs
            "RKMil" => RKMil(),            # Milstein scheme for SDDEs
            "SOSRI" => SOSRI(),          # Strong order 1.5 adaptive solver for SDDEs
            "EulerHeun" => EulerHeun(),  # Heun's method
            )
        default_solver_name = "RKMil"

    end

    # Robust solver selection: handle ss.solver === nothing
    if hasproperty(ss, :solver) && ss.solver !== nothing && !isempty(ss.solver)
        if haskey(solvers, ss.solver)
            solver = solvers[ss.solver]
        else
            @warn "Solver '$(ss.solver)' specified in settings is not valid for the problem type '$(typeof(prob).name.wrapper)'. Falling back to the default solver: $(default_solver_name)."
            solver = solvers[default_solver_name]
        end
    else
        vprint("Solver not specified in settings. Using default solver: $(default_solver_name) for problem type '$(typeof(prob).name.wrapper)'.", level=2)
        solver = solvers[default_solver_name]
    end
    
    return solver
end


# Determine if the solver requires a dt parameter
function solver_needs_dt(solver, prob_is_stochastic::Bool=false)::Bool
    # Solvers that typically need dt (typically fixed-step solvers or certain SDE solvers)
    needs_dt_types = [
        EM, ImplicitEM, EulerHeun, RKMil,  # Fixed-step or SDE solvers
        CompositeAlgorithm                 # May contain fixed-step solvers
    ]
    
    # Check if solver is an instance of any of these types
    for solver_type in needs_dt_types
        if isa(solver, solver_type)
            return true
        end
    end
    
    # For SDE problems, always provide dt for stability
    if prob_is_stochastic
        return true
    end
    
    return false
end



function sol2df(sol::SciMLBase.AbstractTimeseriesSolution, net::Network)::DataFrame
    
    df = DataFrame(sol)
    
    # Generate new column names: timestamp and state variables
    state_vars = get_symbols(get_state_vars(net.vars))
    new_names = [:time]
    append!(new_names, [Symbol(name) for name in state_vars])
    
    # Check if the column count matches
    if length(new_names) == ncol(df)
        rename!(df, new_names)
    else
        @warn "Column count mismatch: $(ncol(df)) columns in data vs $(length(new_names)) variable names"
    end
    
    return df
end

function save_params_and_inits(inits::NamedTuple, net::Network, gs::GeneralSettings, irun::Int=1)::Nothing
    params = get_param_default_values(net.params; return_type="named_tuple")

    # Mixed-type value column to allow both Float64 and String
    df = DataFrame(name=String[], value=Union{Float64,String}[])
    for (k, v) in pairs(merge(params, inits))
        push!(df, (name=string(k), value=v))
    end

    # Add sensory input function string if present
    if !isempty(net.sensory_input_str)
        push!(df, (name="sensory_input_func", value=net.sensory_input_str))
    end

    # Generate filename and save; Create directory if it doesn't exist
    mkpath(gs.path_out)
    filename = joinpath(gs.path_out, "$(net.name)_params_inits_run$(irun).csv")
    CSV.write(filename, df)
    return nothing
    
end


function save_ts_data(df::DataFrame, net::Network, gs::GeneralSettings, irun::Int64=1)::Nothing
    
    filename = joinpath(gs.path_out, "$(net.name)_ts_data_run$(irun).csv")
    CSV.write(filename, df)

    return nothing
end
