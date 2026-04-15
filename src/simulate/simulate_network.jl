"""
    simulate_network(net::Network; 
                     new_params::Union{Nothing, NamedTuple, Dict}=nothing,
                     new_inits::Union{Nothing, Vector{Float64}}=nothing,
                     new_tspan::Union{Nothing, Tuple{Float64, Float64}}=nothing)::DataFrame

Run a single simulation for a network and return results as a DataFrame.

Executes one simulation of the network problem using solvers and parameters
defined in net.settings.simulation_settings. This is the high-level entry point
for simulating a network. For multiple runs, users should loop manually.

Results are returned as a DataFrame with properly named columns: `:time` followed 
by all state variable names in sorted order. If EEG output is specified in network 
settings, it can be computed from the state variables after retrieving the DataFrame.

# Arguments
- `net::Network`: The network object to simulate
- `new_params::Union{Nothing, NamedTuple, Dict}`: Override parameters as NamedTuple or Dict with String/Symbol keys (if nothing, use original)
- `new_inits::Union{Nothing, Vector{Float64}}`: Override initial conditions (if nothing, use original)
- `new_tspan::Union{Nothing, Tuple{Float64, Float64}}`: Override time span (if nothing, use original)

# Returns
- `df::DataFrame`: Solution as DataFrame with columns: `:time` + state variable names

# Example
```julia
net = build_network(settings)

# Single simulation with defaults
df = simulate_network(net)
# Columns: time, N1₊x11, N1₊x21, ...

# With custom initial conditions
new_inits = sample_inits(net.vars; seed=1234)
df = simulate_network(net; new_inits=new_inits)

# For multiple runs with different seeds/parameters:
dfs = [simulate_network(net; new_inits=sample_inits(net.vars; seed=s)) for s in 1:10]
```
"""
function simulate_network(net::Network; 
                         new_params::Union{Nothing, NamedTuple, Dict}=nothing,
                         new_inits::Union{Nothing, Vector{Float64}}=nothing,
                         new_tspan::Union{Nothing, Tuple{Float64, Float64}}=nothing,
                         settings::Union{Nothing, Settings}=nothing,
                         simulation_settings::Union{Nothing, SimulationSettings}=nothing)::DataFrame

    # Resolve simulation settings: use provided, or extract from provided settings, or use network's settings
    if simulation_settings === nothing
        if settings !== nothing
            simulation_settings = settings.simulation_settings
        else
            # Try to use settings from the network
            simulation_settings = net.settings.simulation_settings
        end
    end

    # Warn if parameters mismatch and new_params not provided
    if new_params === nothing
        _warn_param_mismatches!(net)
    end

    # Get solver and kwargs from settings
    solver = get_solver(net.problem, simulation_settings)
    solver_kwargs = get_solver_kwargs(net.problem, simulation_settings)
    
    # Run single simulation with optional parameter overrides
    solution = simulate_network(net.problem; 
                               new_params=new_params, 
                               new_inits=new_inits,
                               new_tspan=new_tspan,
                               solver=solver, 
                               solver_kwargs=solver_kwargs)
    
    vinfo("Network $(net.name) simulation completed successfully.", level=1)
    
    # Convert solution to DataFrame with proper column names
    df = sol2df(solution, net)
    
    return df
end

"""
    simulate_network(prob::SciMLBase.AbstractDEProblem; 
                     new_params::Union{Nothing, NamedTuple, Dict}=nothing, 
                     new_inits::Union{Nothing, Vector{Float64}}=nothing,
                     new_tspan::Union{Nothing, Tuple{Float64, Float64}}=nothing,
                     solver::SciMLBase.AbstractDEAlgorithm=Rodas5(),
                     solver_kwargs::NamedTuple=NamedTuple())::SciMLBase.AbstractODESolution

Simulate a network problem with optional parameter/initial condition overrides.

This is the low-level simulation entry point. It handles problem remapping with custom parameters,
initial conditions, and time spans, then solves using the specified solver and kwargs.

Parameters can be provided as a NamedTuple or Dict. If Dict is provided, it will be automatically
converted to a NamedTuple. Dict keys can be either Strings or Symbols and will be normalized to Symbols.

# Arguments
- `prob::SciMLBase.AbstractDEProblem`: The differential equation problem to solve
- `new_params::Union{Nothing, NamedTuple, Dict}`: Override parameters as NamedTuple or Dict with String/Symbol keys (if nothing, use original)
- `new_inits::Union{Nothing, Vector{Float64}}`: Override initial conditions (if nothing, use original)
- `new_tspan::Union{Nothing, Tuple{Float64, Float64}}`: Override time span (if nothing, use original)
- `solver::SciMLBase.AbstractDEAlgorithm`: Solver algorithm (default: Rodas5())
- `solver_kwargs::NamedTuple`: Solver configuration kwargs

# Returns
- `sol::SciMLBase.AbstractODESolution`: Solution object from the solver
"""
function simulate_network(prob::SciMLBase.AbstractDEProblem; 
                     new_params::Union{Nothing, NamedTuple, Dict}=nothing, 
                     new_inits::Union{Nothing, Vector{Float64}}=nothing,
                     new_tspan::Union{Nothing, Tuple{Float64, Float64}}=nothing,
                     solver::SciMLBase.AbstractDEAlgorithm=Rodas5(),
                     solver_kwargs::NamedTuple=NamedTuple())::SciMLBase.AbstractODESolution
    
    # Convert Dict to NamedTuple if needed
    params_for_remake = new_params
    if new_params isa Dict
        vinfo("Converting parameter Dict to NamedTuple"; level=2)
        # Handle both String and Symbol keys by converting to Symbols
        sym_dict = Dict(Symbol(k) => v for (k, v) in pairs(new_params))
        new_params_nt = NamedTuple(sym_dict)
        # Merge with existing parameters to preserve unchanged ones
        params_for_remake = merge(prob.p, new_params_nt)
    elseif new_params isa NamedTuple && new_params !== nothing
        # Merge NamedTuple with existing parameters to preserve unchanged ones
        params_for_remake = merge(prob.p, new_params)
    end
    
    # Ensure initial conditions are Vector{Float64}, not Vector{Any}
    inits_for_remake = if new_inits !== nothing
        Float64.(new_inits)
    else
        nothing
    end
    
    # Start with empty NamedTuple to collect remake parameters
    remake_kwargs = NamedTuple()

    # Helper to add a field to NamedTuple
    add(nt, k, v) = merge(nt, NamedTuple{(k,)}((v,)))

    # Add only fields that are specified (not nothing)
    if params_for_remake !== nothing
        remake_kwargs = add(remake_kwargs, :p, params_for_remake)
    end
    if inits_for_remake !== nothing
        remake_kwargs = add(remake_kwargs, :u0, inits_for_remake)
    end
    if new_tspan !== nothing
        remake_kwargs = add(remake_kwargs, :tspan, new_tspan)
    end

    # Remake problem with specified overrides (or return original if none specified)
    prob_remade = isempty(remake_kwargs) ? prob : DifferentialEquations.remake(prob; remake_kwargs...)

    # Solve the problem safely with callbacks for instability detection
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
        vwarn("Solver requires a dt parameter, but ss.dt is not defined. Using default dt=1e-4."; level=2)
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
            vwarn("Solver '$(ss.solver)' specified in settings is not valid for the problem type '$(typeof(prob).name.wrapper)'. Falling back to the default solver: $(default_solver_name)."; level=2)
            solver = solvers[default_solver_name]
        end
    else
        vinfo("Solver not specified in settings. Using default solver: $(default_solver_name)"; level=2)
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
    """
        sol2df(sol::AbstractTimeseriesSolution, net::Network)::DataFrame
    
    Convert ODE solution to DataFrame with properly named columns.
    
    Converts the raw solution object to a DataFrame where column names correspond to 
    semantically meaningful variable names instead of generic u[:1], u[:2], etc.
    
    # Arguments
    - `sol::AbstractTimeseriesSolution`: Solution from differential equation solver
    - `net::Network`: Network object (provides variable information for naming)
    
    # Returns
    - `df::DataFrame`: Time series data with columns: `:time` + sorted state variable names
    
    # Column Naming
    - Column 1 (index): `:time` - simulation time points
    - Columns 2+ : State variable names in sorted order (e.g., `:N1₊x11`, `:N1₊x21`, ...)
    
    # Example
    ```julia
    sol = simulate_network(net.problem; ...)
    df = sol2df(sol, net)
    
    # Access results by name
    t = df.time
    x11_values = df[Symbol("N1₊x11")]
    ```
    """
    df = DataFrame(sol)
    
    # Extract state variable names in consistent sorted order
    state_vars = get_symbols(get_state_vars(net.vars); sort=true)
    
    # Build column names: time + all state variables
    new_names = Symbol[:time]
    append!(new_names, Symbol.(state_vars))
    
    # Validate column count
    if ncol(df) != length(new_names)
        @error """
        Column count mismatch in sol2df:
          Solution has $(ncol(df)) columns
          Expected $(length(new_names)) columns: 1 time + $(length(state_vars)) state variables
          State variables: $(join(state_vars, ", "))
        """
        error("Cannot convert solution to DataFrame: column count mismatch")
    end
    
    # Rename columns with semantic names
    rename!(df, new_names)
    
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
    # Output directory: path_out / exp_name / network_name/
    output_dir = construct_output_dir(gs, net.settings.network_settings)
    filename = joinpath(output_dir, "$(net.name)_params_inits_run$(irun).csv")
    CSV.write(filename, df)
    return nothing
end


function save_ts_data(df::DataFrame, net::Network, gs::GeneralSettings, irun::Int64=1)::Nothing
    # Output directory: path_out / exp_name / network_name/
    output_dir = construct_output_dir(gs, net.settings.network_settings)
    filename = joinpath(output_dir, "$(net.name)_ts_data_run$(irun).csv")
    CSV.write(filename, df)

    return nothing
end


function _warn_param_mismatches!(net::Network)::Nothing
    """
        _warn_param_mismatches!(net::Network)::Nothing
    
    Check for parameter mismatches between net.problem.p and stored defaults.
    
    Issues a warning if any parameters in the network problem differ from their
    stored default values in net.params.params. This helps users detect when
    network state diverges from stored configuration.
    
    # Arguments
    - `net::Network`: Network to check for parameter mismatches
    
    # Returns
    - `nothing`
    """
    if !hasproperty(net.params, :params)
        return nothing
    end
    
    try
        prob_params = net.problem.p
        stored_params = net.params.params
        
        # Check for mismatches between problem parameters and stored defaults
        mismatches = []
        for param in stored_params
            param_name = Symbol(param.name)
            if haskey(prob_params, param_name)
                prob_value = getproperty(prob_params, param_name)
                if prob_value != param.default
                    push!(mismatches, (name=param.name, prob_value=prob_value, default=param.default))
                end
            end
        end
        
        if !isempty(mismatches)
            mismatch_str = join(["$(m.name): prob=$(m.prob_value), default=$(m.default)" for m in mismatches], "\n  ")
            vwarn("""
            Parameter mismatch detected between net.problem.p and stored defaults:
              $(mismatch_str)
            Consider passing updated parameters via new_params argument. 
            In this simulation, the parameters from net.problem.p will be used."""; level=1)
        end
    catch
        # Skip warning if comparison fails
    end
    
    return nothing
end
