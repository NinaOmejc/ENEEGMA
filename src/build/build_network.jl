function build_network(settings::Settings)::Network
    check_settings(settings)

    net = ENEEGMA.Network(settings);
    ENEEGMA.build_nodes!(net);            
    ENEEGMA.construct_internode_conn_dynamics!(net);        
    ENEEGMA.construct_sensory_input_dynamics!(net);  
    ENEEGMA.construct_network_dynamics!(net);
    ENEEGMA.construct_network_diffusion_dynamics!(net);   
    ENEEGMA.update_network_parameters!(net);
    ENEEGMA.update_network_vars!(net);
    ENEEGMA.construct_network_problem!(net);
    ENEEGMA.set_network_signature!(net);
    ENEEGMA.export_network(net);

    vinfo("Network $(net.name) successfully built."; level=1)
    return net;
end

function construct_internode_conn_dynamics!(net::Network)::Network
    gs = net.settings.general_settings;
    ns = net.settings.network_settings;
    for it = 1:length(net.nodes)
        target_node = get_node_by_nodeid(net, it)
        target_placeholder_vars = get_vars_by_type(target_node.vars, "internode_input")
        target_placeholder_symbols = get_symbols(target_placeholder_vars; sort=true)
        # target_placeholder_symbol = target_placeholder_symbols[1]
        for target_placeholder_symbol in target_placeholder_symbols
            target_placeholder_var = get_var_by_symbol(target_placeholder_vars, target_placeholder_symbol)
            target_pop = target_placeholder_var.parent_pop
            internode_conn_substitute_term = Num(0)

            for is = 1:length(net.nodes)
                source_node = get_node_by_nodeid(net, is)
                if source_node.id == target_node.id || net.conn[target_node.id, source_node.id] == 0.
                    continue
                end
                source_vars = get_vars_sending_internode_output(source_node.vars)
                # source_var = source_vars.vars[1] 
                for source_var in source_vars.vars
                    if ns.network_delay[target_node.id, source_node.id] != 0.0
                        source_var_final = ExtraVar(; name="h_$(name(source_var))",
                                                     type="state_history",
                                                     eq_idx=eq_idx(source_var),
                                                     parent_pop=source_var.parent_pop,
                                                     is_history_var=true,
                                                     unit=unit(source_var), description="Delayed $(description(source_var))")
                        add_var!(net.vars, source_var_final)
                    else
                        source_var_final = source_var
                    end

                    conn_fun = ns.network_conn_funcs[target_node.id, source_node.id]
                    supported_conn_funcs = get_conn_funcs()
                    if !haskey(supported_conn_funcs, conn_fun)
                        throw(ArgumentError("Invalid transformation dynamics: $(conn_fun). Supported types are: $(keys(supported_conn_funcs))."))
                    end
                    func = supported_conn_funcs[conn_fun]
                    nc_term, nc_params = func(target_pop, VarSet([source_var_final]))

                    highest_constant_idx = get_highest_postfix_index(target_pop.params; pop_id=target_pop.id)
                    c = Param("$(target_pop.parent_node.name)₊c$(target_pop.id)$(highest_constant_idx + length(nc_params.params) + 1)", :node_coupling, target_pop;
                        tunable=true, description="network connectivity strength")
                    add_param!(nc_params, c)
                    join_paramsets!(net.params, [nc_params])

                    internode_conn_substitute_term += c.symbol * nc_term
                end
            end

            for (ieq, eq) in enumerate(target_node.dynamics)
                target_node.dynamics[ieq] = Symbolics.substitute(eq, Dict(symbol(target_placeholder_var) => internode_conn_substitute_term))
            end

            print_internode_conn_info(net, target_node, target_pop.id, internode_conn_substitute_term, gs)
        end
    end
    return net
end

function print_internode_conn_info(net::Network, target_node::Node, target_pop_id::Int, internode_conn_substitute_term, gs::GeneralSettings)
    vinfo("Internode connection configured for node $(target_node.id): $(target_node.name)"; level=2)
    for source_node in net.nodes
        if  source_node.id == target_node.id || net.conn[target_node.id, source_node.id] == 0.
            continue
        end
        vinfo("  Receives from: $(source_node.name) → $(target_node.name)"; level=2)
    end
    conn_params = filter(p -> p.type == :node_coupling && p.parent_pop.parent_node.id == target_node.id,
                         net.params.params)
    if !isempty(conn_params)
        vinfo("  Connection parameters added: $(length(conn_params))"; level=2)
    end
end

function construct_sensory_input_dynamics!(net::Network)::Network
    ss = net.settings.simulation_settings;  
    gs = net.settings.general_settings;
    ns = net.settings.network_settings;
    si = net.sensory_input_str
    n_points = Int(floor((ss.tspan[2]-ss.tspan[1]) / ss.saveat)) + 1
    t_values = collect(ss.tspan[1]:ss.saveat:ss.tspan[2])
    rng = ns.sensory_seed === nothing ? nothing : MersenneTwister(ns.sensory_seed)

    if si == "None" || si === nothing || si == ""
        s_values = zeros(length(t_values))
    else
        try
            use_direct_random = false
            si_stripped = strip(si)
            if si_stripped == "randn" || occursin(r"^randn\s*\(\s*\)\s*$", si_stripped)
                s_values = rng === nothing ? randn(length(t_values)) : randn(rng, length(t_values))
                sensory_input_func = nothing
                use_direct_random = true
            elseif si_stripped == "rand" || occursin(r"^rand\s*\(\s*\)\s*$", si_stripped)
                s_values = rng === nothing ? rand(length(t_values)) : rand(rng, length(t_values))
                sensory_input_func = nothing
                use_direct_random = true
            elseif rng === nothing
                sensory_input_expr_func_str = "t -> " * si
                sensory_input_func = eval(Meta.parse(sensory_input_expr_func_str))
            else
                expr = Meta.parse(si)
                sensory_input_func = eval(quote
                    let rand = (args...)->Random.rand($rng, args...),
                        randn = (args...)->Random.randn($rng, args...)
                        t -> $expr
                    end
                end)
            end
            if use_direct_random
                # s_values already set
            else
                s_values = [Base.invokelatest(sensory_input_func, it) for it in t_values]
            end
        catch e
            throw(ArgumentError("Error processing sensory input expression '$si': $e"))
        end
    end

    net.sensory_input_func = Interpolations.linear_interpolation(t_values, s_values; extrapolation_bc=Interpolations.Flat())
    vinfo("Sensory input dynamics configured: $(si)"; level=2)
    vinfo("  Time span: $(ss.tspan[1]) to $(ss.tspan[2]), $(n_points) sample points"; level=2)
    return net
end

function construct_network_dynamics!(net::Network)::Network
    net.dynamics = Equation[]
    
    for node in net.nodes
        # Process each population to determine if it needs tscale
        for (pop_idx, pop) in enumerate(node.populations)
            # Get parameters for this specific population
            pop_params = ParamSet([p for p in net.params.params 
                                  if p.parent_pop.id == pop.id && 
                                     p.parent_pop.parent_node.id == node.id])
            
            # Check if THIS population needs tscale
            uses_voltage_gated = any(spec -> spec == "voltage_gated_dynamics", 
                                    pop.build_setts.input_dynamics_spec)
            pop_needs_tscale = uses_voltage_gated
            
            # Get equations for this population (based on variable parent_pop)
            pop_eqs = Equation[]
            for eq in node.dynamics
                # Check if any variable in the equation belongs to this population
                # eq.lhs is Differential(t)(variable), so arguments[1] is the variable
                lhs_var = eq.lhs.arguments[1]
                lhs_var_name = string(lhs_var)
                if occursin("₊x$(pop.id)", lhs_var_name)
                    push!(pop_eqs, eq)
                end
            end
            
            if pop_needs_tscale
                # Create population-specific tscale parameter
                tscale_param = Param("$(node.name)₊tscale$(pop.id)", :tscale, pop;
                                    default=10.0, min=0.1, max=1000.0,
                                    tunable=true, 
                                    description="Time scale for pop $(pop.id)")
                
                if !param_in_paramset(net.params, tscale_param)
                    add_param!(net.params, tscale_param)
                end
                
                # Apply tscale to this population's equations
                for eq in pop_eqs
                    push!(net.dynamics, Equation(eq.lhs, tscale_param.symbol * eq.rhs))
                end
            else
                # No scaling needed for this population
                for eq in pop_eqs
                    push!(net.dynamics, Equation(eq.lhs, eq.rhs))
                end
            end
        end
    end
    return net
end

function construct_network_diffusion_dynamics!(net::Network)::Network
    diffusion_eqs = Vector{Equation}()
    for node in net.nodes
        for pop in node.populations
            for (ii, input_dynamics) in enumerate(pop.input_dynamics)
                state_vars = get_state_vars(input_dynamics.vars)
                for state_var in state_vars.vars
                    if eq_idx(state_var) == 0
                        continue
                    elseif eq_idx(state_var) == 1 && length(state_vars.vars) > 1
                        push!(diffusion_eqs, Equation(symbol(state_var), Symbolics.Num(0)))
                    else
                        if pop.build_setts.noise_dynamics_spec[ii] == "" ||
                            lowercase(pop.build_setts.noise_dynamics_spec[ii]) == "none" ||
                            pop.build_setts.noise_dynamics_spec[ii] == "additive"
                            push!(diffusion_eqs, Equation(symbol(state_var), Symbolics.Num(0)))
                        elseif pop.build_setts.noise_dynamics_spec[ii] == "stochastic"
                            noise_rhs::Symbolics.Num = Symbolics.Num(0)
                            
                            if pop.noise_dynamics == "c"
                                highest_constant_idx = get_highest_postfix_index(pop.params; pop_id=pop.id)
                                c = Param("$(pop.parent_node.name)₊c$(pop.id)$(highest_constant_idx + 1)", :noise_std, pop;
                                        tunable=true, description="Controls stochastic noise intensity.")
                                add_param!(net.params, c)
                                noise_rhs = c.symbol
                            elseif tryparse(Float64, pop.noise_dynamics) isa Real
                                parsed_float = tryparse(Float64, pop.noise_dynamics)
                                noise_rhs = Symbolics.Num(parsed_float)
                            else
                                throw(Warning("Unsupported noise dynamics expression: $(pop.noise_dynamics). Setting noise to 0."))
                                noise_rhs = Symbolics.Num(0)
                            end
                            push!(diffusion_eqs, Equation(symbol(state_var), noise_rhs))
                        else
                            throw(ArgumentError("Unsupported noise dynamics type: $(pop.build_setts.noise_dynamics_spec[ii]). Supported types are: 'none', 'additive', 'multiplicative'."))
                        end
                    end
                end
            end
        end
    end
    net.diffusion_dynamics = diffusion_eqs

    if !isempty(diffusion_eqs)
        vinfo("Network diffusion dynamics configured: $(length(diffusion_eqs)) equations"; level=2)
    end
    return net
end

"""Update network parameters: defaults, tunability, and bounds.

This is the main entry point for parameter configuration during network building.
Uses the refactored parameter configuration system for cleaner separation of concerns.

# Arguments
- `net::Network`: Network with parameters to configure
- `settings::Union{Nothing, Settings}`: Optional settings override (uses net.settings if not provided)
"""
function update_network_parameters!(net::Network; 
                                   settings::Union{Nothing, Settings}=nothing)
    sett = something(settings, net.settings)
    
    # Add delay parameters for history variables if not already present
    history_vars = get_history_vars(net.vars)
    if !isempty(history_vars.vars)
        for hv in history_vars.vars
            delay_param = Param("$(hv.delay_param_symbol)", "delay",
                                hv.parent_pop;
                                default=0.1, min=0., max=1000.,
                                isdelay=true, description="Delay for history variable $(hv.symbol)")
            if param_in_paramset(net.params, delay_param)
                vwarn("Delay parameter $(delay_param.name) already exists in network parameters. Skipping addition."; level=2)
                continue
            else
                add_param!(net.params, delay_param)
            end
        end
    end

    # Build node settings list for configure_network_parameters!
    node_settings = []
    for node in net.nodes
        push!(node_settings, Dict(
            :new_param_values => node.build_setts.new_param_values,
            :new_param_tunability => node.build_setts.new_param_tunability
        ))
    end

    # Use new unified parameter configuration system
    configure_network_parameters!(net, sett; node_build_setts=node_settings)
    
    return net
end

function update_network_vars!(net::Network)
    for node in net.nodes
        update_var_inits!(net.vars, node.build_setts.new_state_var_inits)
    end
    return net
end

function construct_network_problem!(net::Network)::Network
    ss = net.settings.simulation_settings;
    gs = net.settings.general_settings;
    delays_flag = isempty(get_history_vars(net.vars).vars) ? false : true
    internal_noise_flag1 = length(filter(eq -> !isequal(eq.rhs, 0), net.diffusion_dynamics)) != 0
    internal_noise_flag2 = has_random_additive_noise(net)
    noise_flag = internal_noise_flag1 || internal_noise_flag2 # || net.sensory_randomness

    eqs_strings = diff_equations_to_strings(net.dynamics)
    rhs_strings = [extract_rhs(eq) for eq in eqs_strings]  # fixed: iterate strings, not chars

    state_vars = get_symbols(get_state_vars(net.vars); sort=true)
    state_vars_symbols = [Symbol(s) for s in state_vars]
    inits = sample_inits(net.vars; subset=state_vars, return_type="vector", sort=true, 
                        seed=net.settings.network_settings.init_seed)
    params_tuple = get_param_default_values(net.params; return_type="named_tuple", sort=true)

    if get_sensory_input_var(net.vars) !== nothing
        sensory_input_var = Symbol(symbol(get_sensory_input_var(net.vars)))
        sensory_input_tuple = (; sensory_input_var => net.sensory_input_func)
        params_tuple = merge(params_tuple, sensory_input_tuple)
    end

    noise_vars = get_vars_by_type(net.vars, "additive_noise")
    if !isempty(noise_vars.vars)
        for noise_var in noise_vars.vars
            noise_symbol = Symbol(symbol(noise_var))
            noise_node = noise_var.parent_pop.parent_node
            noise_pop = noise_var.parent_pop
            noise_tuple = (; noise_symbol => noise_pop.additive_noise_func)
            params_tuple = merge(params_tuple, noise_tuple)
        end
    end

    param_symbols = keys(params_tuple)
    delayed_vars = Tuple{Symbol, Symbol}[]
    for delayed_var in get_history_vars(net.vars).vars
        push!(delayed_vars, (Symbol(symbol(delayed_var)), Symbol(delayed_var.delay_param_symbol)))
    end

    h(p, t) = inits
    diff_strings = alg_equations_to_strings(net.diffusion_dynamics)
    noise_strings = [extract_rhs(eq) for eq in diff_strings]  # fixed: iterate strings, not chars

    if delays_flag && noise_flag
        net.problem = make_sdde_problem_from_strings(rhs_strings, noise_strings, 
            state_vars_symbols, delayed_vars, param_symbols, inits, h, ss.tspan, params_tuple; verbose=verbose[])
    elseif delays_flag && !noise_flag
        net.problem = make_dde_problem_from_strings(rhs_strings, state_vars_symbols, 
            delayed_vars, param_symbols, inits, h, ss.tspan, params_tuple; verbose=verbose[])
    elseif !delays_flag && noise_flag
        net.problem = make_sde_problem_from_strings(rhs_strings, noise_strings, 
            state_vars_symbols, param_symbols, inits, ss.tspan, params_tuple; verbose=verbose[])
    else
        net.problem = make_ode_problem_from_strings(rhs_strings, state_vars_symbols, 
            param_symbols, inits, ss.tspan, params_tuple; verbose=verbose[])
    end

    return net
end

function diff_equations_to_strings(equations::Vector{Equation})
    result = String[]
    
    for eq in equations
        lhs_expr = eq.lhs
        if !(lhs_expr.f isa Symbolics.Differential)
            push!(result, string(eq))
            continue
        end
        
        diff_var = lhs_expr.f.x  # The variable of differentiation (usually t)
        state_var = lhs_expr.arguments  # The state variable being differentiated
        
        rhs_str = string(eq.rhs)
        
        var_name = string(state_var)
        if startswith(var_name, "Any[") && endswith(var_name, "]")
            var_name = var_name[5:end-1]
        else
            var_name = replace(var_name, r"Any\[(.*)\]" => s"\1")
        end
        
        eq_str = "d$(var_name)/d$(diff_var) = $(rhs_str)"
        
        push!(result, eq_str)
    end
    
    return result
end

function alg_equations_to_strings(equations::Vector{Equation})
    result = String[]    
    for eq in equations
        lhs_str = string(eq.lhs)
        rhs_str = string(eq.rhs)        
        push!(result, "$(lhs_str) = $(rhs_str)")
    end    
    return result
end

function extract_rhs(eq_string::String)
    parts = split(eq_string, r"=|~")
    if length(parts) != 2
        error("Invalid equation format: $eq_string")
    end    
    return strip(parts[2])
end

function make_ode_problem_from_strings(rhs_strings, state_vars, param_vars, u0, tspan, p; verbose=false)
    state_assignments = [:($(var) = u[$i]) for (i, var) in enumerate(state_vars)]
    param_assignments = [:($(var) = p.$(var)) for var in param_vars]
    
    eq_assignments = []
    for (i, eq_str) in enumerate(rhs_strings)
        parsed_eq = Meta.parse(eq_str)
        push!(eq_assignments, :(du[$i] = $parsed_eq))
    end
    
    function_expr = quote
        (du, u, p, t) -> begin
            # Assign state variables
            $(state_assignments...)
            # Assign parameters
            $(param_assignments...)
            # Evaluate equations
            $(eq_assignments...)

            return nothing
        end
    end

    drift_f = eval(function_expr)

    # Simple wrapper - tscale is already baked into equations
    f_wrapped = let drift_f = drift_f
        function (du, u, p, t)
            Base.invokelatest(drift_f, du, u, p, t)
            return nothing
        end
    end

    return ODEProblem(f_wrapped, u0, tspan, p)
end

"""
    make_dde_problem_from_strings(drift_strings, state_vars, delayed_vars, param_vars, u0, h, tspan, p; verbose=false)

Creates a DDEProblem from string representations of the drift terms with delayed state variables.
"""
function make_dde_problem_from_strings(drift_strings, state_vars, delayed_vars, param_vars, u0, h, tspan, p; verbose=false)
    
    state_assignments = [:($(var) = u[$i]) for (i, var) in enumerate(state_vars)]
    param_assignments = [:($(var) = p.$(var)) for var in param_vars]
    
    delayed_assignments = []
    unique_delay_param_symbols = Symbol[] 

    for dv_tuple in delayed_vars
        hist_var_sym = dv_tuple[1]    # e.g., :h_y
        delay_param_sym = dv_tuple[2] # e.g., :τ_y

        base_var_name_str = string(hist_var_sym)
        if !startswith(base_var_name_str, "h_")
            error("Delayed variable symbol $(hist_var_sym) must start with 'h_'")
        end
        base_var_name = Symbol(base_var_name_str[3:end])
        
        idx_in_state_vars = findfirst(x -> x == base_var_name, state_vars)
        if idx_in_state_vars === nothing
            error("Base variable $(base_var_name) for history variable $(hist_var_sym) not found in state_vars")
        end

        if !(delay_param_sym in param_vars)
            error("Delay parameter $(delay_param_sym) for history variable $(hist_var_sym) not found in param_vars. Ensure it is defined in `param_vars` and its value is in `p`.")
        end
        
        push!(delayed_assignments, :($(hist_var_sym) = h(p, t - $(delay_param_sym))[$(idx_in_state_vars)]))
        
        if !(delay_param_sym in unique_delay_param_symbols)
            push!(unique_delay_param_symbols, delay_param_sym)
        end
    end
    
    drift_assignments = []
    for (i, eq_str) in enumerate(drift_strings)
        parsed_eq = Meta.parse(eq_str)
        push!(drift_assignments, :(du[$i] = $parsed_eq))
    end
    
    drift_expr = quote
        function f(du, u, h, p, t)
            # Assign state variables
            $(state_assignments...)
            # Assign parameters
            $(param_assignments...)
            # Assign delayed variables
            $(delayed_assignments...)
            # Evaluate deterministic equations
            $(drift_assignments...)
            
            return nothing
        end
    end

    drift_f = eval(drift_expr)

    drift_wrapped = let drift_f = drift_f
        function (du, u, h, p, t)
            Base.invokelatest(drift_f, du, u, h, p, t)
            return nothing
        end
    end

    actual_lags = Float64[] 
    for delay_sym in unique_delay_param_symbols
        param_idx = findfirst(x -> x == delay_sym, param_vars)
        if param_idx === nothing
            error("Delay parameter $(delay_sym) was not found in param_vars when constructing lag values. This indicates an internal logic error.")
        end
        push!(actual_lags, p[param_idx])
    end
    return DDEProblem(drift_wrapped, u0, h, tspan, p; constant_lags=actual_lags)
end

"""
    make_sde_problem_from_strings(drift_strings, diffusion_strings, state_vars, param_vars, u0, tspan, p; verbose=false)

Creates an SDEProblem from string representations of the drift and diffusion terms.
"""
function make_sde_problem_from_strings(drift_strings, diffusion_strings, state_vars, param_vars, u0, tspan, p; verbose=false)
    state_assignments = [:($(var) = u[$i]) for (i, var) in enumerate(state_vars)]
    param_assignments = [:($(var) = p.$(var)) for var in param_vars]
    
    drift_assignments = []
    for (i, eq_str) in enumerate(drift_strings)
        parsed_eq = Meta.parse(eq_str)
        push!(drift_assignments, :(du[$i] = $parsed_eq))
    end
    
    diffusion_assignments = []
    for (i, eq_str) in enumerate(diffusion_strings)
        parsed_eq = Meta.parse(eq_str)
        push!(diffusion_assignments, :(du[$i] = $parsed_eq))
    end
    
    drift_expr = quote
        function f(du, u, p, t)
            # Assign state variables
            $(state_assignments...)
            # Assign parameters
            $(param_assignments...)
            # Evaluate drift equations
            $(drift_assignments...)
            
            return nothing
        end
    end
    
    diffusion_expr = quote
        function g(du, u, p, t)
            # Assign state variables
            $(state_assignments...)
            # Assign parameters
            $(param_assignments...)
            # Evaluate diffusion equations
            $(diffusion_assignments...)
            
            return nothing
        end
    end
    
    drift_f = eval(drift_expr)
    diffusion_g = eval(diffusion_expr)

    drift_wrapped = let drift_f = drift_f
        function (du, u, p, t)
            Base.invokelatest(drift_f, du, u, p, t)
            return nothing
        end
    end

    diffusion_wrapped = let diffusion_g = diffusion_g
        function (du, u, p, t)
            Base.invokelatest(diffusion_g, du, u, p, t)
            return nothing
        end
    end

    return SDEProblem(drift_wrapped, diffusion_wrapped, u0, tspan, p)
end

"""
    make_sdde_problem_from_strings(drift_strings, diffusion_strings, state_vars, delayed_state_vars, param_vars, u0, h, tspan, p; verbose=false)

Creates an SDDEProblem from string representations of the drift and diffusion terms.
"""
function make_sdde_problem_from_strings(drift_strings, diffusion_strings, state_vars, delayed_vars, param_vars, u0, h, tspan, p; verbose=false)
    
    state_assignments = [:($(var) = u[$i]) for (i, var) in enumerate(state_vars)]
    param_assignments = [:($(var) = p.$(var)) for var in param_vars]
    
    delayed_assignments = []
    unique_delay_param_symbols = Symbol[] 

    for dv_tuple in delayed_vars
        hist_var_sym = dv_tuple[1]    # e.g., :h_y
        delay_param_sym = dv_tuple[2] # e.g., :τ_y

        base_var_name_str = string(hist_var_sym)
        if !startswith(base_var_name_str, "h_")
            error("Delayed variable symbol $(hist_var_sym) must start with 'h_'")
        end
        base_var_name = Symbol(base_var_name_str[3:end])
        

        
        idx_in_state_vars = findfirst(x -> x == base_var_name, state_vars)
        if idx_in_state_vars === nothing
            error("Base variable $(base_var_name) for history variable $(hist_var_sym) not found in state_vars")
        end

        if !(delay_param_sym in param_vars)
            error("Delay parameter $(delay_param_sym) for history variable $(hist_var_sym) not found in param_vars. Ensure it is defined in `param_vars` and its value is in `p`.")
        end
        
        push!(delayed_assignments, :($(hist_var_sym) = h(p, t - $(delay_param_sym))[$(idx_in_state_vars)]))
        
        if !(delay_param_sym in unique_delay_param_symbols)
            push!(unique_delay_param_symbols, delay_param_sym)
        end
    end
    
    drift_assignments = []
    for (i, eq_str) in enumerate(drift_strings)
        parsed_eq = Meta.parse(eq_str)
        push!(drift_assignments, :(du[$i] = $parsed_eq))
    end
    
    diffusion_assignments = []
    for (i, eq_str) in enumerate(diffusion_strings)
        parsed_eq = Meta.parse(eq_str)
        push!(diffusion_assignments, :(du[$i] = $parsed_eq))
    end
    
    drift_expr = quote
        function f(du, u, h, p, t)
            # Assign state variables
            $(state_assignments...)
            # Assign parameters
            $(param_assignments...)
            # Assign delayed variables
            $(delayed_assignments...)
            # Evaluate deterministic equations
            $(drift_assignments...)
            
            return nothing
        end
    end
    
    diffusion_expr = quote
        function g(du, u, h, p, t)
            # Assign state variables
            $(state_assignments...)
            # Assign parameters
            $(param_assignments...)
            # Assign delayed variables
            $(delayed_assignments...)
            # Evaluate noise equations
            $(diffusion_assignments...)
            
            return nothing
        end
    end

    drift_f = eval(drift_expr)
    diffusion_g = eval(diffusion_expr)

    drift_wrapped = let drift_f = drift_f
        function (du, u, h, p, t)
            Base.invokelatest(drift_f, du, u, h, p, t)
            return nothing
        end
    end

    diffusion_wrapped = let diffusion_g = diffusion_g
        function (du, u, h, p, t)
            Base.invokelatest(diffusion_g, du, u, h, p, t)
            return nothing
        end
    end

    actual_lags = Float64[] 
    for delay_sym in unique_delay_param_symbols
        param_idx = findfirst(x -> x == delay_sym, param_vars)
        if param_idx === nothing
            error("Delay parameter $(delay_sym) was not found in param_vars when constructing lag values. This indicates an internal logic error.")
        end
        push!(actual_lags, p[param_idx])
    end
    return SDDEProblem(drift_wrapped, diffusion_wrapped, u0, h, tspan, p; constant_lags=actual_lags)
end

# ====== Added: Network structural signature utilities ======

# Deterministic formatting for numeric matrix
_matrix_sig(M::AbstractMatrix{<:Real}) = join([join([@sprintf("%.6g", M[i,j]) for j in 1:size(M,2)], ",") for i in 1:size(M,1)], ";")
# Deterministic formatting for String matrix
_matrix_sig(M::AbstractMatrix{String}) = join([join([M[i,j] == "" ? "none" : M[i,j] for j in 1:size(M,2)], ",") for i in 1:size(M,1)], ";")

function population_signature(pop::Population)
    its  = join(sort(unique(pop.build_setts.input_dynamics_spec)), ",")
    ints = ""
    ots  = pop.build_setts.output_dynamics_spec
    return "P$(pop.id)(I={$its};In={$ints};O={$ots})"
end

function node_signature(n::Node)
    pops_part = join([population_signature(p) for p in n.populations], ",")
    conns_part = "matrix"
    # Convert RuleTree to string for signature if necessary
    model_sig = if n.build_setts.model isa String
        n.build_setts.model
    else
        serialize_rule_tree(n.build_setts.model)
    end
    return "Node$(n.id){model=$(model_sig);pops=[$pops_part];conns=$(conns_part)}"
end

function set_network_signature!(net::Network)::Network
    return net
end

function export_network(net::Network)
    gs = net.settings.general_settings
    ns = net.settings.network_settings
    # Generate filename with network name
    fname_out = "$(net.name)_equations"
    output_dir = construct_output_dir(gs, ns)
    
    if "tex" in gs.save_model_formats
        eqs = vcat(net.dynamics, net.diffusion_dynamics)
        path_tex = joinpath(output_dir, "$(fname_out).tex")
        transform2latex(eqs, show_plot=false, path_tex=path_tex)
        vinfo("LaTeX equations exported to: $path_tex"; level=2)
    end
    if "txt" in gs.save_model_formats
        recipe_raw = net.settings.network_settings.node_models[1]
        # Handle both String and RuleTree node models
        recipe = if recipe_raw isa String
            recipe_raw
        else
            serialize_rule_tree(recipe_raw)
        end
        path_txt = joinpath(output_dir, "$(fname_out).txt")
        open(path_txt, "w") do io
            write(io, "Recipe: $(recipe)\n\n")
        end
        vinfo("Recipe exported to: $path_txt"; level=2)
    end
    return
end
