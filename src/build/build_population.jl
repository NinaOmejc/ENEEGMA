function build_population_dynamics(pop::Population, ss::SimulationSettings)
    
    # Build each component of the dynamics
    build_input_dynamics!(pop)
    input_vars = [d.vars for d in pop.input_dynamics]
    input_params = [d.params for d in pop.input_dynamics]
    pop.vars = join_varsets([pop.vars; input_vars...])
    pop.params = join_paramsets([pop.params; input_params...])

    build_output_dynamics!(pop)
    if !(pop.build_setts.output_dynamics_spec in ("none", "direct_readout"))
        remove_current_output_vars!(pop)
        output_vars = [d.vars for d in pop.output_dynamics]
        output_params = [d.params for d in pop.output_dynamics]
        pop.vars = join_varsets([pop.vars; output_vars...])
        pop.params = join_paramsets([pop.params; output_params...])
    end

    build_additive_noise_dynamics!(pop, ss)

    # Enforce consistency between population-level flags and variable-level flags
    enforce_population_flag_consistency!(pop)

    pop.dynamics = vcat(
        [d.dynamics for d in pop.input_dynamics]...,
        [d.dynamics for d in pop.output_dynamics]...
    )
    return pop
end

function build_input_dynamics!(pop::Population)
    
    supported_dynamics = get_input_dynamics()
    # i, iinput_spec = 1, pop.build_setts.input_dynamics_spec[1]
    for (i, iinput_spec) in enumerate(pop.build_setts.input_dynamics_spec)
        if occursin(r"\bx1\b|\bx2\b", iinput_spec)
            iinput_spec = "poly_kernel"
        end
        chosen_dynamics = ENEEGMA.get_dynamics(iinput_spec, supported_dynamics)
        dyn, vars, params = chosen_dynamics(pop, i)
        push!(pop.input_dynamics, InputDynamics(dyn, vars, params))
        pop.n_state_vars += length(dyn)
        
        # Update highest indices
        if !isempty(vars.vars)
            pop.build_setts.highest_var_idx = get_highest_postfix_index(vars; var_idx_only=true)
        end
        if !isempty(params.params)
            pop.build_setts.highest_param_idx = get_highest_postfix_index(params)
        end
    end
    return pop
end


function build_output_dynamics!(pop::Population)

    supported_dynamics = get_output_dynamics()
    chosen_dynamics = get_dynamics(pop.build_setts.output_dynamics_spec, supported_dynamics)
    dyn, vars, params = chosen_dynamics(pop)
    push!(pop.output_dynamics, OutputDynamics(dyn, vars, params))
    pop.n_state_vars += length(dyn)

    # Update highest indices
    if !isempty(vars.vars)
        pop.build_setts.highest_var_idx = get_highest_postfix_index(vars; var_idx_only=true)
    end
    if !isempty(params.params)
        pop.build_setts.highest_param_idx = get_highest_postfix_index(params)
    end

    return pop
end

function get_dynamics(dynamics_spec::String, supported_dynamics::Dict{String, Function})::Function
    if !haskey(supported_dynamics, dynamics_spec)
        throw(ArgumentError("Invalid dynamics type: $(dynamics_spec). 
            Supported types are: $(keys(supported_dynamics))."))
    end
    return supported_dynamics[dynamics_spec]
end



function remove_current_output_vars!(pop::Population)
    # Remove the current inter-population output variable if it exists
    # This is necessary to avoid conflicts with the new output dynamics

    current_interpop_output_vars = get_vars_sending_interpop_output(pop.vars)
    for var in current_interpop_output_vars.vars
        var.sends_interpop_output = false
    end

    current_internode_output_vars = get_vars_sending_internode_output(pop.vars)
    for var in current_internode_output_vars.vars
        var.sends_internode_output = false
    end
end


function build_additive_noise_dynamics!(pop::Population, ss::SimulationSettings)

    for noise_spec in pop.build_setts.noise_dynamics_spec
        if noise_spec == "additive" && !isempty(pop.noise_dynamics)
            try
                n_points = Int((ss.tspan[2]-ss.tspan[1]) * (1/ss.saveat))
                t_values = range(ss.tspan[1], ss.tspan[2], length=n_points)
                noise_dynamics_expr_func_str = "t -> " * pop.noise_dynamics
                noise_func = eval(Meta.parse(noise_dynamics_expr_func_str))
                noise_dynamics_values = [Base.invokelatest(noise_func, it) for it in t_values]
                pop.additive_noise_func = LinearInterpolation(t_values, noise_dynamics_values, extrapolation_bc=Line())
            catch e
                throw(ArgumentError("Error processing additive noise expression '$(pop.noise_dynamics)' for pop $(pop.id) in node $(pop.parent_node.name): $e"))
            end
        end
    end
    return pop
end

# Ensure vars obey population-level flags
function enforce_population_flag_consistency!(pop::Population)
    for v in pop.vars.vars
        if v isa StateVar
            if !pop.build_setts.sends_internode_output && v.sends_internode_output
                v.sends_internode_output = false
            end
            if !pop.build_setts.gets_internode_input && v.gets_internode_input
                v.gets_internode_input = false
            end
            if !pop.build_setts.gets_sensory_input && v.gets_sensory_input
                v.gets_sensory_input = false
            end
        end
    end
    return pop
end
