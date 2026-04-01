function get_output_dynamics(output_type::Symbol=:dict)
    d = Dict(
        "" => skip_output_dynamics,
        "none" => skip_output_dynamics,
        "direct_readout" => skip_output_dynamics,
        "spatial_gradient" => spatial_gradient_dynamics,
        "membrane_integrator" => membrane_integrator,
        "difference" => difference_dynamics,
    )
    return output_type === :dict   ? d :
           output_type === :keys   ? collect(keys(d)) :
           output_type === :values ? collect(values(d)) :
           throw(ArgumentError("Unsupported output_type=$(output_type). Use :dict, :keys, or :values."))
end

function skip_output_dynamics(pop::Population)
    return Vector{Equation}(), VarSet(), ParamSet()
end

function spatial_gradient_dynamics(pop::Population)
    highest_var_idx = pop.build_setts.highest_var_idx

    x1 = StateVar(; name="$(pop.parent_node.name)₊x$(pop.id)$(highest_var_idx + 1)", eq_idx=1, parent_pop=pop,
                   description="State variable x$(highest_var_idx + 1) for spatial gradient dynamics",
                   sends_internode_output=pop.build_setts.sends_internode_output,
                   sends_interpop_output=true)
    x2 = StateVar(; name="$(pop.parent_node.name)₊x$(pop.id)$(highest_var_idx + 2)", eq_idx=2, parent_pop=pop,
                   description="State variable x$(highest_var_idx + 2) for spatial gradient dynamics")
    output_dynamics_vars = VarSet([x1, x2])

    input_vars = get_vars_sending_interpop_output(pop.vars)
    if isempty(input_vars.vars)
        error("No inter-population output variable found to connect to output dynamics in population $(pop.name)")
    end

    pc_term, pc_params = build_pop_conn_dynamics(pop, input_vars, "saturating_sigmoid")

    highest_constant_idx = pop.build_setts.highest_param_idx + length(pc_params.params)
    c1 = Param("$(pop.parent_node.name)₊c$(pop.id)$(highest_constant_idx + 1)", :rate,
               pop; description="Rate parameter γ for spatial gradient dynamics")
    output_dynamics_params = join_paramsets([pc_params, ParamSet([c1])])
    
    output_dynamics = [
        D(symbol(x1)) ~ symbol(x2),
        D(symbol(x2)) ~ c1.symbol^2 * (-symbol(x1) + pc_term) - 2*c1.symbol*symbol(x2)
    ]

    return (output_dynamics, output_dynamics_vars, output_dynamics_params)
end



# synaptic_reversal_potential
function Ψ(input::Num, h_r::Num, highest_constant_idx::Int, pop::Population)
    c1 = Param("$(pop.parent_node.name)₊c$(pop.id)$(highest_constant_idx + 1)", :potential, pop; description="X -> Y reversal potential (can be negative)")
    ψ_params = ParamSet([c1])
    ψ_term = (c1.symbol - input) / abs(c1.symbol - h_r)
    
    return ψ_term, ψ_params
end

function membrane_integrator(pop::Population)

    # Find the current highest state variable and create a new one
    highest_var_idx = pop.build_setts.highest_var_idx

    x1 = StateVar(; name="$(pop.parent_node.name)₊x$(pop.id)$(highest_var_idx + 1)", 
                   eq_idx=1, parent_pop=pop,
                   description="State variable V for membrane dynamics", 
                   sends_internode_output=pop.build_setts.sends_internode_output,
                   sends_interpop_output=true)
    output_dynamics_vars = VarSet([x1])
    #=     
    # Update the population variables from the input dynamics, so that none of them sends internode output
    current_output_vars = get_vars_sending_internode_output(pop.vars)
    if !isempty(current_output_vars.vars)
        current_output_vars.vars[1].sends_internode_output = false
    end
    =#

    # Find the current highest parameter
    output_dynamics_params = ParamSet()
    highest_constant_idx = pop.build_setts.highest_param_idx
    c1 = Param("$(pop.parent_node.name)₊c$(pop.id)$(highest_constant_idx + 1)", :time_constant, pop; description="Membrane time constant")
    c2 = Param("$(pop.parent_node.name)₊c$(pop.id)$(highest_constant_idx + 2)", :potential, pop; description="Resting potential (can be negative)")
    output_dynamics_params = join_paramsets([output_dynamics_params, ParamSet([c1, c2])])

    # Build the input term
    input_vars = get_vars_by_eq_idx(pop.vars, 1)
    Ψ_terms = []
    for v in input_vars.vars
        highest_constant_idx = get_highest_postfix_index(join_paramsets([pop.params, output_dynamics_params]); node_id=pop.parent_node.id, pop_id=pop.id)
        Ψ_term, Ψ_params = Ψ(symbol(x1), c2.symbol, highest_constant_idx, pop)
        output_dynamics_params = join_paramsets([output_dynamics_params, Ψ_params])
        push!(Ψ_terms, symbol(v) * Ψ_term)
    end

    # Define the internal dynamics equations
    output_dynamics = [
        D(symbol(x1)) ~ 1/c1.symbol * (c2.symbol - symbol(x1) + sum(Ψ_terms))
        ]

    return (output_dynamics, output_dynamics_vars, output_dynamics_params)
end


function difference_dynamics(pop::Population)
    # Define the difference dynamics (for MDF)

    x1 = StateVar(; name="$(pop.parent_node.name)₊x$(pop.id)$(pop.build_setts.highest_var_idx+1)", eq_idx=1, parent_pop=pop, 
                   sends_internode_output=pop.build_setts.sends_internode_output, 
                   sends_interpop_output=true)
    
    output_dynamics_vars = VarSet([x1])
    output_dynamics_params = ParamSet()

    # Select, from each input dynamics block, the variable that sends inter-pop output
    input_vars = Vector{Var}()
    for iinput in pop.input_dynamics
        vars_out = get_vars_sending_intrapop_output(iinput.vars).vars
        if isempty(vars_out)
            vars_out = get_vars_sending_interpop_output(iinput.vars).vars
        end
        push!(input_vars, vars_out[1])
    end

    if length(input_vars) != 2
        throw(ArgumentError("Difference dynamics requires exactly two input dynamics to operate on; found $(length(input_vars))."))
    end

    # Subtract the symbolic values of the input variables
    output_dynamics = [D(symbol(x1)) ~ symbol(input_vars[1]) - symbol(input_vars[2])]

    return (output_dynamics, output_dynamics_vars, output_dynamics_params)
end

