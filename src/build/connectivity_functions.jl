function get_conn_funcs(output_type::Symbol=:dict)
    d = Dict(
        "" => skip_conn,
        "none" => skip_conn,
        "false" => skip_conn,
        "linear" => linear,
        "piecewise_linear" => piecewise_linear,
        "baseline_sigmoid" => baseline_sigmoid,
        "saturating_sigmoid" => saturating_sigmoid,
        "relaxed_rectifier" => relaxed_rectifier,
        "fourier_basis" => fourier_basis,
        "tanh_sigmoid" => tanh_sigmoid,
    )
    return output_type === :dict   ? d :
           output_type === :keys   ? collect(keys(d)) :
           output_type === :values ? collect(values(d)) :
           throw(ArgumentError("Unsupported output_type=$(output_type). Use :dict, :keys, or :values."))
end

# Helper function to build input term with optional connectivity parameter
function build_input_term!(pop, conn_vars, highest_constant_idx, n_params, pc_params; add_c::Bool=true)
    input_term = Num(0)
    iconst_idx = 1
    for v in conn_vars.vars
        if v isa ExtraVar && v.is_sensory_input
            si_fun = symbol(v)
            input_term += si_fun(t)
        elseif v isa ExtraVar
            input_term += symbol(v)
        else
            if add_c  
                c = Param("$(pop.parent_node.name)₊c$(pop.id)$(highest_constant_idx + n_params + iconst_idx)", :population_coupling, pop;
                    tunable=true, description="connectivity strength, inner")
                add_param!(pc_params, c)
                input_term += c.symbol * symbol(v)
                iconst_idx += 1
            else
                input_term += symbol(v)
            end
        end
    end
    return input_term, pc_params
end


function skip_conn(pop::Population, conn_vars::VarSet)::Tuple{Num, ParamSet}
    return Num(0), ParamSet()
end

function linear(pop::Population, conn_vars::VarSet; add_c=true)::Tuple{Num, ParamSet}
    highest_constant_idx = get_highest_postfix_index(pop.params; pop_id=pop.id)
    pc_params = ParamSet()
    n_params = 0
    input_term, pc_params = build_input_term!(pop, conn_vars, highest_constant_idx, n_params, pc_params; add_c=add_c)
    return input_term, pc_params
end


function saturating_sigmoid(pop::Population, conn_vars::VarSet)::Tuple{Num, ParamSet}
    highest_constant_idx = get_highest_postfix_index(pop.params; node_id=pop.parent_node.id, pop_id=pop.id)
    c1 = Param("$(pop.parent_node.name)₊c$(pop.id)$(highest_constant_idx + 1)", :gain, pop;
        tunable=true, description="e0, the maximum output value; or S_max in LileyWright (e0 = Smax/2)")
    c2 = Param("$(pop.parent_node.name)₊c$(pop.id)$(highest_constant_idx + 2)", :rate, pop;
        tunable=true, description="r, the steepness of the sigmoid; or 1/σ in LileyWright (r = sqrt(2)/σ)")
    c3 = Param("$(pop.parent_node.name)₊c$(pop.id)$(highest_constant_idx + 3)", :potential, pop;
        tunable=true, description="θ, the midpoint of the sigmoid; or μ in LileyWright (θ = μ) - can be negative")
    c = Param("$(pop.parent_node.name)₊c$(pop.id)$(highest_constant_idx + 4)", :population_coupling, pop; 
        tunable=true, description="conn strength")
    pc_params = ParamSet([c1, c2, c3, c])
    n_params = length(pc_params.params)
    input_term, pc_params = build_input_term!(pop, conn_vars, highest_constant_idx, n_params, pc_params; add_c=true)
    
    pc_term = c.symbol * c1.symbol / (1 + exp(c2.symbol * (c3.symbol - input_term)))
    
    return pc_term, pc_params
end


function baseline_sigmoid(pop::Population, conn_vars::VarSet)::Tuple{Num, ParamSet}
    highest_constant_idx = get_highest_postfix_index(pop.params; node_id=pop.parent_node.id, pop_id=pop.id)
    c1 = Param("$(pop.parent_node.name)₊c$(pop.id)$(highest_constant_idx + 1)", :gain, pop;
        tunable=true, description="a, the maximum output value")
    c2 = Param("$(pop.parent_node.name)₊c$(pop.id)$(highest_constant_idx + 2)", :potential, pop;
        tunable=true, description="θ, the midpoint of the sigmoid")
    c3 = Param("$(pop.parent_node.name)₊c$(pop.id)$(highest_constant_idx + 3)", :population_coupling, pop; 
        tunable=true, default=1., description="conn strength")
    pc_params = ParamSet([c1, c2, c3])
    n_params = length(pc_params.params)
    input_term, pc_params = build_input_term!(pop, conn_vars, highest_constant_idx, n_params, pc_params; add_c=true)
    
    pc_term = c3.symbol*(1 / (1 + exp(-c1.symbol * (input_term - c2.symbol))) - (1 / (1 + exp(c1.symbol * c2.symbol))))
    
    return pc_term, pc_params
end

function relaxed_rectifier(pop::Population, conn_vars::VarSet)::Tuple{Num, ParamSet}
    highest_constant_idx = get_highest_postfix_index(pop.params; node_id=pop.parent_node.id, pop_id=pop.id)

    # Parameters: a (gain), b (threshold), d (steepness)
    c1 = Param("$(pop.parent_node.name)₊c$(pop.id)$(highest_constant_idx + 1)", :gain, pop;
        tunable=true, description="a, input gain")
    c2 = Param("$(pop.parent_node.name)₊c$(pop.id)$(highest_constant_idx + 2)", :offset, pop;
        tunable=true, description="b, input threshold")
    c3 = Param("$(pop.parent_node.name)₊c$(pop.id)$(highest_constant_idx + 3)", :rate, pop;
        tunable=true, description="d, nonlinearity steepness")

    pc_params = ParamSet([c1, c2, c3])
    n_params = length(pc_params.params)

    input_term, pc_params = build_input_term!(pop, conn_vars, highest_constant_idx, n_params, pc_params; add_c=true)

    # Let u = a * x - b
    u = c1.symbol * input_term - c2.symbol

    # Define H(x) = u / (1 - exp(-d * u))
    pc_term = u / (1 - exp(-c3.symbol * u))

    return pc_term, pc_params
end

function piecewise_linear(pop::Population, conn_vars::VarSet)::Tuple{Num, ParamSet}
    highest_constant_idx = get_highest_postfix_index(pop.params; node_id=pop.parent_node.id, pop_id=pop.id)

    f1_conn(target_var, input_term) = ifelse(
        target_var < 0,
        0.0,
        -target_var*input_term
    )
  
    pc_params = ParamSet()
    n_params = 0
    input_term, pc_params = build_input_term!(pop, conn_vars, highest_constant_idx, n_params, pc_params; add_c=false)

    target_var = symbol(get_vars_getting_interpop_input(pop.vars).vars[1])
    
    pc_term = f1_conn(target_var, input_term)

    return pc_term, pc_params
end

function fourier_basis(pop::Population, conn_vars::VarSet)::Tuple{Num, ParamSet}
    # Truncated Fourier series for two phase variables φ1, φ2:
    # F = ω + Σ a_m sin(n1*φ1 + n2*φ2) + Σ b_m cos(n1*φ1 + n2*φ2)
    phase_syms = [symbol(v) for v in conn_vars.vars]
    if length(phase_syms) < 2
        highest_constant_idx = get_highest_postfix_index(pop.params; pop_id=pop.id)
        pc_params = ParamSet()
        n_params = 0
        input_term, pc_params = build_input_term!(pop, conn_vars, highest_constant_idx, n_params, pc_params; add_c=true)
        return input_term, pc_params
    end

    φ1, φ2 = phase_syms[1], phase_syms[2]

    # Ensure we use expressions, not callable symbolic functions
    call_if_fn(x) = try x(t) catch _ x end
    φ1e, φ2e = call_if_fn(φ1), call_if_fn(φ2)

    # Harmonic combinations (n1, n2)
    harmonics = (
        (1,0), (0,1), (2,0), (0,2),
        (1,1), (1,-1),
        (1,2), (1,-2),
        (2,1), (2,-1),
        (2,2), (2,-2)
    )

    highest_constant_idx = get_highest_postfix_index(pop.params; node_id=pop.parent_node.id, pop_id=pop.id)
    params = Param[]
    next = 1

    # Intrinsic frequency ω
    ω = Param("$(pop.parent_node.name)₊c$(pop.id)$(highest_constant_idx + next)", :frequency,
              pop; tunable=true, description="Intrinsic frequency ω")
    push!(params, ω); next += 1

    series = ω.symbol
    for (n1, n2) in harmonics
        a = Param("$(pop.parent_node.name)₊c$(pop.id)$(highest_constant_idx + next)", :gain,
              pop; tunable=true, description="Sine coefficient a_(n1=$(n1),n2=$(n2))")
        push!(params, a); next += 1
        b = Param("$(pop.parent_node.name)₊c$(pop.id)$(highest_constant_idx + next)", :gain,
              pop; tunable=true, description="Cosine coefficient b_(n1=$(n1),n2=$(n2))")
        push!(params, b); next += 1
        arg = n1*φ1e + n2*φ2e
        series += a.symbol * sin(arg) + b.symbol * cos(arg)
    end

    pc_params = ParamSet(params)
    return series, pc_params
end

function tanh_sigmoid(pop::Population, conn_vars::VarSet)::Tuple{Num, ParamSet}
    # sigmoid(x, T, δ)  = 0.5 * (1 + tanh((x - T)/δ))       # generic tanh sigmoids
    highest_constant_idx = get_highest_postfix_index(pop.params; node_id=pop.parent_node.id, pop_id=pop.id)
    c1 = Param("$(pop.parent_node.name)₊c$(pop.id)$(highest_constant_idx + 1)", :potential, pop;
        tunable=true, description="T, the midpoint of the sigmoid")
    c2 = Param("$(pop.parent_node.name)₊c$(pop.id)$(highest_constant_idx + 2)", :rate, pop;
        tunable=true, description="δ, the steepness of the sigmoid")
    pc_params = ParamSet([c1, c2])
    n_params = length(pc_params.params)

    input_term, pc_params = build_input_term!(pop, conn_vars, highest_constant_idx, n_params, pc_params; add_c=false)
    pc_term = 0.5 * (1 + tanh((input_term - c1.symbol) / c2.symbol))
    return pc_term, pc_params
end
