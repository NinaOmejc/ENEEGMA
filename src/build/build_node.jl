function build_nodes!(net::Network)::Network
    netsett = net.settings.network_settings
    simsett = net.settings.simulation_settings
    sampsett = net.settings.sampling_settings

    canonical_node_models = list_canonical_node_models()
    nodes = Vector{Node}(undef, netsett.n_nodes)
    # node_id, node_model =1, netsett.node_models[1]

    for (node_id, node_model) in enumerate(netsett.node_models)
        node = Node(node_id, netsett.node_names[node_id], node_model; node_coordinates=netsett.node_coords[node_id])

        # Convert RuleTree to string representation for model checking
        model_str = if node_model isa String
            node_model
        else
            # RuleTree: will be handled by configure_node_model!
            serialize_rule_tree(node_model)
        end
        
        if any(occursin(model_str, canonical_node) for canonical_node in canonical_node_models)
            pops, node = get_canonical_node_model_info!(node)
        else
            pops, node = configure_node_model!(node, sampsett)
        end
        node.populations = [build_population_dynamics(pop, simsett) for pop in pops]
        node.n_pops = length(node.populations)
        nodes[node_id] = build_node_dynamics!(node)
    end

    net.nodes = nodes
    net.vars = join_varsets([n.vars for n in nodes])
    net.params = join_paramsets([n.params for n in nodes])

    print_network_nodes_info(net)
    return net
end


function build_node_dynamics!(node::Node)::Node
    additional_node_vars = VarSet()

    for ip = 1:node.n_pops
        pop = get_pop_by_id(node, ip)
        
        # --- Group inputs ---
        input_groups = Dict{Int, Dict{String, VarSet}}()
        _group_interpop_inputs!(input_groups, node, pop)
        _group_sensory_inputs!(input_groups, additional_node_vars, pop)
        _group_internode_inputs!(input_groups, additional_node_vars, pop)

        # --- Build and substitute all grouped connection terms ---
        # target_input_idx, groups, conn_fun, conn_vars = 1, input_groups[1], "fourier_basis",input_groups[1]["fourier_basis"]   # avoid "unused variable" warnings
        substitutions = Dict{Num, Num}()
        for (target_input_idx, groups) in input_groups
            total_term_for_input = Num(0)
            for (conn_fun, conn_vars) in groups
                if !isempty(conn_vars.vars)
                    pc_term, pc_params = build_pop_conn_dynamics(pop, conn_vars, conn_fun)
                    total_term_for_input += pc_term
                    join_paramsets!(pop.params, [pc_params])
                end
            end
            
            placeholder_var_name = "$(pop.parent_node.name)₊inputs$(target_input_idx)"
            placeholder_var = get_var_by_name(pop.vars, placeholder_var_name)
            substitutions[symbol(placeholder_var)] = total_term_for_input
        end

        # Apply all substitutions at once
        if !isempty(substitutions)
            for i in eachindex(pop.dynamics)
                pop.dynamics[i] = Symbolics.substitute(pop.dynamics[i], substitutions)
            end
        end

        # --- Add additive noise directly to target state equations (after substitution) ---
        if pop.build_setts.noise_dynamics_spec == "additive"
            # Register one additive noise var at node scope (for binding in construct_network_problem!)
            noise_var = ExtraVar(; name="$(pop.parent_node.name)₊n$(pop.id)", type="additive_noise", eq_idx=0, parent_pop=pop, is_additive_noise=true)
            add_var!(additional_node_vars, noise_var)
            noise_fun = string2symbolicfun(name(noise_var))

            # For every equation D(x) ~ rhs, if x.gets_additive_noise => rhs += noise_fun(t)
            for i in eachindex(pop.dynamics)
                eq = pop.dynamics[i]
                # Extract the lhs state variable symbol from D(x)
                lhs_arg = eq.lhs.arguments
                lhs_sym = lhs_arg isa AbstractVector ? Num(lhs_arg[1]) : Num(lhs_arg)
                v = get_var_by_symbol(pop.vars, lhs_sym)
                if v !== nothing && v.gets_additive_noise
                    pop.dynamics[i] = Equation(eq.lhs, eq.rhs + noise_fun(t))
                end
            end
        elseif pop.build_setts.noise_dynamics_spec == "stochastic"
            # Stochastic noise handled at the solver level; no changes needed her
        end
    end

    node.dynamics = [eq for pop in node.populations for eq in pop.dynamics]
    node.vars = join_varsets([[pop.vars for pop in node.populations]..., additional_node_vars])
    node.params = join_paramsets([[pop.params for pop in node.populations]...])
    return node
end


 "Helper to group inter-population connections."
function _group_interpop_inputs!(input_groups, node::Node, pop::Population)
    # 1) Per-pop input counts (number of input dynamics blocks)
    n_pops = node.n_pops
    inputs_per_pop = [length(get_pop_by_id(node, pid).input_dynamics) for pid in 1:n_pops]
    n_rows = sum(inputs_per_pop)

    # 2) Infer n_cols from flattened pop_conn length
    total = length(node.build_setts.pop_conn)
    @assert total % n_rows == 0 "pop_conn length ($total) not divisible by total target rows ($n_rows)"
    n_cols = div(total, n_rows)

    # 3) Reshape into a matrix (column-major, consistent with Julia’s reshape)
    pop_conn_matrix = reshape(node.build_setts.pop_conn, n_rows, n_cols)

    # 4) Build row and column index maps:
    #    - rows: each row is a target input slot; map to (target_pop_id, local_input_idx)
    #    - cols: map each column to a source population id
    # Row map: [ (pop_id, local_input_idx) for all target slots in pop order ]
    row_map = Tuple{Int,Int}[]
    for pid in 1:n_pops
        for k in 1:inputs_per_pop[pid]
            push!(row_map, (pid, k))
        end
    end

    # Column map:
    # If grammar encoded columns per population, n_cols == n_pops and this is trivial.
    # If grammar mirrored the row expansion (square n_rows == n_cols), a natural scheme
    # is to repeat each population id by its number of input slots.
    col_map = Int[]
    if n_cols == n_pops
        append!(col_map, 1:n_pops)
    elseif n_cols == n_rows
        for pid in 1:n_pops
            append!(col_map, fill(pid, inputs_per_pop[pid]))
        end
    else
        error("Unsupported conn matrix shape: $(n_rows)×$(n_cols). Expected columns to be n_pops ($(n_pops)) or n_rows ($(n_rows)).")
    end
    @assert length(col_map) == n_cols

    # 5) Determine the row range for the CURRENT target population
    # Compute the prefix sum offsets per pop to get the starting row for each pop
    row_offsets = cumsum([0; inputs_per_pop[1:end-1]])
    my_start = row_offsets[pop.id] + 1
    my_len = inputs_per_pop[pop.id]
    my_rows = my_start:(my_start + my_len - 1)

    # 6) Group sources for each target input slot (local index 1..my_len)
    for (i, r) in enumerate(my_rows)  # i is local target_input_idx; r is global row index
        
        # Ensure the nested dict exists for this target input slot
        get!(input_groups, i, Dict{String, VarSet}())

        for c in 1:n_cols
            conn_fun = pop_conn_matrix[r, c]
            if conn_fun != "none"
                source_pop_id = col_map[c]
                source_pop = get_pop_by_id(node, source_pop_id)
                source_vars = get_vars_sending_interpop_output(source_pop.vars)
                
                get!(input_groups[i], conn_fun, VarSet())
                for source_var in source_vars.vars
                    add_var!(input_groups[i][conn_fun], source_var)
                end
            end
        end
    end
end

"Helper to group sensory inputs."
function _group_sensory_inputs!(input_groups, additional_node_vars, pop::Population)
    for (target_input_idx, sensory_conn_func) in enumerate(pop.build_setts.sensory_conn_func)
        if sensory_conn_func != "none"
            si = ExtraVar(; name="SI", type="sensory_input", eq_idx=0, parent_pop=pop, is_sensory_input=true)
            add_var!(additional_node_vars, si)
            get!(input_groups, target_input_idx, Dict{String, VarSet}())
            get!(input_groups[target_input_idx], sensory_conn_func, VarSet())
            add_var!(input_groups[target_input_idx][sensory_conn_func], si)
        end
    end
end

"Helper to group internode inputs."
function _group_internode_inputs!(input_groups, additional_node_vars, pop::Population)
    for (target_input_idx, internode_conn_func) in enumerate(pop.build_setts.internode_conn_func)
        if internode_conn_func != "none"
            ni = ExtraVar(; name="NI$(pop.id)", type="internode_input", eq_idx=0, parent_pop=pop)
            add_var!(additional_node_vars, ni)
            get!(input_groups, target_input_idx, Dict{String, VarSet}())
            get!(input_groups[target_input_idx], internode_conn_func, VarSet())
            add_var!(input_groups[target_input_idx][internode_conn_func], ni)
        end
    end
end

function build_pop_conn_dynamics(pop::Population, conn_vars::VarSet, conn_fun::String)::Tuple{Num, ParamSet}

    supported_conn_funcs = get_conn_funcs()

    if !haskey(supported_conn_funcs, conn_fun)
        throw(ArgumentError("Invalid transformation dynamics: $(conn_fun). 
                            Supported types are: $(keys(supported_conn_funcs))."))
    end

    func = supported_conn_funcs[conn_fun]
    pc_term, pc_params = func(pop, conn_vars)

    return pc_term, pc_params
end


# -------------------- Grammar-driven node configuration --------------------


function configure_node_model!(node::Node, sampsett::SamplingSettings)::Tuple{Vector{Population}, Node}
    
    # Load grammar from the grammar_file path (already complete path)
    grammar = load_grammar(sampsett.grammar_file)
    ensure_rule_ids!(grammar)
    # list_rules(grammar)
    
    # Convert RuleTree model to string if necessary
    model_str = if node.build_setts.model isa String
        node.build_setts.model
    else
        serialize_rule_tree(node.build_setts.model)
    end
    
    # Parse rule ids into parsed rules
    model_rule_ids = [parse(Int, m.match) for m in eachmatch(r"\d+", model_str)]
    parsed_rules = Vector{ParsedRule}(undef, length(model_rule_ids))
    for (rule_pos, rule_id) in enumerate(model_rule_ids)
        chosen_rule = if rule_id == 0
            # Wildcard sentinel emitted by terminals2rules for "any" terminals.
            # The grammar slot is irrelevant for this model's dynamics (e.g. CF
            # for single-pop models whose pop_conn is "none").  Substitute the
            # first terminal rule of the CF nonterminal so that downstream
            # lookups (_get_parsed_rule_by_lhs "CF") succeed with a valid lhs.
            cf_rules = get(grammar.rules, "CF", GrammarRule[])
            cf_term_idx = findfirst(r -> r.is_terminal, cf_rules)
            cf_term_idx !== nothing ? cf_rules[cf_term_idx] :
                error("configure_node_model!: no terminal rule found for CF (cannot resolve wildcard)")
        else
            _find_rule_by_id(grammar, rule_id)
        end
        parsed_rules[rule_pos] = ParsedRule(rule_pos, chosen_rule)
    end
    # list_rules(parsed_rules)

    node.n_pops = length(_get_parsed_rule_by_lhs(parsed_rules, "Pop")) 
    pops = Vector{Population}(undef, node.n_pops)
    for ip = 1:node.n_pops
        pops[ip] = _set_pop_from_grammar(parsed_rules, ip, node)
    end

    # Build pop connectivity
    (node.build_setts.pop_conn, node.build_setts.pop_conn_motif) = 
        _set_pop_conn_from_grammar(parsed_rules)
    
    return pops, node
end


function _get_parsed_rule_by_lhs(parsed_rules::Vector{ParsedRule}, lhs::String; order::Int=0)::Union{Vector{ParsedRule}, ParsedRule}
    # Support exact LHS match, and pattern groups for "Pop" (PopN) and "Pops" (PopsN)
    matched_rules = if lhs == "Pop"
        [r for r in parsed_rules if occursin(r"^Pop\d+$", r.lhs)]
    elseif lhs == "Pops"
        [r for r in parsed_rules if occursin(r"^Pops\d+$", r.lhs)]
    else
        [r for r in parsed_rules if r.lhs == lhs]
    end
    if order > 0
        return matched_rules[order]
    else
        return matched_rules
    end
end


function _set_pop_from_grammar(parsed_rules::Vector{ParsedRule}, ip::Int, node::Node)::Population

    # Resolve connection functions (CF/SCF) with custom fallback
    resolveCF(x::AbstractString, custom) = (x == "custom" || isempty(x)) ? custom : x
    customCF = ENEEGMA._get_parsed_rule_by_lhs(parsed_rules, "CF"; order=1).rhs
    scf_rules = ENEEGMA._get_parsed_rule_by_lhs(parsed_rules, "SCF")  # ::Vector{ParsedRule}
    get_scf_rhs(i) = i <= length(scf_rules) ? scf_rules[i].rhs : "custom"

    # Pop block: locate the ip-th PopN rule in the parsed derivation
    pop_name = "P$(ip)"
    pop_rule = ENEEGMA._get_parsed_rule_by_lhs(parsed_rules, "Pop"; order=ip)
    pop_rule_pos_start = pop_rule.pos
    pop_rule_pos_end = try
        ENEEGMA._get_parsed_rule_by_lhs(parsed_rules, "Pop"; order=ip+1).pos
    catch
        ENEEGMA._get_parsed_rule_by_lhs(parsed_rules, "CF"; order=1).pos
    end
    pop_parsed_rules = parsed_rules[pop_rule_pos_start:pop_rule_pos_end-1]


    # Get dynamics & store population info in kwargs
    kwargs = Dict{Symbol,Any}()

    # Inputs
    input_dynamics_rules = _get_parsed_rule_by_lhs(pop_parsed_rules, "InputDyn")
    kwargs[:input_dynamics_spec] = String[]

    for idyn in input_dynamics_rules
        if idyn.rhs == "S S"
            if isempty(_get_parsed_rule_by_lhs(pop_parsed_rules, "V"))
                push!(kwargs[:input_dynamics_spec], "poly_kernel")
            else
                poly_parsed_rules_start = idyn.pos
                poly_parsed_rules = parsed_rules[poly_parsed_rules_start:end]

                polys = _build_input_polynomials(poly_parsed_rules)
                push!(kwargs[:input_dynamics_spec], polys)
            end
        else
            push!(kwargs[:input_dynamics_spec], idyn.rhs)
        end
    end

    # Output dynamics
    od = _get_parsed_rule_by_lhs(pop_parsed_rules, "OutputDyn", order=1)
    kwargs[:output_dynamics_spec] = od.rhs

    # Collect MCFs in this pop block (order matters)
    pop_mcfs = [r.rhs for r in pop_parsed_rules if r.lhs == "ECF"]
    to_conn = x -> lowercase(x) == "custom" ? customCF : x
    mapped = to_conn.(pop_mcfs)

    if length(mapped) == 2
        kwargs[:sensory_conn_func]   = mapped[1]
        kwargs[:internode_conn_func] = mapped[2]
    elseif length(mapped) == 4
        kwargs[:sensory_conn_func]   = mapped[1:2]
        kwargs[:internode_conn_func] = mapped[3:4]
    else
        error("Unexpected number of ECF rules ($(length(pop_mcfs))) in Pop$(ip) block.")
    end

    # Stochastic noise
    noise = _get_parsed_rule_by_lhs(pop_parsed_rules, "SN"; order=1).rhs
    if noise == "true"
        kwargs[:noise_dynamics_spec] = "stochastic"
        kwargs[:noise_dynamics] = "c"
    end

    return Population(ip, pop_name, node; kwargs...)
end


function _set_pop_conn_from_grammar(parsed_rules::Vector{ParsedRule})::Tuple{Vector{String}, String}
    customCF = _get_parsed_rule_by_lhs(parsed_rules, "CF"; order=1).rhs
    node_rule = _get_parsed_rule_by_lhs(parsed_rules, "Node"; order=1)
    pop_scf = [rule for rule in _get_parsed_rule_by_lhs(parsed_rules, "SCF")]
    n_conns = length(pop_scf)
    out = String[]

    node_rhs = node_rule.rhs

    if occursin("CM", node_rhs)
        # include connectivity motifs
        seed = nothing

        if occursin("CM1", node_rhs)
            # single-pop connectivity: "null" or "full"
            motif_name = _get_parsed_rule_by_lhs(parsed_rules, "CM1"; order=1).rhs

        elseif occursin("CM2", node_rhs)
            # 2×2 connectivity: "full", "ring", or Digit → c2_*
            cm2_rule = _get_parsed_rule_by_lhs(parsed_rules, "CM2"; order=1)
            motif = cm2_rule.rhs

            if motif == "Digit" || occursin("Digit", motif)
                # CM2 -> Digit -> "k"  ⇒ motif :c2_k
                digit_rules = _get_parsed_rule_by_lhs(parsed_rules, "Digit")
                digit_rule = digit_rules[end]              # there should be exactly one
                motif_name = "c2_" * digit_rule.rhs
            else
                # CM2 -> "full" | "ring"
                motif_name = motif
            end

        else
            # old CM: e.g. "ring", "star", "small_world" Digit Digit, ...
            motif = _get_parsed_rule_by_lhs(parsed_rules, "CM"; order=1).rhs
            if occursin("Digit", motif)
                # seeded random motifs (small_world / scale_free)
                digit_rules = _get_parsed_rule_by_lhs(parsed_rules, "Digit")
                # concatenate all digit RHSs like "3","7" → "37"
                seed = parse(Int, join([r.rhs for r in digit_rules]))
                motif_name = match(r"^\"([^\"]+)\"", motif).captures[1]
            else
                motif_name = motif
            end
        end

        pop_conns = conn_mask(motif_name, n_conns; seed=seed)

        for (i, conn) in enumerate(pop_conns)
            if conn == 0
                push!(out, "none")
            else
                # pick the nearest preceding SCF for this column
                scf_idx = fld(i - 1, n_conns) + 1
                pop_cf = pop_scf[scf_idx].rhs == "linear" ? "linear" : customCF
                push!(out, pop_cf)  # "linear" or resolved customCF
            end
        end

    else
        # legacy MCF-based connectivity
        n_conns = length(collect(eachmatch(r"\bMCF\b", node_rule.rhs)))
        pop_conns = [rule for rule in _get_parsed_rule_by_lhs(parsed_rules, "MCF")][end-n_conns+1:end]

        for mcf in pop_conns
            if lowercase(mcf.rhs) == "false"
                push!(out, "none")
            else
                # pick the nearest preceding SCF for this column
                scf_idx = findlast(r -> r.pos < mcf.pos, pop_scf)
                pop_cf = pop_scf[scf_idx].rhs == "linear" ? "linear" : customCF
                push!(out, pop_cf)  # "linear" or resolved customCF
            end
        end
    end

    return out, motif_name
end


function _find_rule_by_id(grammar::Grammar, rule_id::Int)::GrammarRule
    for group_rules in values(grammar.rules)
        for rule in group_rules
            if rule.rule_id == rule_id
                return rule
            end
        end
    end
    error("No rule found with ID: $rule_id")
end

"""
_build_input_polynomials(parsed_rules) -> String

Given a parsed derivation that includes an InputDyn like "S S",
expand each S into a polynomial over terminals x1/x2 with + and * only,
following the chosen rules in `parsed_rules`.

This consumes S/P/V rules in the order they appear (by `pos`) and
supports any number of S tokens in InputDyn (e.g., "S", "S S S", ...).
"""
function _build_input_polynomials(parsed_rules::Vector{ParsedRule})::String
    # 1) Find the InputDyn rule and its RHS nonterminals (e.g. ["S","S"])
    input_dyn_rule = first(r for r in parsed_rules if r.lhs == "InputDyn")
    roots = split(input_dyn_rule.rhs)  # e.g. ["S", "S"]

    # 2) Collect rules by LHS, sorted by pos (derivation order)
    fam_rules = Dict{String,Vector{ParsedRule}}(
        "S" => sort([r for r in parsed_rules if r.lhs == "S"]; by = r -> r.pos),
        "P" => sort([r for r in parsed_rules if r.lhs == "P"]; by = r -> r.pos),
        "V" => sort([r for r in parsed_rules if r.lhs == "V"]; by = r -> r.pos),
    )

    # 3) Pointers for how many rules we’ve consumed per LHS (shared across all S roots)
    ptr = Dict(lhs => 1 for lhs in keys(fam_rules))

    # Local helper to consume the next rule for a given LHS
    function next_rule(lhs::String)
        lst = get(fam_rules, lhs, ParsedRule[])
        i   = get(ptr, lhs, 1)
        i > length(lst) && error("Ran out of rules for $lhs during expansion.")
        ptr[lhs] = i + 1
        return lst[i]
    end

    # Parenthesize only when needed (to keep precedence when multiplying sums)
    needs_paren(s::String) = occursin('+', s)
    paren_if_needed(s::String) = needs_paren(s) ? "($s)" : s

    # Recursive expansion for S, P, V
    function expand_S()::String
        r = next_rule("S")
        if occursin('+', r.rhs)          # grammar form like: S -> P "+" S
            left  = expand_P()
            right = expand_S()
            return string(left, " + ", right)
        else                             # grammar form like: S -> P
            return expand_P()
        end
    end

    function expand_P()::String
        r = next_rule("P")
        if occursin('*', r.rhs)          # grammar form like: P -> V "*" P
            left  = expand_P()
            right = expand_V()
            return string(paren_if_needed(left), " * ", paren_if_needed(right))
        else                             # grammar form like: P -> V
            return expand_V()
        end
    end

    function expand_V()::String
        r = next_rule("V")
        # In your rules, rhs is just "x1" or "x2"
        return r.rhs
    end

    # 4) Expand each root symbol in InputDyn
    out = String[]
    for sym in roots
        sym == "S" || error("Unexpected InputDyn symbol '$sym' (expected S).")
        push!(out, expand_S())
    end

    return join(out, ", ")
end


# -------------------- Printing helpers --------------------

# Print formatted info about each node, its dynamics, variables, and parameters
function print_network_nodes_info(net::Network)
    is_verbose(1) || return
    total_nodes = length(net.nodes)
    total_pops = sum(length(node.populations) for node in net.nodes)
    vinfo("Network $(net.name): $(total_nodes) nodes, $(total_pops) populations"; level=2)
    for node in net.nodes
        node_states = max(length(node.vars.vars), sum(max(pop.n_state_vars, length(pop.dynamics)) for pop in node.populations))
        # Convert RuleTree to string for display if necessary
        model_display = if node.build_setts.model isa String
            node.build_setts.model
        else
            serialize_rule_tree(node.build_setts.model)
        end
        vinfo("  - $(node.name) ($model_display): pops=$(node.n_pops), states=$(node_states)"; level=2)
        for pop in node.populations
            state_count = pop.n_state_vars > 0 ? pop.n_state_vars : length(pop.dynamics)
            input_spec = isempty(pop.build_setts.input_dynamics_spec) ? "none" : join(unique(pop.build_setts.input_dynamics_spec), "/")
            output_spec = pop.build_setts.output_dynamics_spec
            conn_spec = pop.build_setts.input2output_conn_func
            inputs = String[]
            pop.build_setts.gets_sensory_input && push!(inputs, "sensory")
            pop.build_setts.gets_internode_input && push!(inputs, "internode")
            input_desc = isempty(inputs) ? "no external inputs" : join(inputs, "/")
            vinfo("      - Pop $(pop.id): states=$(state_count) input=$(input_spec) output=$(output_spec) conn=$(conn_spec) ($(input_desc))"; level=2)
            if is_verbose(2)
                vinfo("         vars=$(length(pop.vars.vars)) params=$(length(pop.params.params)) dyn=$(length(pop.dynamics))"; level=2)
            end
        end
    end
end

