"""
    _parse_from(g, symbol, terminals, idx)

Internal helper for terminals2rules.

Tries to parse from nonterminal `symbol`, starting at `terminals[idx]`,
and returns either `nothing` or `(tree, new_idx)`.
"""
function _parse_from(g::Grammar,
                     symbol::AbstractString,
                     terminals::Vector{String},
                     idx::Int)

    lhs = String(symbol)
    rs = get(g.rules, lhs, nothing)
    rs === nothing && return nothing   # symbol not in grammar

    # Wildcard: "any" matches the first terminal rule for this nonterminal.
    # It is serialized as rule_id 0 (sentinel) in the recipe string so that
    # edit_distance_recipe can treat it as a zero-cost match against anything.
    if idx <= length(terminals) && terminals[idx] == "any"
        first_term = findfirst(r -> r.is_terminal, rs)
        first_term !== nothing && return (RuleTree(0, RuleTree[]), idx + 1)
    end

    for r in rs
        if r.is_terminal
            # Leaf rule: must match exactly one terminal
            if idx <= length(terminals) && terminals[idx] == r.rhs
                return (RuleTree(r.rule_id, RuleTree[]), idx + 1)
            else
                continue
            end
        else
            # Nonterminal rule: walk through RHS tokens
            child_syms = split(r.rhs, ' ')
            children = RuleTree[]
            cur_idx = idx
            ok = true

            for cs in child_syms
                isempty(cs) && continue

                if is_terminal(cs)
                    # Inline terminal in RHS (e.g. "\"+\"" or "\"*\"")
                    op = cs[1] == '"' ? cs[2:end-1] : cs
                    if cur_idx <= length(terminals) && terminals[cur_idx] == op
                        cur_idx += 1
                    else
                        ok = false
                        break
                    end
                else
                    # Normal nonterminal: recurse
                    res = _parse_from(g, cs, terminals, cur_idx)
                    if res === nothing
                        ok = false
                        break
                    end
                    (subtree, new_idx) = res
                    push!(children, subtree)
                    cur_idx = new_idx
                end
            end

            if ok
                return (RuleTree(r.rule_id, children), cur_idx)
            end
        end
    end

    return nothing
end



"""
    terminals2rules(g::Grammar,
                    terminal_list::Vector{String};
                    start_symbol::AbstractString = infer_start_symbol(g),
                    full::Bool = false) :: String

Given a grammar `g` and an ordered list of `terminal_list` (strings),
reconstruct a rule tree that could have produced them and return a
serialized tree string.

- `start_symbol` (default: inferred; usually "Node" for your grammars)
- `full = false` → uses `serialize_rule_tree` (compact)
- `full = true`  → uses `serialize_rule_tree_full` (always prints braces)
"""
function terminals2rules(g::Grammar,
                         terminal_list::Vector{String};
                         start_symbol::AbstractString = infer_start_symbol(g),
                         full::Bool = false) :: String

    isempty(terminal_list) && error("terminal_list is empty.")

    # Start parsing from the chosen start symbol
    res = _parse_from(g, start_symbol, terminal_list, 1)
    res === nothing && error("Could not parse terminals with given grammar.")

    (tree, next_idx) = res

    # We require that the entire terminal list is consumed
    if next_idx != length(terminal_list) + 1
        error("Parsing did not consume all terminals: stopped at index $(next_idx-1) of $(length(terminal_list)).")
    end

    return full ? serialize_rule_tree_full(tree) : serialize_rule_tree(tree)
end

#= 

# Example usage:

using ENMEEG
grammar_path = "D:\\Experiments\\ENMEEG-Lab\\grammars\\grammar_gs.cfg"
g = load_grammar(grammar_path)

terms = [
    "exp_kernel",
    "direct_readout",
    "custom",
    "custom",
    "exp_kernel",
    "direct_readout",
    "custom",
    "false",
    "baseline_sigmoid",
    "custom",
    "custom",
    "full",
]

tree_str = terminals2rules(g, terms; start_symbol="Node")
tree_str_full = terminals2rules(g, terms; start_symbol="Node", full=true)
println(tree_str)
# e.g. "2{6{12,16,21,21},6{12,16,21,22},25,23,23,36}"
 =#