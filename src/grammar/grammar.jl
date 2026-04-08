# Minimal grammar utilities for:
#   - loading a PCFG from a .cfg file
#   - sampling derivations
#   - working with rule trees
#
# This version does NOT build grammars programmatically.
# It assumes all rules and probabilities are defined in the .cfg file.

using Random
using OrderedCollections: OrderedDict
using Distributions: Categorical

# ----------------------------------
# Core types
# ----------------------------------

"""
A single grammar rule:

    lhs -> rhs [probability]

`lhs` is a nonterminal symbol (String).
`rhs` is either:
  - a single terminal symbol (is_terminal = true), or
  - a space-separated sequence of nonterminals (is_terminal = false).
"""
mutable struct GrammarRule
    rule_id::Int           # integer id (for trees); -1 means "unassigned"
    lhs::String
    rhs::String
    probability::Float64
    is_terminal::Bool
end

# Convenience constructors
GrammarRule(lhs::AbstractString,
            rhs::AbstractString,
            p::Float64,
            is_terminal::Bool=false) =
    GrammarRule(-1, String(lhs), String(rhs), p, is_terminal)

GrammarRule(id::Int,
            lhs::AbstractString,
            rhs::AbstractString,
            p::Float64) =
    GrammarRule(id, String(lhs), String(rhs), p, false)


"""
ParsedRule

A rule that has been applied in a derivation,
annotated with its position in the sequence.
"""
struct ParsedRule
    pos::Int
    rule_id::Int
    lhs::String
    rhs::String
    probability::Float64
    is_terminal::Bool

    function ParsedRule(pos::Int, rule::GrammarRule)
        new(pos, rule.rule_id, rule.lhs, rule.rhs, rule.probability, rule.is_terminal)
    end
end


"""
Grammar

Holds all production rules, grouped by LHS nonterminal.
"""
mutable struct Grammar
    path::AbstractString
    rules::OrderedDict{String, Vector{GrammarRule}}
end

Grammar(file_path::AbstractString) = Grammar(file_path, OrderedDict{String, Vector{GrammarRule}}())


# ----------------------------------
# Utility: normalize probabilities and rule ids
# ----------------------------------

"""
Normalize probabilities for each group of rules with the same lhs.
"""
function normalize_rule_groups!(g::Grammar; drop_zeros::Bool=true)
    for (lhs, rs) in g.rules
        if drop_zeros
            filter!(r -> r.probability > 0.0, rs)
        end
        isempty(rs) && continue
        s = sum(r.probability for r in rs)
        s <= 0 && error("Sum of probabilities for lhs = $lhs is non-positive")
        invs = 1.0 / s
        for r in rs
            r.probability *= invs
        end
    end
    return g
end

"""
Assign integer rule_ids.

Default mode: assign a unique id to every rule (across all lhs),
starting at 0.
"""
function assign_rule_ids!(g::Grammar; start_id::Int=1)
    id = start_id
    for (_, rs) in g.rules
        for r in rs
            r.rule_id = id
            id += 1
        end
    end
    return g
end

"""
Ensure that all rule_ids are assigned (non-negative).
If any rule has rule_id < 0, all ids are reassigned from 0.
"""
function ensure_rule_ids!(g::Grammar; start_id::Int=1)
    needs = any(r.rule_id < 0 for rs in values(g.rules) for r in rs)
    needs && assign_rule_ids!(g; start_id=start_id)
    return g
end


# ----------------------------------
# Loading / saving grammars (.cfg)
# ----------------------------------

"""
    load_grammar(path)::Grammar

Load a grammar from a .cfg file with lines of the form

    LHS -> "terminal" [p] | NonTerm1 NonTerm2 [p] | ...

Terminals must be in double quotes.
Non-terminals are unquoted and separated by spaces.

Empty lines and lines starting with `#` are ignored.
"""
function load_grammar(grammar_path::AbstractString)
    g = Grammar(grammar_path)
    rulemap = OrderedDict{String,Vector{GrammarRule}}()
    isempty(grammar_path) && return g

    pat_line = r"^(\w+)\s*->\s*(.+)$"
    pat_alt  = r"((?:\"[^\"]+\"|[^|])+?)\s*\[(\d*\.?\d+(?:[eE][+-]?\d+)?)\]"

    open(grammar_path, "r") do io
        for ln in eachline(io)
            s = strip(ln)
            isempty(s) && continue
            startswith(s, "#") && continue

            m = match(pat_line, s)
            m === nothing && continue
            lhs, rhs = m.captures

            for alt in split(rhs, "|")
                a = strip(alt)
                m2 = match(pat_alt, a)
                m2 === nothing && continue
                raw_sym = strip(m2.captures[1])
                prob    = parse(Float64, m2.captures[2])

                is_term = startswith(raw_sym, "\"") && endswith(raw_sym, "\"")
                sym     = is_term ? raw_sym[2:end-1] : raw_sym

                vec = get!(rulemap, lhs, GrammarRule[])
                push!(vec, GrammarRule(lhs, sym, prob, is_term))
            end
        end
    end

    g.rules = rulemap
    normalize_rule_groups!(g; drop_zeros=true)
    ensure_rule_ids!(g; start_id=1)
    return g
end


"""
Save a Grammar back to a .cfg file.
(Useful if you have modified probabilities.)
"""
function save_grammar(g::Grammar, path::AbstractString)
    open(path, "w") do io
        for (lhs, rs) in g.rules
            # group rules with same lhs on one line
            alts = String[]
            for r in rs
                rhs_str = r.is_terminal ? "\"$(r.rhs)\"" : r.rhs
                push!(alts, "$(rhs_str) [$(r.probability)]")
            end
            println(io, lhs, " -> ", join(alts, " | "))
        end
    end
    return nothing
end


# ----------------------------------
# Pretty-print rules
# ----------------------------------

"""
List all rules in a grammar, grouped by lhs.
"""
function list_rules(g::Grammar)
    n = sum(length(rs) for rs in values(g.rules))
    if n == 0
        println("Grammar has 0 rules.")
        return
    end
    lhs_set = collect(keys(g.rules))
    println("Grammar has $(length(lhs_set)) non-terminals and $n rules.")
    for (lhs, rs) in g.rules
        println(lhs, ":")
        for r in rs
            rhs_str = r.is_terminal ? "\"$(r.rhs)\"" : r.rhs
            println("  ", lpad(string(r.rule_id), 3), ": ",
                    lhs, " -> ", rhs_str, " [", r.probability, "]")
        end
    end
end

function list_rules(parsed_rules::Vector{ParsedRule})
    for r in parsed_rules
        rhs_str = r.is_terminal ? "\"$(r.rhs)\"" : r.rhs
        println("Pos ", lpad(string(r.pos),3), ": id ",
                lpad(string(r.rule_id),3), ": ",
                r.lhs, " -> ", rhs_str, " [", r.probability, "]")
    end
end


# ----------------------------------
# Sampling
# ----------------------------------

"""
Sample a single rule with given `lhs` from grammar `g`.
Choice is weighted by `probability`.
"""
function sample_rule(g::Grammar, lhs::AbstractString;
                     rng::AbstractRNG = Random.GLOBAL_RNG)
    key = String(lhs)
    rs = get(g.rules, key) do
        error("Nonterminal $(key) not found in grammar.")
    end
    ps = [r.probability for r in rs]
    idx = rand(rng, Categorical(ps))
    return rs[idx]
end


# ----------------------------------
# Rule tree type and utilities
# ----------------------------------

"""
Tree of rule applications used in a derivation.
Each node stores `rule_id` and the child subtrees.
"""
struct RuleTree
    rule_id::Int
    children::Vector{RuleTree}
end

# Full serializer: prints all children, regardless of number.
function serialize_rule_tree_full(t::RuleTree)::String
    isempty(t.children) && return string(t.rule_id)
    return string(t.rule_id, '{',
                  join(serialize_rule_tree_full.(t.children), ","),
                  '}')
end

# Compact serializer: omit braces when a node has a single child.
function serialize_rule_tree(t::RuleTree)::String
    isempty(t.children) && return string(t.rule_id)
    if length(t.children) == 1
        return string(t.rule_id, '{', serialize_rule_tree(t.children[1]), '}')
    else
        return string(t.rule_id, '{',
                      join(serialize_rule_tree.(t.children), ","),
                      '}')
    end
end

Base.show(io::IO, t::RuleTree) = print(io, serialize_rule_tree(t))


"""
    _resolve_output_path(base_dir::AbstractString, filename::AbstractString)

Given a base directory and filename, resolve the full path with collision handling.

If the file already exists, appends "_2", "_3", etc. to the filename stem until
a non-existent path is found.

# Example
```julia
path = _resolve_output_path("results/exp1", "grammar_samples.txt")
# Returns: "results/exp1/grammar_samples.txt" (or "_2", "_3" variant if exists)
```
"""
function _resolve_output_path(base_dir::AbstractString, filename::AbstractString)
    # Ensure base directory exists
    !isdir(base_dir) && mkpath(base_dir)
    
    # Split filename into stem and extension
    path = joinpath(base_dir, filename)
    
    # If file doesn't exist, return it as-is
    !isfile(path) && return path
    
    # File exists, find a non-conflicting name
    parts = splitext(filename)
    stem = parts[1]
    ext = parts[2]
    
    counter = 2
    while true
        new_filename = "$(stem)_$(counter)$(ext)"
        new_path = joinpath(base_dir, new_filename)
        !isfile(new_path) && return new_path
        counter += 1
    end
end


"""
    save_parse_trees(trees::Vector{RuleTree}, filepath::AbstractString)

Save a vector of parse trees to a CSV file with model names and parse_tree strings.

Creates a table with two columns:
- `model_name`: Auto-generated names (G1, G2, ...)
- `parse_tree`: Serialized parse tree in compact string format (e.g., "2{6{12,16,21},25,23}")

# Arguments
- `trees::Vector{RuleTree}`: Vector of parse trees to save
- `filepath::AbstractString`: Output file path (typically .csv)

# Example
```julia
trees = sample_from_grammar(settings)[:parse_tree]
save_parse_trees(trees, "parse_trees.csv")
```
"""
function save_parse_trees(trees::Vector{RuleTree}, filepath::AbstractString)
    model_names = ["G$i" for i in 1:length(trees)]
    parse_trees = [serialize_rule_tree(tree) for tree in trees]
    
    df = DataFrame(
        model_name = model_names,
        parse_tree = parse_trees
    )
    
    # CSV.write automatically quotes fields containing the delimiter (comma)
    # so parse_trees like "1{8{11,16,21,22,58},25,23,34}" become quoted in output:
    # G1,"1{8{11,16,21,22,58},25,23,34}"
    CSV.write(filepath, df)
    return nothing
end


"""
    save_parse_trees(trees::Vector{RuleTree}, settings)

Save a vector of parse trees using settings configuration.

Output file path is constructed as: `settings.general_settings.path_out / settings.general_settings.exp_name / grammar_samples.csv`

Creates a CSV table with columns: model_name (G1, G2, ...) and parse_tree (serialized parse trees).

If the file already exists, "_2", "_3", etc. are appended to avoid overwrites.

# Arguments
- `trees::Vector{RuleTree}`: Vector of parse trees to save
- `settings`: Settings object with `general_settings.path_out` and `general_settings.exp_name`

# Example
```julia
settings = create_default_settings()
trees = sample_from_grammar(settings)[:parse_tree]
save_parse_trees(trees, settings)
```
"""
function save_parse_trees(trees::Vector{RuleTree}, settings)
    gen_s = settings.general_settings
    base_dir = joinpath(gen_s.path_out, gen_s.exp_name)
    filepath = _resolve_output_path(base_dir, "grammar_samples.csv")
    save_parse_trees(trees, filepath)
    return nothing
end


"""
    save_parse_trees(samples_dict::Dict, filepath::AbstractString)

Save parse trees from a grammar sampling result dictionary to a CSV table.

Extracts `:parse_tree` key from the dict returned by `sample_from_grammar()`.
Creates CSV with model_name and parse_tree columns.

# Arguments
- `samples_dict::Dict`: Dictionary returned by `sample_from_grammar()` with `:parse_tree` key
- `filepath::AbstractString`: Output file path (typically .csv)

# Example
```julia
results = sample_from_grammar(settings)
save_parse_trees(results, "parse_trees.csv")
```
"""
function save_parse_trees(samples_dict::Dict, filepath::AbstractString)
    if !haskey(samples_dict, :parse_tree)
        error("samples_dict does not contain :parse_tree key")
    end
    save_parse_trees(samples_dict[:parse_tree], filepath)
    return nothing
end


"""
    save_parse_trees(samples_dict::Dict, settings)

Save parse trees from a grammar sampling result dictionary to a CSV table using settings configuration.

Extracts `:parse_tree` key and uses settings for output directory configuration.
Creates CSV with model_name (G1, G2, ...) and parse_tree columns.

# Arguments
- `samples_dict::Dict`: Dictionary returned by `sample_from_grammar()` with `:parse_tree` key
- `settings`: Settings object with `general_settings.path_out` and `general_settings.exp_name`

# Example
```julia
settings = create_default_settings()
results = sample_from_grammar(settings)
save_parse_trees(results, settings)
```
"""
function save_parse_trees(samples_dict::Dict, settings)
    if !haskey(samples_dict, :parse_tree)
        error("samples_dict does not contain :parse_tree key")
    end
    save_parse_trees(samples_dict[:parse_tree], settings)
    return nothing
end


"""
Parse a serialized rule tree (produced by `serialize_rule_tree` or `_full`)
back into a `RuleTree`.
"""
function parse_rule_tree(s::AbstractString)::RuleTree
    str = replace(s, r"\s+" => "")
    i = Ref(1)

    function parse_node()
        @assert i[] <= lastindex(str) "Unexpected end while parsing"
        # parse integer
        start = i[]
        while i[] <= lastindex(str) && isdigit(str[i[]])
            i[] += 1
        end
        @assert i[] > start "Expected rule id at position $(start)"
        rid = parse(Int, str[start:i[]-1])

        if i[] <= lastindex(str) && str[i[]] == '{'
            i[] += 1
            childs = RuleTree[]
            while i[] <= lastindex(str) && str[i[]] != '}'
                push!(childs, parse_node())
                if i[] <= lastindex(str) && str[i[]] == ','
                    i[] += 1
                end
            end
            @assert i[] <= lastindex(str) && str[i[]] == '}' "Missing closing brace"
            i[] += 1
            return RuleTree(rid, childs)
        else
            return RuleTree(rid, RuleTree[])
        end
    end

    node = parse_node()
    @assert i[] > lastindex(str) || i[] == lastindex(str) + 1 "Trailing characters after tree"
    return node
end


# ----------------------------------
# Mapping rule_id → GrammarRule
# ----------------------------------

"""
Return a Dict mapping rule_id => GrammarRule.
Useful for reconstructing strings from RuleTrees.
"""
function rule_id_map(g::Grammar)
    idmap = Dict{Int,GrammarRule}()
    for rs in values(g.rules)
        for r in rs
            r.rule_id >= 0 && (idmap[r.rule_id] = r)
        end
    end
    return idmap
end

# Collect terminal leaf surface forms from a rule tree (depth-first, left-to-right)
function _tree_to_string(t::RuleTree, idmap::Dict{Int,GrammarRule})
    terms = String[]
    _collect_terms!(terms, t, idmap)
    return join(terms, " ")
end

function _collect_terms!(acc::Vector{String}, t::RuleTree, idmap::Dict{Int,GrammarRule})
    r = idmap[t.rule_id]
    if r.is_terminal
        push!(acc, r.rhs)
    else
        for c in t.children
            _collect_terms!(acc, c, idmap)
        end
    end
end


# ----------------------------------
# Start-symbol inference
# ----------------------------------

"""
Try to infer a start symbol:

  - "Start" if present,
  - else "Node" if present,
  - else the first key in `g.rules`.
"""
function infer_start_symbol(g::Grammar)
    if haskey(g.rules, "Start")
        return "Start"
    elseif haskey(g.rules, "Node")
        return "Node"
    elseif !isempty(g.rules)
        return first(keys(g.rules))
    else
        error("Grammar has no rules; cannot infer start symbol.")
    end
end


# ----------------------------------
# Recursive expansion / sampling
# ----------------------------------

"""
    expand!(g, symbol, rng; sample_dict=nothing, counter=Ref(0))

Recursively expand from `symbol` by sampling rules until only terminals remain.

Returns:
  (generated_string::String,
   sample_dict::OrderedDict{String,String},
   rule_tree::RuleTree)

`sample_dict` uses keys like "0:Node", "1:InputDyn", ... where the part
after ':' is the LHS (rule.lhs) and the value is rule.rhs.
"""
function expand!(g::Grammar, symbol::AbstractString, rng::AbstractRNG; sample_dict=nothing, counter=Ref(0))
    sample_dict = something(sample_dict, OrderedDict{String,String}())

    rule = sample_rule(g, symbol; rng=rng)

    key = "$(counter[]):$(rule.lhs)"
    sample_dict[key] = String(rule.rhs)
    counter[] += 1

    if rule.is_terminal
        return (rule.rhs, sample_dict, RuleTree(rule.rule_id, RuleTree[]))
    end

    child_strings = String[]
    child_trees   = RuleTree[]
    for child_sym in split(rule.rhs, ' ')
        isempty(child_sym) && continue
        if is_terminal(child_sym)
            # Inline operator terminals ("+", "*") or quoted terminals embedded in RHS
            push!(child_strings, child_sym[1] == '"' ? child_sym[2:end-1] : child_sym)
            # Do NOT push a RuleTree node (no rule_id for inline symbol)
        else
            cstr, _, ctree = expand!(g, child_sym, rng; sample_dict=sample_dict, counter=counter)
            push!(child_strings, cstr)
            push!(child_trees, ctree)
        end
    end

    return (join(child_strings, " "), sample_dict, RuleTree(rule.rule_id, child_trees))
end

# If there was an overload expand!(g::Grammar, symbol::SubString{String}, ...) it can now
# just forward to the AbstractString method to avoid duplication:
function expand!(g::Grammar, symbol::SubString{String}, rng::AbstractRNG; sample_dict=nothing, counter=Ref(0))
    return expand!(g, String(symbol), rng; sample_dict=sample_dict, counter=counter)
end


# ----------------------------------
# Sampling interface
# ----------------------------------

# simple RNG helper
make_rng(seed::Union{Int,Nothing}) =
    seed === nothing ? Random.GLOBAL_RNG : Random.MersenneTwister(seed)

# Internal helper (new) to generate one sample.
function _sample_once(g::Grammar, rng::AbstractRNG, start_symbol::String, return_mode::Symbol)
    tree, dict, flat = expand!(g, start_symbol, rng; sample_dict=nothing)
    return return_mode === :parse_tree         ? tree  :
           return_mode === :terminals         ? flat :
           return_mode === :full_rule_expansion ? dict  :
           error("Unknown return_mode=$(return_mode)")
end

# Replace the two large sample_from_grammar method bodies with a shared core.

# Internal helper: normalize requested modes into canonical set.
function _canonical_modes(return_mode::Union{Symbol,AbstractVector{Symbol}})
    canonical(sym::Symbol) = sym in (:model, :models, :full_rule_expansion)   ? :full_rule_expansion  :
                             sym in (:terminal, :terminals, :string, :strings) ? :terminals :
                             sym in (:tree, :trees, :parse_tree)     ? :parse_tree    :
                             error("Unsupported return_mode symbol: $(sym)")
    requested = return_mode isa Symbol ? (return_mode,) :
                return_mode isa AbstractVector{Symbol} ? Tuple(return_mode) :
                error("return_mode must be Symbol or Vector{Symbol}")
    unique(canonical.(requested))
end

# Internal core performing the sampling given a loaded Grammar.
function _sample_from_grammar_core(g::Grammar;
                                   n_samples::Integer,
                                   rng::AbstractRNG,
                                   return_mode::Union{Symbol,AbstractVector{Symbol}},
                                   start_symbol::AbstractString)

    modes = _canonical_modes(return_mode)
    need_full_rule_expansion = :full_rule_expansion in modes
    need_terminals           = :terminals           in modes
    need_parse_tree          = :parse_tree          in modes

    idmap = rule_id_map(g)

    samples_dict      = need_full_rule_expansion ? Vector{OrderedDict{String,String}}(undef, n_samples) : nothing
    samples_terminals = need_terminals           ? Vector{String}(undef, n_samples)                     : nothing
    samples_trees     = need_parse_tree          ? Vector{RuleTree}(undef, n_samples)                   : nothing

    for i in 1:n_samples
        s_str, s_dict, s_tree = expand!(g, start_symbol, rng)
        # Only attempt canonical reconstruction if no inline operator terminals (which are absent from the tree).
        if !(occursin('+', s_str) || occursin('*', s_str))
            recon = _tree_to_string(s_tree, idmap)
            recon != s_str && (s_str = recon)
        end
        need_full_rule_expansion && (samples_dict[i]      = s_dict)
        need_terminals           && (samples_terminals[i] = s_str)
        need_parse_tree          && (samples_trees[i]     = s_tree)
    end

    if length(modes) == 1
        m = first(modes)
        m === :full_rule_expansion && return samples_dict
        m === :terminals           && return samples_terminals
        m === :parse_tree          && return samples_trees
    else
        out = Dict{Symbol,Any}()
        need_full_rule_expansion && (out[:full_rule_expansion] = samples_dict)
        need_terminals           && (out[:terminals]           = samples_terminals)
        need_parse_tree          && (out[:parse_tree]          = samples_trees)
        return out
    end
end

# Public wrapper: already-loaded Grammar.
function sample_from_grammar(g::Grammar;
                             n_samples::Integer=1,
                             seed::Union{Int,Nothing}=nothing,
                             return_mode::Union{Symbol,AbstractVector{Symbol}}=:full_rule_expansion,
                             start_symbol::Union{Nothing,AbstractString}=nothing)
    rng = make_rng(seed)
    start_sym = start_symbol === nothing ? infer_start_symbol(g) : String(start_symbol)
    return _sample_from_grammar_core(g;
        n_samples=n_samples,
        rng=rng,
        return_mode=return_mode,
        start_symbol=start_sym)
end

# Public wrapper: load from file path, then delegate.
function sample_from_grammar(grammar_path::AbstractString;
                             n_samples::Integer=1,
                             seed::Union{Int,Nothing}=nothing,
                             return_mode::Union{Symbol,AbstractVector{Symbol}}=:full_rule_expansion,
                             start_symbol::Union{Nothing,AbstractString}=nothing)
    g = load_grammar(grammar_path)
    return sample_from_grammar(g;
        n_samples=n_samples,
        seed=seed,
        return_mode=return_mode,
        start_symbol=start_symbol)
end

# Public wrapper: Settings-based interface.
# Extracts grammar configuration from Settings object.
# Default: returns dict with :full_rule_expansion, :parse_tree, :terminals keys
#
# Seed priority: 
#   1. sampling_settings.grammar_seed (if not nothing)
#   2. general_settings.seed (fallback if grammar_seed is nothing)
#   3. nothing (random seed)
function sample_from_grammar(settings;
                             return_mode::Union{Symbol,AbstractVector{Symbol}}=[:full_rule_expansion, :parse_tree, :terminals],
                             start_symbol::Union{Nothing,AbstractString}=nothing)
    samp_s = settings.sampling_settings
    if samp_s === nothing
        error("settings.sampling_settings is nothing; cannot access grammar configuration")
    end
    
    # Determine which seed to use: grammar_seed takes priority, fall back to general seed
    grammar_seed = samp_s.grammar_seed
    if grammar_seed === nothing && settings.general_settings !== nothing
        grammar_seed = settings.general_settings.seed
    end
    
    g = load_grammar(samp_s.grammar_file)
    return sample_from_grammar(g;
        n_samples=samp_s.n_samples,
        seed=grammar_seed,
        return_mode=return_mode,
        start_symbol=start_symbol)
end


# ------------------------------------------------------------------
# Recognize inline operator terminals even if quotes were stripped.
const INLINE_OPERATOR_TERMINALS = Set(["+", "*"])

is_terminal(sym::AbstractString) = (
    (startswith(sym, "\"") && endswith(sym, "\"")) ||
    (sym in INLINE_OPERATOR_TERMINALS)
)
