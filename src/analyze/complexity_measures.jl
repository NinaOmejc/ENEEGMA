"""
Complexity measure utilities for ENMEEG model equations.

API
- compute_model_complexity(model; complexity_measures=["num_op", "num_rules"]) -> NamedTuple
    Returns a NamedTuple with any of the keys: :num_ops, :tree_depth, :num_rules
"""

using SymbolicUtils


# Example: heuristic operator weights for a weighted complexity measure
const OP_WEIGHTS = Dict{String,Float64}(
    # basic arithmetic
    "+"     => 1.0,
    "-"     => 1.0,
    "*"     => 1.0,
    "/"     => 2.0,
    "^"     => 2.0,   # treat x^2, x^3 as moderately heavier than *
    "neg"   => 0.5,   # if you ever encode unary minus explicitly

    # simple nonlinearity / shaping
    "abs"   => 2.0,

    # transcendental / saturating
    "exp"   => 6.0,
    "log"   => 6.0,
    "sin"   => 6.0,
    "cos"   => 6.0,
    "tanh"  => 8.0,
    "σ"     => 10.0,  # logistic; adjust to taste

    # fallback for other smooth functions if you map them to a name
    "nonlinear" => 8.0,
)

function weighted_opcount(ex)
    if !SymbolicUtils.istree(ex)
        return 0.0
    end
    op  = string(SymbolicUtils.operation(ex))
    w   = get(OP_WEIGHTS, op, 1.0)  # default 1.0
    return w + sum(weighted_opcount, SymbolicUtils.arguments(ex))
end

"Compute complexity metrics for a network or model that has an `equations` field and a `recipe` string."
function compute_model_complexity(model; complexity_measures::Vector{String}=["num_ops", "num_rules", "tree_depth", "num_params"])::NamedTuple
    eqs = getfield(model, :equations)
    eqs === nothing && error("compute_model_complexity: model.equations is nothing")
    rhs = [eq.rhs for eq in eqs]
    n_eqs = length(eqs)

    ist = SymbolicUtils.istree
    args = SymbolicUtils.arguments

    # count all internal nodes (operations/function calls); leaves contribute 0
    function opcount(ex)
        ist(ex) ? (1 + sum(opcount, args(ex))) : 0
    end

    # tree depth: leaves have depth 1
    function treedepth(ex)
        ist(ex) ? (1 + maximum(treedepth.(args(ex)); init=0)) : 1
    end
    
    complexities = NamedTuple()
    for measure in complexity_measures
        
        if measure == "num_ops"
            nops = sum(opcount.(rhs))
            complexities = merge(complexities, (num_ops=nops,))
            complexities = merge(complexities, (num_ops_norm=nops/n_eqs,))
        
        elseif measure == "tree_depth"
            depths = sum(treedepth.(rhs))
            complexities = merge(complexities, (tree_depth=depths,))
            complexities = merge(complexities, (tree_depth_norm=depths/n_eqs,))
        
        elseif measure == "num_rules"
            nnums = length(collect(m for m in eachmatch(r"\d+", getfield(model, :recipe))))
            complexities = merge(complexities, (num_rules=nnums,))
        
        elseif measure == "num_wops"
            wops = sum(weighted_opcount.(rhs))
            complexities = merge(complexities, (num_wops=wops,))
            complexities = merge(complexities, (num_wops_norm=wops/n_eqs,))
        
        elseif measure == "num_vars" || measure == "num_eqs"
            complexities = merge(complexities, (num_eqs=n_eqs,))
        
        elseif measure == "num_params"
            # Count unique parameter identifiers (e.g. N1₊c11) across all equations.
            param_tokens = String[]
            for irhs in rhs
                append!(param_tokens, [m.match for m in eachmatch(r"N[0-9]+₊c[0-9A-Za-z_]+", string(irhs))])
            end
            nparams = length(unique(param_tokens))
            complexities = merge(complexities, (num_params=nparams,))
        
        else
            error("Unknown complexity measure: $measure")
        end
    end
    
    return complexities
end
