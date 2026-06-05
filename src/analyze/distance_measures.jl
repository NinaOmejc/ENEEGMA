"""
Distance / dissimilarity utilities between sampled models and known models.

Two notions of equation distance:
1. Histogram / structural overlap via operation label histograms 
2. Token sequence edit distance for canonicalized equations

api: calculate_distances_to_known_models(new_model, known_models; distance_types=[...]) -> NamedTuple of distance vectors
"""

using SymbolicUtils, Symbolics, StringDistances, Distances
const SymbolicEquation = Symbolics.Equation

# --- small helper for safe normalization of scalar distances in [0, ∞) ---
# mirrors `normalize_safe` logic used in check_analysis.jl but for scalars
_normalize_safe_scalar(x::Float64)::Float64 = isfinite(x) ? (x == 0.0 ? 0.0 : x / max(x, 1.0)) : 0.0

# --- Tokenization of recipe strings ---
tokenize_string(s::AbstractString)::Vector{String} = [String(m.match) for m in eachmatch(r"-?\d+", s)]

"""
    _levenshtein_wildcard(ta, tb; wildcard="0") -> Int

Levenshtein edit distance between two token sequences where any token equal to
`wildcard` matches anything at zero cost (substitution, insertion, deletion).

This is used so that recipe positions encoded as rule_id 0 (from `"any"
terminals in `terminals_gsn`) are ignored in distance calculations.
"""
function _levenshtein_wildcard(ta::Vector{String}, tb::Vector{String};
                               wildcard::String="0")::Int
    m, n = length(ta), length(tb)
    # dp[i+1, j+1] = edit distance between ta[1:i] and tb[1:j]
    dp = zeros(Int, m + 1, n + 1)
    # base cases: inserting/deleting wildcards costs 0
    for i in 1:m
        dp[i+1, 1] = dp[i, 1] + (ta[i] == wildcard ? 0 : 1)
    end
    for j in 1:n
        dp[1, j+1] = dp[1, j] + (tb[j] == wildcard ? 0 : 1)
    end
    for i in 1:m
        for j in 1:n
            if ta[i] == tb[j] || ta[i] == wildcard || tb[j] == wildcard
                dp[i+1, j+1] = dp[i, j]          # match or wildcard (0 cost)
            else
                dp[i+1, j+1] = min(
                    dp[i,   j] + 1,               # substitute
                    dp[i+1, j] + 1,               # insert
                    dp[i,   j+1] + 1,             # delete
                )
            end
        end
    end
    return dp[m+1, n+1]
end

function edit_distance_recipe(a::AbstractString, b::AbstractString; method::String="levenshtein")
    ta = tokenize_string(a)
    tb = tokenize_string(b)
    if method == "levenshtein"
        return _levenshtein_wildcard(ta, tb)
    else
        error("Unknown edit distance method: $method")
    end
end

# --- Histogram based distances on equations ---
_isnode(x)  = SymbolicUtils.istree(x)
_args(x)    = SymbolicUtils.arguments(x)
_oplabel(x) = string(SymbolicUtils.operation(x))

"Make a histogram of operation labels from a list of RHS expressions."
function _op_histogram(rhs_exprs)::Dict{String,Int}
    h = Dict{String,Int}()
    for ex in rhs_exprs
        stack = Any[ex]
        while !isempty(stack)
            x = pop!(stack)
            if _isnode(x)
                key = _oplabel(x)
                h[key] = get(h, key, 0) + 1
                append!(stack, _args(x))
            end
        end
    end
    return h
end

"Cosine distance between two histograms."
function _hist_cosine_distance(h1::Dict{String,Int}, h2::Dict{String,Int})
    ks = union(keys(h1), keys(h2))
    v1 = Float64[get(h1, k, 0) for k in ks]
    v2 = Float64[get(h2, k, 0) for k in ks]
    return cosine_dist(v1, v2)
end

"""
    edit_distance_equations(eqsA, eqsB; metric=:cosine)

Histogram-based distance between two sets of equations, comparing RHS operation label counts.
"""
function set_distance_equations(eqsA::Union{Nothing,Vector{SymbolicEquation}},
                                 eqsB::Union{Nothing,Vector{SymbolicEquation}};
                                 metric::Symbol = :cosine)::Float64
    rhsA = (eqsA === nothing) ? Any[] : [eq.rhs for eq in eqsA]
    rhsB = (eqsB === nothing) ? Any[] : [eq.rhs for eq in eqsB]
    hA = _op_histogram(rhsA)
    hB = _op_histogram(rhsB)

    if metric == :cosine
        return _hist_cosine_distance(hA, hB)
    elseif metric == :jaccard
        sA, sB = Set(keys(hA)), Set(keys(hB))
        inter = length(intersect(sA, sB))
        uni   = length(union(sA, sB))
        return uni == 0 ? 0.0 : 1.0 - inter/uni
    elseif metric == :l1
        ks = union(keys(hA), keys(hB))
        v1 = Float64[get(hA, k, 0) for k in ks]
        v2 = Float64[get(hB, k, 0) for k in ks]
        return sum(abs.(v1 .- v2))
    else
        error("Unknown metric: $metric")
    end
end

# --- Canonicalization + token sequence edit distance ---

const COMMUTATIVE_OPS = Set{Any}((+, *))

function _canonical_term_type(ex)
    T = Symbolics.symtype(ex)
    return T === Any ? Real : T
end

function canonicalize_expr(ex)
    if !Symbolics.istree(ex)
        return ex
    end

    op   = Symbolics.operation(ex)
    args = Symbolics.arguments(ex)

    canon_args = map(canonicalize_expr, args)

    if op in COMMUTATIVE_OPS
        sort!(canon_args; by = string)
    end

    return Symbolics.term(op, canon_args...; type=_canonical_term_type(ex))
end

canonicalize_equation(eq::SymbolicEquation) =
    SymbolicEquation(canonicalize_expr(eq.lhs), canonicalize_expr(eq.rhs))

function canonicalize_equations(eqs::Vector{SymbolicEquation})
    ceqs = canonicalize_equation.(eqs)
    # still sort by lhs,rhs for consistent ordering
    sort!(ceqs; by = eq -> (string(eq.lhs), string(eq.rhs)))
    return ceqs
end

function postfix_tokens(ex)::Vector{String}
    if !Symbolics.istree(ex)
        return [string(ex)]
    end
    op   = Symbolics.operation(ex)
    args = Symbolics.arguments(ex)

    toks = String[]
    for a in args
        append!(toks, postfix_tokens(a))
    end
    push!(toks, string(op))
    return toks
end

"""
    rhs_tokens(eq) -> Vector{String}

Return postfix tokens of RHS only.
"""
function rhs_tokens(eq::SymbolicEquation)::Vector{String}
    return postfix_tokens(eq.rhs)
end

"""
    edit_distance_equations_rhs(eqs_a, eqs_b; method="levenshtein", normalize=false)

Distance between two systems of equations as sum of RHS-only token-level edit
distances, plus penalty for extra equations.
"""
function edit_distance_equations_rhs(eqs_a::Vector{SymbolicEquation},
                                     eqs_b::Vector{SymbolicEquation};
                                     method::String="levenshtein",
                                     normalize::Bool=false)

    ca = canonicalize_equations(eqs_a)
    cb = canonicalize_equations(eqs_b)

    ma, mb = length(ca), length(cb)
    k = min(ma, mb)

    dist = 0.0
    total_tokens = 0

    # matched equations
    for i in 1:k
        ta = rhs_tokens(ca[i])
        tb = rhs_tokens(cb[i])

        d = if method == "levenshtein"
            StringDistances.Levenshtein()(ta, tb)
        else
            error("Unknown edit distance method: $method")
        end

        dist += d
        total_tokens += max(length(ta), length(tb))
    end

    # extra equations in A (RHS distance to empty)
    if ma > mb
        for i in (k+1):ma
            ta = rhs_tokens(ca[i])
            d_extra = StringDistances.Levenshtein()(ta, String[])
            dist += d_extra
            total_tokens += length(ta)
        end
    end

    # extra equations in B
    if mb > ma
        for i in (k+1):mb
            tb = rhs_tokens(cb[i])
            d_extra = StringDistances.Levenshtein()(tb, String[])
            dist += d_extra
            total_tokens += length(tb)
        end
    end

    if normalize && total_tokens > 0
        return dist / total_tokens
    else
        return dist
    end
end


"""
    dim_distance_equations(eqsA, eqsB) -> Float64

Distance based purely on the difference in number of equations
(e.g. state variables).

Returns a value in [0, 1], where 0 means same number of equations,
1 means one system is empty and the other non-empty.
"""
function dim_distance_equations(eqsA, eqsB)::Float64
    nA = (eqsA === nothing) ? 0 : length(eqsA)
    nB = (eqsB === nothing) ? 0 : length(eqsB)

    if nA == 0 && nB == 0
        return 0.0
    else
        return abs(nA - nB) / max(nA, nB)
    end
end



# Robust min-max scaling with clamping.
# Maps lo -> 0, hi -> 1, clamps outside.
@inline function _robust_minmax(x::Real, lo::Real, hi::Real)::Float64
    den = (hi - lo)
    den <= 0 && return 0.0
    return Float64(clamp((x - lo) / (den + eps(Float64)), 0.0, 1.0))
end

# Compute (lo, hi) robust bounds (defaults: 5th–95th percentiles)
function _robust_bounds(vals::AbstractVector{<:Real}; qlo=0.05, qhi=0.95)
    isempty(vals) && error("Cannot compute robust bounds on an empty vector.")
    lo = quantile(Float64.(vals), qlo)
    hi = quantile(Float64.(vals), qhi)
    return lo, hi
end


function combined_distance_editcos(
    eqsA::Vector{SymbolicEquation},
    eqsB::Vector{SymbolicEquation};
    edit_lo::Real=0.0, edit_hi::Real=984.0,
    w_edit::Real=0.5, w_set::Real=0.5
)::Float64
    d_edit = edit_distance_equations_rhs(eqsA, eqsB; normalize=false)
    d_set  = set_distance_equations(eqsA, eqsB; metric=:cosine)

    d_edit_norm = _robust_minmax(d_edit, edit_lo, edit_hi)

    return Float64(w_edit * d_edit_norm + w_set * d_set)
end




"""
    hamming_distance(f1::Vector{Int}, f2::Vector{Int}) -> Float64

Normalized Hamming distance between two fixed-length grammar feature vectors.

Interprets all entries as categorical IDs:
- distance = 0.0  if all positions match
- distance = 1.0  if all positions differ
"""
function hamming_distance(f1::Vector{Int}, f2::Vector{Int})::Float64
    @assert length(f1) == length(f2) "Feature vectors must have the same length"

    n = length(f1)
    n == 0 && return 0.0

    mismatches = 0
    @inbounds for i in 1:n
        f1[i] == f2[i] || (mismatches += 1)
    end

    return mismatches / n
end


"""
    compute_distances_to_known_models(new_model, known_models;
        distance_types = [...])

Compute a NamedTuple of distance vectors for a new_model against a list of known_models.

Allowed distance types:
  - edit_distance_recipe
  - set_distance_equations
  - edit_distance_norm
  - edit_distance
  - feature_distance
"""
function compute_distances_to_known_models(
        new_model,
        known_models::Vector;
        distance_types::Vector{String} = ["combined_distance_editcos"])::NamedTuple

    cols = Dict{String, Vector{Float64}}(t => Float64[] for t in distance_types)

    eqs_new    = getfield(new_model, :equations)
    recipe_new = getfield(new_model, :recipe)
    features_new = getfield(new_model, :features)

    for km in known_models
        eqs_k    = getfield(km, :equations)
        recipe_k = getfield(km, :recipe)
        features_k = getfield(km, :features)

        for t in distance_types
            val = if t == "edit_distance_recipe"
                edit_distance_recipe(recipe_new, recipe_k)
            elseif t == "set_distance_equations" || t == "set_distance_cos"
                set_distance_equations(eqs_new, eqs_k; metric=:cosine)
            elseif t == "set_distance_jac"
                set_distance_equations(eqs_new, eqs_k; metric=:jaccard)
            elseif t == "set_distance_l1"
                set_distance_equations(eqs_new, eqs_k; metric=:l1)
            elseif t == "edit_distance_norm"
                edit_distance_equations_rhs(eqs_new, eqs_k; normalize=true)
            elseif t == "edit_distance_equations" || t == "edit_distance"
                edit_distance_equations_rhs(eqs_new, eqs_k; normalize=false)
            elseif t == "geometric_distance"
                geometric_distance(eqs_new, eqs_k)
            elseif t == "feature_distance"
                hamming_distance(features_new, features_k)
            elseif t == "combined_distance_editcos"
                combined_distance_editcos(eqs_new, eqs_k)
            else
                error("Unknown distance type: $t")
            end
            push!(cols[t], val)
        end
    end

    return (; (Symbol(d) => cols[d] for d in distance_types)...)
end
