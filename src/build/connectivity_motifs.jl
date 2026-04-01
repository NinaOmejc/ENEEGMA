# connectivity_motifs.jl
#
# Connectivity motif registries + helpers.
#
# Grammar tokens supported:
#   Deterministic:
#     'null' | 'full' | 'chain' | 'ring' | 'star' | 'star_tail' | 'two_modules'
#     | 'ei_block' | 'multi_cycle' | 'star_feedback_tail' | 'star_loop_extended'
#     | 'ei_extended' | all c2_* (2×2 digit motifs)
#   Seeded:
#     'erdos_renyi' Digit Digit
#     | 'small_world' Digit Digit
#     | 'scale_free'  Digit Digit
#
# Extend with:
#   add_conn_motif(:my_motif, mask)
#   add_conn_motif_builder(:my_family, n -> mask)
#   add_random_conn_motif(:my_rand, (n, rng; kwargs...) -> mask)
#
# Use:
#   conn_mask("star_tail", 5)                  # deterministic
#   conn_mask("erdos_renyi", 4; seed=37)       # seeded
# Discover:
#   list_conn_motifs()                         # lists registered names

using Random
using Graphs: DiGraph, is_weakly_connected
############################
# motif registries         #
############################

# Deterministic motifs: builder(n::Int; kwargs...) -> Bool[n,n]
const CONN_MOTIF_BUILDERS = Dict{Symbol, Function}()

# Seeded motifs: builder(n::Int, rng::AbstractRNG; kwargs...) -> Bool[n,n]
const RANDOM_CONN_MOTIF_BUILDERS = Dict{Symbol, Function}()

############################
# user-facing registration #
############################

"""
    add_conn_motif(name::Symbol, mask::AbstractMatrix{Bool})

Register a *fixed-size* deterministic motif.
For this motif, `conn_mask(name, n)` will only work when `n` matches `size(mask,1)`.
"""
function add_conn_motif(name::Symbol, mask::AbstractMatrix{Bool})
    n1, n2 = size(mask)
    @assert n1 == n2 "Motif mask must be square, got $(size(mask))"

    CONN_MOTIF_BUILDERS[name] = function (n::Int; kwargs...)
        n == n1 || error("Motif $name is defined for n=$n1, but got n=$n")
        return copy(mask)
    end

    return name
end

"""
    add_conn_motif_builder(name::Symbol, builder::Function)

Register a *family* of deterministic motifs that can depend on `n`.

The builder must have signature:
    builder(n::Int; kwargs...) -> Bool[n,n]
"""
function add_conn_motif_builder(name::Symbol, builder::Function)
    CONN_MOTIF_BUILDERS[name] = builder
    return name
end

"""
    add_random_conn_motif(name::Symbol, builder::Function)

Register a seeded / random motif family.

The builder must have signature:
    builder(n::Int, rng::AbstractRNG; kwargs...) -> Bool[n,n]

`conn_mask(name, n; seed=...)` will construct a local `MersenneTwister(seed)`
and call this builder.
"""
function add_random_conn_motif(name::Symbol, builder::Function)
    RANDOM_CONN_MOTIF_BUILDERS[name] = builder
    return name
end
###########################
# only for 2 by 2 matrices
###########################

############################
# 2×2 digit motifs         #
############################

# Helper: enforce this family is 2×2-only
const C2_ERR = "c2_* motifs are only defined for n = 2"

function c2_0_mask(n::Int)
    n == 2 || error("$C2_ERR (got n = $n)")
    m = falses(n, n)
    # [0 1;
    #  0 0]
    m[1, 2] = true
    return m
end

function c2_1_mask(n::Int)
    n == 2 || error("$C2_ERR (got n = $n)")
    m = falses(n, n)
    # [0 1;
    #  0 1]
    m[1, 2] = true
    m[2, 2] = true
    return m
end

function c2_2_mask(n::Int)
    n == 2 || error("$C2_ERR (got n = $n)")
    m = falses(n, n)
    # [1 1;
    #  0 0]
    m[1, 1] = true
    m[1, 2] = true
    return m
end

function c2_3_mask(n::Int)
    n == 2 || error("$C2_ERR (got n = $n)")
    m = falses(n, n)
    # [1 1;
    #  0 1]
    m[1, 1] = true
    m[1, 2] = true
    m[2, 2] = true
    return m
end

function c2_4_mask(n::Int)
    n == 2 || error("$C2_ERR (got n = $n)")
    m = falses(n, n)
    # [0 0;
    #  1 0]  pure 2 → 1
    m[2, 1] = true
    return m
end

function c2_5_mask(n::Int)
    n == 2 || error("$C2_ERR (got n = $n)")
    m = falses(n, n)
    # [1 1;
    #  1 0]
    m[1, 1] = true
    m[1, 2] = true
    m[2, 1] = true
    return m
end

function c2_6_mask(n::Int)
    n == 2 || error("$C2_ERR (got n = $n)")
    m = falses(n, n)
    # [0 1;
    #  1 1]
    m[1, 2] = true
    m[2, 1] = true
    m[2, 2] = true
    return m
end

function c2_7_mask(n::Int)
    n == 2 || error("$C2_ERR (got n = $n)")
    m = falses(n, n)
    # [1 0;
    #  1 0]
    m[1, 1] = true
    m[2, 1] = true
    return m
end

function c2_8_mask(n::Int)
    n == 2 || error("$C2_ERR (got n = $n)")
    m = falses(n, n)
    # [0 0;
    #  1 1]
    m[2, 1] = true
    m[2, 2] = true
    return m
end

function c2_9_mask(n::Int)
    n == 2 || error("$C2_ERR (got n = $n)")
    m = falses(n, n)
    # [1 0;
    #  1 1]
    m[1, 1] = true
    m[2, 1] = true
    m[2, 2] = true
    return m
end


############################
# basic deterministic masks #
############################

null_mask(n::Int) = falses(n, n)
full_mask(n::Int) = trues(n, n)

function chain_mask(n::Int)
    m = falses(n, n)
    for i in 1:(n-1)
        m[i, i+1] = true
    end
    return m
end

function ring_mask(n::Int)
    m = chain_mask(n)
    m[n, 1] = true
    return m
end

function star_mask(n::Int)
    m = falses(n, n)
    # node 1 is hub; bidirectional with all others
    for j in 2:n
        m[1, j] = true
        m[j, 1] = true
    end
    return m
end

function star_tail_mask(n::Int)
    m = falses(n, n)

    if n <= 1
        # single node: no clear hub/tail; leave it isolated
        return m
    elseif n == 2
        # approximate: just a tiny hub+tail → symmetric pair
        m[1, 2] = true
        m[2, 1] = true
        return m
    else
        # original definition
        # hub = 1, connected bidirectionally to 2..(n-1)
        for j in 2:(n-1)
            m[1, j] = true
            m[j, 1] = true
        end
        # tail from n-1 -> n
        m[n-1, n] = true
        return m
    end
end

function two_modules_mask(n::Int)
    m = falses(n, n)
    n <= 1 && return m

    # simple split into two modules A and B
    k = max(1, n ÷ 2)  # A = 1..k, B = k+1..n
    A = 1:k
    B = (k+1):n

    # dense within each module (no self if you prefer)
    for i in A, j in A
        if i != j
            m[i, j] = true
        end
    end
    for i in B, j in B
        if i != j
            m[i, j] = true
        end
    end

    # sparse A -> B links
    for i in A, j in B
        m[i, j] = true
    end

    return m
end


function ei_block_mask(n::Int)
    m = falses(n, n)
    n <= 1 && return m

    # split into E (first half) and I (second half)
    nE = max(1, n ÷ 2)
    nI = n - nE
    E = 1:nE
    I = (nE+1):n

    # E->E dense (no self)
    for i in E, j in E
        if i != j
            m[i, j] = true
        end
    end

    # I->I sparse (chain)
    if nI >= 2
        for i in first(I):(last(I)-1)
            m[i, i+1] = true
        end
    end

    # E->I dense
    for i in E, j in I
        m[i, j] = true
    end

    # I->E sparse (back to first excitatory)
    for i in I
        m[i, first(E)] = true
    end
    return m
end



function multi_cycle_mask(n::Int)
    m = falses(n, n)

    if n <= 1
        # nothing meaningful, leave isolated
        return m
    elseif n == 2
        # tiny "cycle": 1 ↔ 2
        m[1, 2] = true
        m[2, 1] = true
        return m
    elseif n == 3
        # 3-cycle: 1 -> 2 -> 3 -> 1
        m[1, 2] = true
        m[2, 3] = true
        m[3, 1] = true
        return m
    elseif n == 4
        # approximate with a ring
        return ring_mask(n)
    else
        # n ≥ 5: embed original 5-node pattern in first 5 nodes
        # P1 ↔ E  (1 ↔ 3)
        m[1, 3] = true
        m[3, 1] = true

        # P2 ↔ I1 (2 ↔ 4)
        m[2, 4] = true
        m[4, 2] = true

        # I2 self-loop (5 ↔ 5)
        m[5, 5] = true

        # extras: give remaining nodes self-loops to mimic extra inhibitory units
        for i in 6:n
            m[i, i] = true
        end

        return m
    end
end

function star_loop_extended_mask(n::Int)
    m = falses(n, n)

    if n <= 1
        # nothing meaningful, leave isolated
        return m

    elseif n == 2
        # tiny "loop": 1 ↔ 2
        m[1, 2] = true
        m[2, 1] = true
        return m

    elseif n == 3
        # simple 3-cycle as a minimal loop structure
        m[1, 2] = true
        m[1, 3] = true
        m[2, 1] = true
        m[2, 2] = true
        m[3, 1] = true
        return m
    elseif n == 4
        m[1, 4] = true
        m[2, 3] = true
        m[3, 1] = true
        m[3, 2] = true
        m[3, 3] = true
        m[4, 1] = true
        m[4, 2] = true
        return m
    else
        m[1, 5] = true
        m[2, 3] = true
        m[2, 4] = true
        m[3, 1] = true
        m[3, 2] = true
        m[4, 3] = true
        m[4, 4] = true   # self-loop
        m[5, 1] = true
        m[5, 2] = true
        return m
    end
end



function ei_extended_mask(n::Int)
    m = falses(n, n)

    if n == 1
        # single self-loop
        m[1, 1] = true
        return m
    elseif n == 2
        # 1 -> 2, and 2 loops
        m[1, 2] = true
        m[2, 2] = true
        return m
    elseif n == 3
        # compressed version of the 4-node pattern
        m[1, 1] = true
        m[2, 3] = true
        m[3, 2] = true
        return m
    else
        # n ≥ 4: original 4-node pattern in the first 4 nodes
        # Row 1: self-loop
        m[1, 1] = true

        # Row 2: to 3
        m[2, 3] = true

        # Row 3: self-loop
        m[3, 3] = true

        # Row 4: to 2
        m[4, 2] = true

        # For nodes > 4, just extend a chain outward from 4
        for i in 4:(n-1)
            m[i, i+1] = true
        end

        return m
    end
end

function star_feedback_tail_mask(n::Int)
    m = falses(n, n)

    if n <= 1
        # no edges
        return m
    elseif n == 2
        # minimal symmetric pair
        m[1, 2] = true
        m[2, 1] = true
        return m
    else
        # 1 ↔ 2..(n-1): bidirectional star
        for j in 2:(n-1)
            m[1, j] = true
            m[j, 1] = true
        end

        # feedback tail: (n-1) -> n -> 1
        m[n-1, n] = true  # e.g. 4 -> 5
        m[n, 1]   = true  # e.g. 5 -> 1

        return m
    end
end


############################
# additional deterministic motifs (general, n=3–5)
############################

# helper to clear diagonal if needed
function _maybe_clear_diag!(m::AbstractMatrix{Bool}, allow_self::Bool)
    if !allow_self
        n = size(m, 1)
        @inbounds for i in 1:n
            m[i, i] = false
        end
    end
    return m
end

"""
    bidir_chain_mask(n::Int; allow_self::Bool=false)

Bidirectional nearest-neighbor chain: i ↔ i+1 for i=1..n-1.
Gives local recurrence without closing into a ring.
"""
function bidir_chain_mask(n::Int; allow_self::Bool=false)
    m = falses(n, n)
    n <= 1 && return _maybe_clear_diag!(m, allow_self)

    for i in 1:(n-1)
        m[i, i+1] = true
        m[i+1, i] = true
    end

    return _maybe_clear_diag!(m, allow_self)
end

"""
    bowtie_mask(n::Int; core::Int=clamp(ceil(Int,n/2),2,n-1), allow_self::Bool=false)

Bowtie: inputs -> hub -> outputs (directional bottleneck + broadcast).

- Hub is `core` (default middle-ish).
- Nodes < core feed into hub.
- Hub feeds to nodes > core.
- If one side is empty (e.g. core=2), it degrades gracefully.
"""
function bowtie_mask(n::Int; core::Int=clamp(ceil(Int, n/2), 2, max(1, n-1)), allow_self::Bool=false)
    m = falses(n, n)
    n <= 1 && return _maybe_clear_diag!(m, allow_self)

    core = clamp(core, 1, n)

    # inputs: 1..(core-1) -> core
    for i in 1:(core-1)
        m[i, core] = true
    end

    # outputs: core -> (core+1)..n
    for j in (core+1):n
        m[core, j] = true
    end

    # optional: if n==2, this produces 1 -> 2 (fine)
    return _maybe_clear_diag!(m, allow_self)
end

"""
    core_periphery_mask(n::Int; core_size::Int=clamp(ceil(Int,n/2),2,n), mode::Symbol=:both, allow_self::Bool=false)

Core-periphery:
- Dense directed core (all-to-all within core, excluding self unless allow_self=true).
- Sparse periphery connected to the core.

`mode` controls direction of periphery links:
- :in     => periphery -> core only
- :out    => core -> periphery only
- :both   => bidirectional between periphery and core
"""
function core_periphery_mask(
    n::Int;
    core_size::Int = clamp(ceil(Int, n/2), 2, n),
    mode::Symbol = :both,
    allow_self::Bool=false
)
    m = falses(n, n)
    n <= 1 && return _maybe_clear_diag!(m, allow_self)

    core_size = clamp(core_size, 1, n)
    core = 1:core_size
    per  = (core_size+1):n

    # dense core (directed)
    for i in core, j in core
        if allow_self || (i != j)
            m[i, j] = true
        end
    end

    # periphery linkage
    if core_size < n
        if mode == :in || mode == :both
            for p in per, c in core
                m[p, c] = true
            end
        end
        if mode == :out || mode == :both
            for c in core, p in per
                m[c, p] = true
            end
        end
    end

    return _maybe_clear_diag!(m, allow_self)
end


############################
# seeded / random masks    #
############################

"""
    erdos_renyi_mask(n::Int, p::Float64; rng)

Directed Erdos–Renyi with edge probability `p`.
"""
function erdos_renyi_mask(n::Int, p::Float64; rng::AbstractRNG)
    m = falses(n, n)
    for i in 1:n, j in 1:n
        if rand(rng) < p
            m[i, j] = true
        end
    end
    return m
end

function connected_erdos_renyi_mask(n::Int, p::Float64; rng=MersenneTwister())
    while true
        M = erdos_renyi_mask(n, p; rng=rng)
        # Convert Bool matrix → directed graph
        g = DiGraph(M)
        if is_weakly_connected(g)   # weak connectivity is what you want for motifs
            return M
        end
    end
end

"""
    small_world_mask(n::Int, k::Int, beta::Float64; rng)

Watts–Strogatz-like small-world on n nodes.
"""
function small_world_mask(n::Int, k::Int, beta::Float64; rng::AbstractRNG)
    if n <= 1
        return falses(n, n)
    elseif n == 2
        m = falses(2, 2)
        # 2-node "small world" ≈ fully connected pair
        m[1, 2] = true
        m[2, 1] = true
        return m
    end

    # original behaviour for n ≥ 3
    k = min(k, (n-1) ÷ 2)
    m = falses(n, n)

    # ring lattice
    for i in 1:n
        for d in 1:k
            j = ((i - 1 + d) % n) + 1
            m[i, j] = true
            m[j, i] = true
        end
    end

    # rewiring
    for i in 1:n
        for j in 1:n
            if m[i, j] && rand(rng) < beta
                m[i, j] = false
                newj = i
                while newj == i
                    newj = rand(rng, 1:n)
                end
                m[i, newj] = true
            end
        end
    end

    return m
end


"""
    scale_free_mask(n::Int; m0::Int=2, m::Int=1, rng)

Barabási–Albert-like preferential attachment.
Edges are directed from existing nodes to new nodes.
"""
function scale_free_mask(n::Int; m0::Int=2, m::Int=1, rng::AbstractRNG)
    if n <= 1
        return falses(n, n)
    end

    m0 = clamp(m0, 1, n)
    m  = max(1, m)

    A   = falses(n, n)
    deg = zeros(Int, n)

    # fully connected core
    if m0 >= 2
        for i in 1:m0, j in 1:m0
            if i != j
                A[i, j] = true
                deg[i] += 1
            end
        end
    end

    for new in (m0+1):n
        for e in 1:m
            weights = Float64[]
            for i in 1:(new-1)
                push!(weights, deg[i] + 1)
            end
            total = sum(weights)
            r = rand(rng) * total
            acc = 0.0
            chosen = 1
            for i in 1:(new-1)
                acc += weights[i]
                if r <= acc
                    chosen = i
                    break
                end
            end
            A[chosen, new] = true
            deg[chosen] += 1
        end
    end

    return A
end


############################
# register built-in motifs #
############################

# deterministic families
add_conn_motif_builder(:null,         (n; kwargs...) -> null_mask(n))
add_conn_motif_builder(:full,         (n; kwargs...) -> full_mask(n))
add_conn_motif_builder(:chain,        (n; kwargs...) -> chain_mask(n))
add_conn_motif_builder(:ring,         (n; kwargs...) -> ring_mask(n))
add_conn_motif_builder(:star,         (n; kwargs...) -> star_mask(n))
add_conn_motif_builder(:star_tail,    (n; kwargs...) -> star_tail_mask(n))
add_conn_motif_builder(:hub_tail,     (n; kwargs...) -> star_tail_mask(n)) # legacy name
add_conn_motif_builder(:star_feedback_tail, (n; kwargs...) -> star_feedback_tail_mask(n))
add_conn_motif_builder(:star_loop_extended, (n; kwargs...) -> star_loop_extended_mask(n))
add_conn_motif_builder(:two_modules,  (n; kwargs...) -> two_modules_mask(n))
add_conn_motif_builder(:ei_block,     (n; kwargs...) -> ei_block_mask(n))
add_conn_motif_builder(:multi_cycle,  (n; kwargs...) -> multi_cycle_mask(n))
add_conn_motif_builder(:chain_loop,   (n; kwargs...) -> ei_extended_mask(n)) # legacy name
add_conn_motif_builder(:ei_extended,   (n; kwargs...) -> ei_extended_mask(n))

# additional general-purpose motifs with allow_self support
add_conn_motif_builder(:bidir_chain,    (n; kwargs...) -> bidir_chain_mask(n; kwargs...))
add_conn_motif_builder(:bowtie,         (n; kwargs...) -> bowtie_mask(n; kwargs...))
add_conn_motif_builder(:core_periphery, (n; kwargs...) -> core_periphery_mask(n; kwargs...))

############################
# register 2×2 digit motifs#
############################

add_conn_motif_builder(:c2_0, (n; kwargs...) -> c2_0_mask(n))
add_conn_motif_builder(:c2_1, (n; kwargs...) -> c2_1_mask(n))
add_conn_motif_builder(:c2_2, (n; kwargs...) -> c2_2_mask(n))
add_conn_motif_builder(:c2_3, (n; kwargs...) -> c2_3_mask(n))
add_conn_motif_builder(:c2_4, (n; kwargs...) -> c2_4_mask(n))
add_conn_motif_builder(:c2_5, (n; kwargs...) -> c2_5_mask(n))
add_conn_motif_builder(:c2_6, (n; kwargs...) -> c2_6_mask(n))
add_conn_motif_builder(:c2_7, (n; kwargs...) -> c2_7_mask(n))
add_conn_motif_builder(:c2_8, (n; kwargs...) -> c2_8_mask(n))
add_conn_motif_builder(:c2_9, (n; kwargs...) -> c2_9_mask(n))

# seeded families
add_random_conn_motif(:erdos_renyi,
    (n, rng; p=0.3, kwargs...) -> erdos_renyi_mask(n, p; rng=rng))

add_random_conn_motif(:connected_erdos_renyi,
    (n, rng; p=0.3, kwargs...) -> connected_erdos_renyi_mask(n, p; rng=rng))

add_random_conn_motif(:small_world,
    (n, rng; k=2, beta=0.3, kwargs...) -> small_world_mask(n, k, beta; rng=rng))

add_random_conn_motif(:scale_free,
    (n, rng; m0=2, m=1, kwargs...) -> scale_free_mask(n; m0=m0, m=m, rng=rng))

############################
# top-level API            #
############################
"""
    conn_mask(motif::AbstractString, n::Int; seed=nothing, kwargs...)

Return the connectivity mask for motif with given name and population count `n`.

- If `motif` is a deterministic motif (registered via `add_conn_motif` or
  `add_conn_motif_builder`), `seed` is ignored and the builder is called as
  `builder(n; kwargs...)`.

- If `motif` is a seeded motif (registered via `add_random_conn_motif`),
  `seed` must be provided and a local `MersenneTwister(seed)` is used.
"""
function conn_mask(motif::AbstractString, n::Int; seed=nothing, kwargs...)
    return conn_mask(Symbol(motif), n; seed=seed, kwargs...)
end

function conn_mask(motif::Symbol, n::Int; seed=nothing, kwargs...)
    @assert n >= 1 "n must be >= 1 (got $n)"

    if haskey(CONN_MOTIF_BUILDERS, motif)
        # deterministic: ignore seed
        builder = CONN_MOTIF_BUILDERS[motif]
        return builder(n; kwargs...)
    elseif haskey(RANDOM_CONN_MOTIF_BUILDERS, motif)
        # seeded: require seed
        seed === nothing && error("Motif $motif requires a seed")
        rng = MersenneTwister(seed::Int)
        builder = RANDOM_CONN_MOTIF_BUILDERS[motif]
        return builder(n, rng; kwargs...)
    else
        error("Unknown motif name: $motif")
    end
end

"""
    list_conn_motifs(; deterministic::Bool=true, random::Bool=true)

Return a NamedTuple of available motif names:
    (deterministic = [...], random = [...])

Use flags to include/exclude categories.
"""
function list_conn_motifs(; deterministic::Bool=true, random::Bool=true)
    d = deterministic ? collect(keys(CONN_MOTIF_BUILDERS)) : Symbol[]
    r = random        ? collect(keys(RANDOM_CONN_MOTIF_BUILDERS)) : Symbol[]
    return (deterministic = sort(d), random = sort(r))
end

"""
    seed_from_digits(d1::Char, d2::Char) -> Int

Convert two digit characters ('0'–'9') into a seed 0–99.
Useful with grammar tokens for Digit Digit.
"""
function seed_from_digits(d1::Char, d2::Char)
    @assert '0' <= d1 <= '9' "Expected digit char, got $d1"
    @assert '0' <= d2 <= '9' "Expected digit char, got $d2"
    return (Int(d1) - Int('0')) * 10 + (Int(d2) - Int('0'))
end


############################
# motif detection          #
############################

"""
    detect_conn_motif(mask::AbstractMatrix{Bool}) -> Union{Symbol,Nothing}

Try to identify which deterministic motif (null, full, chain, ring, star,
star_tail, two_modules, ei_block) produced `mask`.

Returns the motif name as a Symbol, or `nothing` if no exact match.
"""
function detect_conn_motif(mask::AbstractMatrix{Bool})
    n1, n2 = size(mask)
    n1 == n2 || error("Mask must be square, got $(size(mask))")

    # Only deterministic motifs; random ones like :erdos_renyi can’t be matched
    deterministic = (:null, :full, :chain, :ring, :star, :star_tail, :two_modules, :ei_block)

    for name in deterministic
        haskey(CONN_MOTIF_BUILDERS, name) || continue
        ref = CONN_MOTIF_BUILDERS[name](n1)
        if ref == mask
            return name
        end
    end

    return nothing
end

"""
    detect_conn_motif(A::AbstractMatrix{<:AbstractString}) -> Union{Symbol,Nothing}

Convenience wrapper for matrices of strings like "none", "saturating_sigmoid", etc.
Treats any entry != "none" as a connection.
"""
function detect_conn_motif(A::AbstractMatrix{<:AbstractString})
    n1, n2 = size(A)
    n1 == n2 || error("Matrix must be square, got $(size(A))")
    mask = A .!= "none"
    return detect_conn_motif(mask)
end
