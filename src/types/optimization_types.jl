"""
Optimization-related type definitions and structures.
"""

# ---------------------------------------------------------------------------
# Parameter Transform Types and Reparametrization Specifications
# ---------------------------------------------------------------------------

"""
    ParamTransform

Abstract base type for parameter transformations. Concrete subtypes define
how to map between physical parameter space and optimization space.
"""
abstract type ParamTransform end

struct Affine01 <: ParamTransform
    min::Float64
    max::Float64
end

struct ExpPos <: ParamTransform
    scale::Float64
end

struct SoftplusPos <: ParamTransform
    scale::Float64
end

struct SigmoidBound <: ParamTransform
    min::Float64
    max::Float64
end

struct TanhBound <: ParamTransform
    scale::Float64
end

struct Identity <: ParamTransform end

"""
    ParamReparamTransform

Container that stores the per-dimension mappings needed to map parameter bounds
from their native range to a shared target range (default [-5, 5]).
"""
struct ParamReparamTransform
    transforms::Vector{ParamTransform}
    target_min::Float64
    target_max::Float64

    function ParamReparamTransform(transforms::Vector{ParamTransform}; target_min::Float64=-5.0, target_max::Float64=5.0)
        target_min < target_max || error("Target minimum must be smaller than target maximum.")
        return new(copy(transforms), target_min, target_max)
    end
end

Base.length(transform::ParamReparamTransform) = length(transform.transforms)

# ---------------------------------------------------------------------------
# Optimization Logging and Specification Types
# ---------------------------------------------------------------------------

"""
    OptLogEntry

Tracks a single optimization iteration including restart number, iteration count,
loss value, elapsed time, and the parameter values at that point.

# Fields
- `irestart::Int64`: Restart index (which restart this comes from)
- `iter::Int64`: Iteration number within the restart
- `loss::Float64`: Loss value at this iteration
- `time::Second`: Elapsed wall-clock time since optimization start
- `params::Union{Nothing, Vector{Float64}}`: Tunable parameters and state initials, or nothing if not logged
"""
struct OptLogEntry
    irestart::Int64
    iter::Int64
    loss::Float64
    time::Second
    params::Union{Nothing, Vector{Float64}}
end


"""
    ReparamSpec

Lightweight container describing how a block of decision variables should be
re-parameterized. Stores the optional transform and block length.

Used by the optimization framework to define parameter space transformations
for reparametrization strategies (e.g., logit transform for bounded parameters).

# Fields
- `transform::Union{ParamReparamTransform, Nothing}`: Transformation to apply, or nothing for identity
- `n::Int`: Number of variables in this block

# Constructors
- `ReparamSpec(n::Integer)`: Identity transform (no reparametrization)
- `ReparamSpec(transform::ParamReparamTransform, n::Integer)`: With specified transform
"""
struct ReparamSpec
    transform::Union{ParamReparamTransform, Nothing}
    n::Int
    function ReparamSpec(transform::Union{ParamReparamTransform, Nothing}, n::Integer)
        return new(transform, Int(n))
    end
end

ReparamSpec(n::Integer) = ReparamSpec(nothing, Int(n))

Base.length(spec::ReparamSpec) = spec.n
