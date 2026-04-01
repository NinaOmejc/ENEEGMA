# ---------------------------------------------------------------------------
# Parameter reparametrization utilities
# ---------------------------------------------------------------------------

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

to_phys(z::Real, tr::ParamTransform, ::Float64, ::Float64) = throw(ArgumentError("Unknown transform $(typeof(tr))"))
to_opt(x::Real, tr::ParamTransform, ::Float64, ::Float64) = throw(ArgumentError("Unknown transform $(typeof(tr))"))

function to_phys(z::Real, tr::Identity, ::Float64, ::Float64)
    return Float64(z)
end

function to_opt(x::Real, tr::Identity, ::Float64, ::Float64)
    return Float64(x)
end

function to_phys(z::Real, tr::Affine01, target_min::Float64, target_max::Float64)
    span_target = target_max - target_min
    span_target <= eps(Float64) && return tr.min
    frac = (Float64(z) - target_min) / span_target
    frac = clamp(frac, 0.0, 1.0)
    return tr.min + (tr.max - tr.min) * frac
end

function to_opt(x::Real, tr::Affine01, target_min::Float64, target_max::Float64)
    span_phys = tr.max - tr.min
    span_phys <= eps(Float64) && return target_min
    frac = clamp((Float64(x) - tr.min) / span_phys, 0.0, 1.0)
    return target_min + frac * (target_max - target_min)
end

function to_phys(z::Real, tr::ExpPos, ::Float64, ::Float64)
    scale = tr.scale <= 0 ? 1.0 : tr.scale
    return scale * exp(Float64(z))
end

function to_opt(x::Real, tr::ExpPos, target_min::Float64, target_max::Float64)
    scale = tr.scale <= 0 ? 1.0 : tr.scale
    val = max(Float64(x) / scale, eps(Float64))
    return clamp(log(val), target_min, target_max)
end

function to_phys(z::Real, tr::SoftplusPos, ::Float64, ::Float64)
    scale = tr.scale <= 0 ? 1.0 : tr.scale
    return scale * log1p(exp(Float64(z)))
end

function to_opt(x::Real, tr::SoftplusPos, target_min::Float64, target_max::Float64)
    scale = tr.scale <= 0 ? 1.0 : tr.scale
    y = max(Float64(x) / scale, eps(Float64))
    return clamp(log(expm1(y)), target_min, target_max)
end

function to_phys(z::Real, tr::SigmoidBound, ::Float64, ::Float64)
    span = tr.max - tr.min
    span <= eps(Float64) && return tr.min
    sig = 1 / (1 + exp(-Float64(z)))
    return tr.min + span * sig
end

function to_opt(x::Real, tr::SigmoidBound, target_min::Float64, target_max::Float64)
    span = tr.max - tr.min
    span <= eps(Float64) && return target_min
    y = (Float64(x) - tr.min) / span
    y = clamp(y, 1e-12, 1 - 1e-12)
    return clamp(log(y / (1 - y)), target_min, target_max)
end

function to_phys(z::Real, tr::TanhBound, ::Float64, ::Float64)
    scale = tr.scale <= 0 ? 1.0 : tr.scale
    return scale * tanh(Float64(z))
end

function to_opt(x::Real, tr::TanhBound, target_min::Float64, target_max::Float64)
    scale = tr.scale <= 0 ? 1.0 : tr.scale
    y = clamp(Float64(x) / scale, -0.999999999999, 0.999999999999)
    return clamp(atanh(y), target_min, target_max)
end


"""
    ReparamSpec

Lightweight container describing how a block of decision variables should be
re-parameterized. Stores the optional transform and block length.
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


"""
    build_param_reparam_transform(mins, maxs; target_bounds=(-5.0, 5.0), types=nothing, strategy=:typed)

Construct a `ParamReparamTransform` using per-dimension mappings that depend on
parameter bounds and optional type metadata.
"""
function build_param_reparam_transform(
    mins::AbstractVector{<:Real},
    maxs::AbstractVector{<:Real};
    target_bounds::Tuple{<:Real,<:Real}=(-5.0, 5.0),
    types::Union{Nothing, AbstractVector}=nothing,
    strategy::Symbol=:typed,
    type_scales::Union{Nothing, AbstractDict{Symbol, Float64}}=nothing
)
    n = length(mins)
    n == length(maxs) || error("Parameter bound vectors must have the same length.")
    if types !== nothing && length(types) != n
        error("Type metadata must match parameter vector length.")
    end
    target_min, target_max = Float64(target_bounds[1]), Float64(target_bounds[2])
    transforms = ParamTransform[]
    for i in 1:n
        lb = Float64(mins[i])
        ub = Float64(maxs[i])
        ty = types === nothing ? nothing : _normalize_param_type_symbol(types[i])
        push!(transforms, _build_transform_for_dim(lb, ub, ty, strategy, type_scales))
    end
    return ParamReparamTransform(transforms; target_min=target_min, target_max=target_max)
end

# --- transform groups aligned with the cleaned-up type system ---

# Positive-only (if bounds are finite -> SigmoidBound, else ExpPos)
const _POSITIVE_TRANSFORM_TYPES = union(POSITIVE_ONLY_PARAM_TYPES, Set([
    :frequency, :time_constant, :rate, :damping, :gain, :noise_std
]))

# Signed weights / coefficients (finite -> SigmoidBound, else TanhBound)
const _SIGNED_WEIGHT_TRANSFORM_TYPES = Set([
    :population_coupling, :node_coupling, :sensory_coupling, :poly_coeff, :unknown
])

# Signed offsets (finite -> SigmoidBound, else TanhBound)
const _SIGNED_OFFSET_TRANSFORM_TYPES = Set([
    :offset, :potential, :initial_condition
])

# [0,1]
const _PROBABILITY_TRANSFORM_TYPES = Set([:probability])


const _DEFAULT_UNBOUNDED_TYPE_SCALES = Dict(
    # amplitudes
    :gain => 10.0,
    :noise_std => 1.0,

    # weights / coefficients
    :population_coupling => 10.0,
    :poly_coeff => 10.0,
    :unknown => 10.0,

    # signed offsets
    :offset => 10.0,
    :potential => 50.0,          # voltage-like: often larger magnitude than offset
    :initial_condition => 10.0,

    # time-like / rate-like
    :time_constant => 1.0,
    :frequency => 1.0,
    :rate => 1.0,
    :damping => 1.0,

    # probabilities
    :probability => 1.0,
)

function _default_unbounded_scale(ty::Union{Symbol, Nothing}, overrides::Union{Nothing, AbstractDict{Symbol, Float64}}=nothing)
    base = ty === nothing ? 5.0 : get(_DEFAULT_UNBOUNDED_TYPE_SCALES, ty, 5.0)
    overrides === nothing && return base
    if ty !== nothing && haskey(overrides, ty)
        return max(Float64(overrides[ty]), eps())
    elseif haskey(overrides, :default)
        return max(Float64(overrides[:default]), eps())
    end
    return base
end

_normalize_param_type_symbol(val) = nothing

function _normalize_param_type_symbol(val::Symbol)
    sym = Symbol(lowercase(String(val)))
    sym == :timeconst && return :time_constant
    sym == :poly && return :poly_coeff
    sym == :coeff && return :poly_coeff
    sym == :coupling && return :population_coupling
    return sym
end

function _normalize_param_type_symbol(val::AbstractString)
    sym = Symbol(lowercase(strip(val)))
    return sym == :timeconst ? :time_constant : sym
end

function _build_transform_for_dim(lb::Float64, ub::Float64, ty::Union{Symbol, Nothing}, strategy::Symbol,
                                  type_scales::Union{Nothing, AbstractDict{Symbol, Float64}})
    if strategy == :legacy
        return _legacy_transform(lb, ub)
    elseif strategy == :typed
        return _typed_transform(lb, ub, ty, type_scales)
    else
        return Identity()
    end
end

function _legacy_transform(lb::Float64, ub::Float64)
    if !isfinite(lb) || !isfinite(ub) || lb >= ub
        return Identity()
    end
    return Affine01(lb, ub)
end

function _typed_transform(lb::Float64, ub::Float64, ty::Union{Symbol, Nothing},
                          type_scales::Union{Nothing, AbstractDict{Symbol, Float64}})
    finite_span = _has_finite_span(lb, ub)

    if ty !== nothing
        # Unit interval - always hard-bound to [0, 1]
        if ty in _PROBABILITY_TRANSFORM_TYPES
            return SigmoidBound(0.0, 1.0)
        end

        # Positive-only
        if ty in _POSITIVE_TRANSFORM_TYPES
            if finite_span
                return SigmoidBound(lb, ub)
            else
                scale = _default_unbounded_scale(ty, type_scales)
                return ExpPos(scale)
            end
        end

        # Signed weights/coefficients
        if ty in _SIGNED_WEIGHT_TRANSFORM_TYPES
            fallback = _default_unbounded_scale(ty, type_scales)
            return finite_span ? SigmoidBound(lb, ub) : _tanh_from_bounds(lb, ub; fallback_scale=fallback)
        end

        # Signed offsets/potentials
        if ty in _SIGNED_OFFSET_TRANSFORM_TYPES
            fallback = _default_unbounded_scale(ty, type_scales)
            return finite_span ? SigmoidBound(lb, ub) : _tanh_from_bounds(lb, ub; fallback_scale=fallback)
        end
    end

    return _fallback_transform(lb, ub)
end

_has_finite_span(lb::Float64, ub::Float64) = isfinite(lb) && isfinite(ub) && ub > lb

function _infer_positive_scale(lb::Float64, ub::Float64)
    vals = Float64[1.0]
    isfinite(lb) && push!(vals, abs(lb))
    isfinite(ub) && push!(vals, abs(ub))
    return max(1.0, maximum(vals))
end

function _tanh_from_bounds(lb::Float64, ub::Float64; fallback_scale::Float64=1.0)
    vals = Float64[max(1.0, fallback_scale)]
    isfinite(lb) && push!(vals, abs(lb))
    isfinite(ub) && push!(vals, abs(ub))
    scale = max(1.0, maximum(vals))
    return scale <= 0 ? Identity() : TanhBound(scale)
end

function _sigmoid_or_tanh(lb::Float64, ub::Float64; fallback_scale::Float64=1.0)
    if isfinite(lb) && isfinite(ub) && ub > lb
        return SigmoidBound(lb, ub)
    end
    return _tanh_from_bounds(lb, ub; fallback_scale=fallback_scale)
end

function _fallback_transform(lb::Float64, ub::Float64)
    if isfinite(lb) && isfinite(ub) && ub > lb
        if lb < 0 < ub
            return _tanh_from_bounds(lb, ub)
        else
            return SigmoidBound(lb, ub)
        end
    elseif lb >= 0
        return ExpPos(_infer_positive_scale(lb, ub))
    else
        return Identity()
    end
end

#
# the thing is that i will mix my dynamics blocs, have 1000 different models and it could be that param ranges are different for each model. that's why I'd like to be general as possible. Could i set up a hyperparameter or sth with which param ranges will be best controled? Or what would you suggest? I just also keep the default values and just extend them

"""
    reparametrize_params(values, transform)

Map a vector of parameter values from their native range into the target range
defined by `transform`.
"""
function reparametrize_params(values::AbstractVector{<:Real}, transform::ParamReparamTransform)
    length(values) == length(transform) || error("Value vector length does not match transform length.")
    normalized = Vector{Float64}(undef, length(transform))
    @inbounds for i in 1:length(transform)
        normalized[i] = to_opt(Float64(values[i]), transform.transforms[i], transform.target_min, transform.target_max)
    end
    return normalized
end


"""
    restore_params_from_reparam!(values, transform, n_params=length(transform))

In-place inverse of `reparametrize_params`. Converts the first `n_params`
entries of `values` from the target range back to their native ranges.
"""
function restore_params_from_reparam!(values::AbstractVector, transform::ParamReparamTransform, n_params::Int=length(transform))
    limit = min(n_params, length(transform), length(values))
    @inbounds for i in 1:limit
        values[i] = to_phys(values[i], transform.transforms[i], transform.target_min, transform.target_max)
    end
    return values
end


"""
    map_to_shared_space(values, spec)

Project `values` (expressed in physical/native units) into the shared target
range described by `spec`. Returns a new vector.
"""
function map_to_shared_space(values::AbstractVector, spec::ReparamSpec)
    vec = Vector{Float64}(values)
    length(vec) == spec.n || error("Input length does not match ReparamSpec length.")
    return spec.transform === nothing ? vec : reparametrize_params(vec, spec.transform)
end


"""
    reparametrize_to_phys(values, spec)

Map values from the shared target range back to their native ranges according
to `spec`. Returns a new vector.
"""
function reparametrize_to_phys(values::AbstractVector, spec::ReparamSpec)
    vec = Vector{Float64}(values)
    if spec.transform === nothing
        return vec
    end
    restore_params_from_reparam!(vec, spec.transform, spec.n)
    return vec
end


"""
    build_reparam_spec(mins, maxs, active; types=nothing, strategy=:typed)

Return a `ReparamSpec` for a block of decision variables. When `active` is
false, the spec encodes a no-op transform.
"""
function build_reparam_spec(
    mins::AbstractVector{<:Real},
    maxs::AbstractVector{<:Real},
    active::Bool;
    target_bounds::Tuple{<:Real,<:Real}=(-5.0, 5.0),
    types::Union{Nothing, AbstractVector}=nothing,
    strategy::Symbol=:typed,
    type_scales::Union{Nothing, AbstractDict{Symbol, Float64}}=nothing
)
    if !active || isempty(mins) || strategy == :none
        return ReparamSpec(length(mins))
    end
    transform = build_param_reparam_transform(mins, maxs;
                                              target_bounds=target_bounds,
                                              types=types,
                                              strategy=strategy,
                                              type_scales=type_scales)
    return ReparamSpec(transform, length(mins))
end


"""
    reparam_bounds(transform)

Return the lower and upper bounds in the target space for the provided
transform.
"""
function reparam_bounds(transform::ParamReparamTransform)
    n = length(transform)
    return (fill(transform.target_min, n), fill(transform.target_max, n))
end


"""
    reparam_bounds(spec, native_lb, native_ub)

Return bounds for the optimizer given a `ReparamSpec`. Falls back to the native
min/max when no transform is active.
"""
function reparam_bounds(spec::ReparamSpec, native_lb::AbstractVector, native_ub::AbstractVector)
    if spec.transform === nothing
        return native_lb, native_ub
    else
        return reparam_bounds(spec.transform)
    end
end


"""
    wrap_loss_for_reparam(lossfun, spec_theta, specu0; active=false)

Return a loss function wrapper that converts optimization variables from the
shared target range back to their native scales before invoking `lossfun`.
"""
function wrap_loss_for_reparam(lossfun::Function,
                               param_spec::ReparamSpec,
                               init_spec::ReparamSpec;
                               active::Bool=false)
    if !active
        return lossfun
    end

    buf = Vector{Float64}(undef, length(param_spec) + length(init_spec))
    return function(z, args)
        if length(buf) != length(z)
            resize!(buf, length(z))
        end
        copyto!(buf, z)
        decode_reparam_solution!(buf, param_spec, init_spec)
        return lossfun(buf, args)
    end
end


"""
    materialize_logged_params(u, spec_theta, specu0)

Create a Float64 copy of the optimization vector for logging purposes, making
sure that tunable parameters are expressed in their native ranges.
"""
function materialize_logged_params(u::AbstractVector, spectheta::ReparamSpec, specu0::ReparamSpec)
    logged = Vector{Float64}(u)
    decode_reparam_solution!(logged, spectheta, specu0)
    return logged
end

materialize_logged_params(::Nothing, ::ReparamSpec, ::ReparamSpec) = nothing


"""
    decode_reparam_solution!(values, spec_theta, specu0)

Convert an optimization vector in-place from the shared target range back to
physical units using the provided specs.
"""
function decode_reparam_solution!(values::AbstractVector, spectheta::ReparamSpec, specu0::ReparamSpec)
    n_params = length(spectheta)
    if spectheta.transform !== nothing && n_params > 0
        restore_params_from_reparam!(view(values, 1:n_params), spectheta.transform, n_params)
    end
    n_inits = length(specu0)
    if specu0.transform !== nothing && n_inits > 0
        restore_params_from_reparam!(view(values, n_params + 1:n_params + n_inits), specu0.transform, n_inits)
    end
    return values
end


"""
    prepare_optimization_blocks(net, os)

Collect reusable information about tunable parameters and state initial
conditions for a network/optimization-settings pair. Returns a NamedTuple with
parameter bounds, initialization bounds, native initial values, and the
corresponding `ReparamSpec` instances. Note: tscale parameters are now created
as proper Param objects during network construction (in build_network.jl), so
they are automatically included in the tunable parameter set.
"""
function prepare_optimization_blocks(net::Network, os::OptimizationSettings)
    tunable_param_set = get_tunable_params(net.params)
    tunable_params = get_symbols(tunable_param_set)
    tunable_params_symbols = Symbol.(tunable_params)
    tunable_params_lb, tunable_params_ub = get_param_minmax_values(net.params; p_subset=tunable_params, return_type="vector")
    tunable_param_types = Symbol[p.type for p in tunable_param_set.params]

    # Note: tscale parameters are now created as Param objects in construct_network_dynamics!
    # and are automatically included in tunable_param_set if needed

    state_vars = get_state_vars(net.vars)
    n_states = length(net.problem.u0)
    default_bounds = (-5.0, 5.0)
    if !isempty(state_vars.vars)
        inits_lb = Float64[]
        inits_ub = Float64[]
        init_types = Symbol[]
        for sym in get_symbols(state_vars)
            var = try
                get_var_by_symbol(state_vars, sym)
            catch
                nothing
            end
            lo = var === nothing ? default_bounds[1] : Float64(var.init_min)
            hi = var === nothing ? default_bounds[2] : Float64(var.init_max)
            if !isfinite(lo)
                lo = default_bounds[1]
            end
            if !isfinite(hi)
                hi = default_bounds[2]
            end
            push!(inits_lb, lo)
            push!(inits_ub, hi)
            push!(init_types, :initial_condition)
        end
    else
        inits_lb = fill(default_bounds[1], n_states)
        inits_ub = fill(default_bounds[2], n_states)
        init_types = fill(:initial_condition, n_states)
    end

    u0_container = net.problem.u0
    initial_values_native = u0_container isa NamedTuple ? collect(values(u0_container)) : collect(u0_container)
    initial_values_native = Vector{Float64}(initial_values_native)
    length(initial_values_native) == n_states || error("Initial state vector length mismatch.")

    use_reparam = os.reparametrize && os.reparam_strategy != :none
    type_scales = isempty(os.reparam_type_scales) ? nothing : os.reparam_type_scales
    param_spec = build_reparam_spec(tunable_params_lb, tunable_params_ub, use_reparam;
                                    types=tunable_param_types,
                                    strategy=os.reparam_strategy,
                                    type_scales=type_scales)
    init_spec = build_reparam_spec(inits_lb, inits_ub, use_reparam;
                                   types=init_types,
                                   strategy=os.reparam_strategy,
                                   type_scales=type_scales)

    return (; tunable_params_symbols,
            tunable_params_lb,
            tunable_params_ub,
            init_lb=inits_lb,
            init_ub=inits_ub,
            initial_values_native,
            param_spec,
            init_spec)
end
