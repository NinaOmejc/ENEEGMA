"""
-------------
LOSSES - Module containing loss functions and 
         metrics used in optimization.
-------------

"""

# Main function that takes the loss function as an argument
"""
    get_metric_function()::Function

Returns the unified metric function (region-weighted MAE).

Since all metrics now use the unified region-weighted MAE approach,
this function returns the same function regardless of parameters.

# Returns
- `metric_function`: weighted_mae function
"""
function get_metric_function()::Function
    return weighted_mae
end


"""
    get_loss_function()::Function

Returns the unified loss function wrapped with compute_loss for optimization.

Since all loss computations now use the unified region-weighted MAE approach,
this function returns a single loss wrapper regardless of input.

# Returns
- `loss_function`: Function that computes weighted MAE loss with optimization checks
"""
function get_loss_function()::Function
    return (x, p) -> compute_loss(x, p, weighted_mae)
end


"""
    compute_loss(x, p, loss_fun::Function)

Generic loss computation framework that handles parameter updating, ODE solving, 
and error detection, then applies a specific metric function to calculate the final loss.

# Arguments
- `x`: Parameter vector
- `p`: Tuple containing (prob, simulation_settings, data, setter, diffcache, solver, freq_range_idx)
- `loss_fun`: Function that calculates the specific metric between true and predicted data

# Returns
- `loss`: Computed loss value
"""

function compute_loss(new_params, args, metric_fun::Function)
    prob          = args.prob
    data        = args.data
    setter        = args.setter
    all_params    = args.all_params
    tspan         = args.tspan
    loss_settings = args.loss_settings
    brain_source_idx = args.brain_source_idx
    solver        = args.solver
    solver_kwargs = args.solver_kwargs

    n_inits = length(prob.u0)
    n_param_block = length(new_params) - n_inits
    @assert n_param_block >= 0 "Decision vector smaller than expected"

    θ = new_params[1:n_param_block]
    iv = new_params[n_param_block + 1 : n_param_block + n_inits]
    sigma_effective = max(loss_settings.measurement_noise_std, 0.0)
    updated_all_params = setter(all_params, θ)
    new_prob = remake(prob; p=updated_all_params, u0=iv, tspan=tspan)
    sol = safe_solve(new_prob, solver; solver_kwargs=solver_kwargs)
    
    if !SciMLBase.successful_retcode(sol)
        return 1e9
    end

    # Discard initial transient, e.g. first 1–2 s =
    keep_idx = findall(t -> t >= (tspan[1] + 2), sol.t)
    if isempty(keep_idx)
        return 1e9  # Solver terminated before transient period ended
    end
    model_prediction = Vector(sol[brain_source_idx, keep_idx])

    if any(!isfinite, model_prediction)
        return 1e9
    end

    # crude growth detection to bail out exploding trajectories before PSD computation
    n = length(model_prediction)
    w = min(200, max(50, n ÷ 5))
    max_rms_growth_threshold = 100.0  # Safety threshold: trajectory can grow up to 100x
    if n ≥ 2w
        # Compute RMS directly without allocating abs_trace
        head_sum_sq = 0.0
        tail_sum_sq = 0.0
        @inbounds for i in 1:w
            head_sum_sq += model_prediction[i]^2
            tail_sum_sq += model_prediction[n - w + i]^2
        end
        head_rms = sqrt(head_sum_sq / w)
        tail_rms = sqrt(tail_sum_sq / w)
        if isfinite(head_rms) && head_rms > 0 && tail_rms > max_rms_growth_threshold * head_rms
            return 1e9
        end
    end

    # Check max absolute value for signal bounds (safety threshold)
    max_abs_signal_threshold = 100.0  # Safety threshold: signal cannot exceed ±100
    max_abs = maximum(abs, model_prediction)
    if max_abs > max_abs_signal_threshold
        return 1e9
    end

    # do demean
    model_prediction .-= mean(model_prediction)

    loss = metric_fun(model_prediction, data, 1/solver_kwargs[:saveat], loss_settings)
    
    if !isfinite(loss)
        return 1e9
    end
    return loss
end


function loss_empty(new_params, args)
    return NaN  # Or some default high loss value like 1e9
end

"""
    weighted_mae(model_prediction, target::Data, fs, loss_settings::LossSettings)

Unified frequency-domain MAE loss with region-based weighting.

Computes MAE between model and target PSDs, with different weights applied to:
- ROI (regions of interest): peaks, harmonics, or manually specified bands
- Background: all other frequency regions

Masks and weights are pre-computed and stored in target.freq_peak_metadata during data preparation.
If no metadata is available, defaults to uniform (unweighted) MAE.

# Arguments
- `model_prediction`: Time-domain model prediction
- `target::Data`: Target data with freq_peak_metadata containing masks
- `fs::Float64`: Sampling frequency
- `loss_settings::LossSettings`: Contains PSD settings and region weights

# Returns
- `Float64`: Weighted MAE loss
"""
function weighted_mae(model_prediction, target::Data, fs, loss_settings::LossSettings)
    # Compute model PSD
    model_freqs, modeled_powers = _compute_noisy_psd_avg(model_prediction, fs, loss_settings)
    aligned_model = _interpolate_psd(model_freqs, modeled_powers, target.freqs)
    
    # Without region masks: simple uniform MAE
    if target.freq_peak_metadata === nothing
        return mean(abs.(target.powers .- aligned_model))
    end
    
    pm = target.freq_peak_metadata
    roi_mask = pm.roi_mask
    bg_mask = pm.bg_mask
    
    # Compute MAE in each region
    roi_diff = abs.(target.powers[roi_mask] .- aligned_model[roi_mask])
    bg_diff = abs.(target.powers[bg_mask] .- aligned_model[bg_mask])
    
    roi_mae = isempty(roi_diff) ? 0.0 : mean(roi_diff)
    bg_mae = isempty(bg_diff) ? 0.0 : mean(bg_diff)
    
    # Weighted combination
    roi_weight = pm.roi_weight
    bg_weight = pm.bg_weight
    total_weight = roi_weight + bg_weight
    
    if total_weight <= 0
        # Fallback: equal weighting if both are zero
        return 0.5 * (roi_mae + bg_mae)
    end
    
    return (roi_weight * roi_mae + bg_weight * bg_mae) / total_weight
end


# Helper functions used by weighted_mae
function _interpolate_psd(freqs_src::AbstractVector{<:Real}, values_src::AbstractVector{<:Real}, freqs_target::AbstractVector{<:Real})
    length(freqs_src) == length(values_src) || error("Source PSD arrays must have the same length.")
    isempty(freqs_target) && return Float64[]
    result = Vector{Float64}(undef, length(freqs_target))
    j = 1
    src_last = length(freqs_src)
    for (i, f) in enumerate(freqs_target)
        if f <= freqs_src[1]
            result[i] = Float64(values_src[1])
            continue
        elseif f >= freqs_src[end]
            result[i] = Float64(values_src[end])
            continue
        end
        while j < src_last && freqs_src[j + 1] < f
            j += 1
        end
        span = freqs_src[j + 1] - freqs_src[j]
        if span <= eps(Float64)
            result[i] = Float64(values_src[j])
        else
            α = (f - freqs_src[j]) / span
            result[i] = (1 - α) * Float64(values_src[j]) + α * Float64(values_src[j + 1])
        end
    end
    return result
end

# ================================
# == Measurement noise application ==
# ================================

# Draw a white-noise template (N(0,1)) of requested length
# If loss_noise_seed is nothing/null → use non-deterministic random noise
# If loss_noise_seed is an integer → use deterministic seeded noise
function _measurement_noise_template(len::Int, loss_settings::LossSettings)
    if len <= 0
        return Float64[]
    end
    if loss_settings.loss_noise_seed === nothing
        # No seed specified: use Julia's default random state (non-deterministic)
        vprint("Applying measurement noise, but loss_noise_seed is null: note that results will vary between runs/calls", level=1)
        return randn(len)
    else
        # Seed specified: use deterministic seeded RNG
        rng = Random.MersenneTwister(Int(loss_settings.loss_noise_seed))
        return randn(rng, len)
    end
end

function apply_measurement_noise!(data::AbstractVector, sigma, loss_settings::LossSettings)
    sigma === nothing && return data
    if Base.iszero(sigma)
        return data
    end
    len = length(data)
    len == 0 && return data
    noise = _measurement_noise_template(len, loss_settings)
    @inbounds @simd for i in 1:len
        data[i] += sigma * noise[i]
    end
    return data
end


# ================================
# == PSD computation with noise averaging ==
# ================================

function _compute_noisy_psd_avg(model_prediction::AbstractVector{<:Real},
                                fs::Real,
                                loss_settings::LossSettings)
    reps = max(loss_settings.psd_noise_avg_reps, 1)
    sigma_effective = max(loss_settings.measurement_noise_std, 0.0)

    if sigma_effective <= 0
        return compute_preprocessed_welch_psd(model_prediction, fs; loss_settings=loss_settings)
    end

    seed_backup = loss_settings.loss_noise_seed
    freqs = Float64[]
    accum = Float64[]
    for rep in 1:reps
        if seed_backup === nothing
            loss_settings.loss_noise_seed = nothing
        else
            loss_settings.loss_noise_seed = Int(seed_backup) + rep - 1
        end
        noisy = Float64.(model_prediction)
        apply_measurement_noise!(noisy, sigma_effective, loss_settings)
        freqs_rep, powers_rep = compute_preprocessed_welch_psd(noisy, fs; loss_settings=loss_settings)
        if isempty(freqs)
            freqs = freqs_rep
            accum = zeros(length(powers_rep))
        end
        accum .+= powers_rep
    end
    loss_settings.loss_noise_seed = seed_backup
    return freqs, accum ./ reps
end
