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
    return (x, p) -> compute_loss(x, p, ENEEGMA.weighted_mae)
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
    brain_source_idx = args.brain_source_idx
    solver        = args.solver
    solver_kwargs = args.solver_kwargs
    loss_settings = args.loss_settings
    data_settings = haskey(args, :data_settings) ? args.data_settings : nothing

    n_inits = length(prob.u0)
    n_param_block = length(new_params) - n_inits
    @assert n_param_block >= 0 "Decision vector smaller than expected"

    θ = new_params[1:n_param_block]
    iv = new_params[n_param_block + 1 : n_param_block + n_inits]
    sigma_effective = if data_settings !== nothing && hasproperty(data_settings, :measurement_noise_std)
        max(data_settings.measurement_noise_std, 0.0)
    else
        0.0
    end
    updated_all_params = setter(all_params, θ)
    new_prob = remake(prob; p=updated_all_params, u0=iv, tspan=tspan)
    sol = safe_solve(new_prob, solver; solver_kwargs=solver_kwargs)
    
    # Create a temporary Settings object for compatibility with validation functions
    # Extract transient duration from data_settings if available
    transient_duration = data_settings.psd.transient_period_duration
    fs_actual = 1.0 / solver_kwargs[:saveat]  # Infer sampling frequency from time step
    keep_idx = ENEEGMA.get_indices_after_transient_removal(sol.t, transient_duration, tspan[1], fs_actual)
    
    # Validate simulation success (all checks in one place)
    success, error_msg, model_prediction = ENEEGMA.validate_simulation_success(
        sol, brain_source_idx, keep_idx, tspan[1], tspan[2], transient_duration
    )
    
    if !success
        return 1e9
    end

    # do demean
    model_prediction .-= Statistics.mean(model_prediction)

    loss = metric_fun(model_prediction, data, 1/solver_kwargs[:saveat], loss_settings, data_settings)
    
    if !isfinite(loss)
        return 1e9
    end
    return loss
end


function loss_empty(new_params, args)
    return NaN  # Or some default high loss value like 1e9
end


# ============================================================================
# HELPER FUNCTIONS FOR VALIDATION
# ============================================================================

"""
    validate_simulation_success(
        sol,
        brain_source_idx::Int,
        keep_idx,
        tspan_start::Float64,
        tspan_end::Float64,
        transient_duration::Float64
    )::Tuple{Bool, String, Vector{Float64}}

Comprehensive validation of simulation success with multiple safety checks.

Checks:
1. ODE solver completed successfully (retcode)
2. Valid keep indices after transient removal
3. All model prediction values are finite
4. Simulation wasn't terminated early (longer than expected based on tspan)
5. No individual state variables grew explosively
6. Model prediction max absolute value within bounds
7. Model prediction doesn't show exponential growth (RMS growth check)

# Returns
- `(success::Bool, error_msg::String, model_prediction::Vector{Float64})`
"""
function validate_simulation_success(
    sol,
    brain_source_idx::Int,
    keep_idx,
    tspan_start::Float64,
    tspan_end::Float64,
    transient_duration::Float64
)::Tuple{Bool, String, Vector{Float64}}
    
    # Check 1: Solver succeeded
    if !SciMLBase.successful_retcode(sol)
        return false, "Simulation failed with retcode: $(sol.retcode)", Float64[]
    end
    
    # Check 2: Keep indices valid
    if isempty(keep_idx)
        return false, "No simulation time points after transient period", Float64[]
    end
    
    # Extract prediction
    local model_prediction
    try
        model_prediction = Vector{Float64}(sol[brain_source_idx, keep_idx])
    catch e
        return false, "Failed to extract brain source: $e", Float64[]
    end
    
    # Check 3: Finite values in prediction
    if any(!isfinite, model_prediction)
        return false, "Model prediction contains non-finite values (Inf or NaN)", Float64[]
    end
    
    # Check 4: Simulation terminated early (check if got enough time points)
    # Calculate expected number of samples for full tspan
    expected_duration_sec = (tspan_end - tspan_start) / 1000.0  # Convert ms to seconds
    actual_duration_sec = (sol.t[end] - sol.t[1])
    if actual_duration_sec < 0.9 * expected_duration_sec  # Allow 10% tolerance
        return false, "Simulation terminated early (covered $(round(actual_duration_sec, digits=2))s of $(round(expected_duration_sec, digits=2))s)", Float64[]
    end
    
    # Check 5: All state variables bounded (check for explosions in internal states)
    # Look at early vs late RMS for each variable
    n_steps = length(sol.t)
    if n_steps >= 100
        n_sample = min(50, n_steps ÷ 5)
        
        for var_idx in 1:length(sol.u[1])
            # Compute early RMS from first n_sample steps
            early_sum_sq = 0.0
            for i in 1:n_sample
                early_sum_sq += abs(sol.u[i][var_idx])^2
            end
            early_rms = sqrt(early_sum_sq / n_sample)
            
            # Compute late RMS from last n_sample steps
            late_sum_sq = 0.0
            for i in (n_steps - n_sample + 1):n_steps
                late_sum_sq += abs(sol.u[i][var_idx])^2
            end
            late_rms = sqrt(late_sum_sq / n_sample)
            
            if early_rms > 0 && late_rms > 100.0 * early_rms
                return false, "State variable $var_idx exploded (RMS grew from $(round(early_rms, digits=2)) to $(round(late_rms, digits=2)))", Float64[]
            end
        end
    end
    
    # Check 6: Max absolute value within bounds
    max_abs_signal_threshold = 100.0
    max_abs = maximum(abs, model_prediction)
    if max_abs > max_abs_signal_threshold
        return false, "Signal exceeded threshold: max(|signal|) = $(round(max_abs, digits=2)) > $max_abs_signal_threshold", Float64[]
    end
    
    # Check 7: Model prediction doesn't show exponential growth (RMS check)
    n = length(model_prediction)
    w = min(200, max(50, n ÷ 5))
    max_rms_growth_threshold = 100.0
    if n ≥ 2w
        head_sum_sq = 0.0
        tail_sum_sq = 0.0
        @inbounds for i in 1:w
            head_sum_sq += model_prediction[i]^2
            tail_sum_sq += model_prediction[n - w + i]^2
        end
        head_rms = sqrt(head_sum_sq / w)
        tail_rms = sqrt(tail_sum_sq / w)
        if isfinite(head_rms) && head_rms > 0 && tail_rms > max_rms_growth_threshold * head_rms
            growth_factor = round(tail_rms / head_rms, digits=1)
            return false, "Prediction grew explosively (RMS growth $growth_factor x > threshold)", Float64[]
        end
    end
    
    return true, "", model_prediction
end



# ============================================================================
# LOSS FUNCTION IMPLEMENTATIONS
# ============================================================================

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
function weighted_mae(model_prediction, target::Data, fs, loss_settings::LossSettings, 
                      data_settings::Union{Nothing, DataSettings}=nothing)
    
    # Compute model PSD
    model_freqs, modeled_powers = ENEEGMA.compute_noisy_preprocessed_welch_psd(model_prediction, fs, loss_settings, data_settings)
    aligned_model = ENEEGMA._interpolate_psd(model_freqs, modeled_powers, target.freqs)
    
    # Without region masks: simple uniform MAE
    if target.freq_peak_metadata === nothing
        return Statistics.mean(abs.(target.powers .- aligned_model))
    end
    
    pm = target.freq_peak_metadata
    roi_mask = pm.roi_mask
    bg_mask = pm.bg_mask
    
    # Compute MAE in each region
    roi_diff = abs.(target.powers[roi_mask] .- aligned_model[roi_mask])
    bg_diff = abs.(target.powers[bg_mask] .- aligned_model[bg_mask])
    
    roi_mae = isempty(roi_diff) ? 0.0 : Statistics.mean(roi_diff)
    bg_mae = isempty(bg_diff) ? 0.0 : Statistics.mean(bg_diff)
    
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
# If noise_seed (in psd settings) is nothing/null → use non-deterministic random noise
# If noise_seed is an integer → use deterministic seeded RNG
# Note: Default noise_seed is 42 (deterministic). Set to nothing for non-deterministic behavior.
# The RNG is created once during loss setup and reused for all evaluations
function _measurement_noise_template(len::Int, rng::Union{Random.AbstractRNG, Nothing})
    if len <= 0
        return Float64[]
    end
    if rng === nothing
        # Non-deterministic RNG: use Julia's default random state
        return randn(len)
    else
        # Use pre-created seeded RNG
        return randn(rng, len)
    end
end

function apply_measurement_noise!(data::AbstractVector, sigma, rng::Union{Random.AbstractRNG, Nothing})
    sigma === nothing && return data
    if Base.iszero(sigma)
        return data
    end
    len = length(data)
    len == 0 && return data
    noise = _measurement_noise_template(len, rng)
    @inbounds @simd for i in 1:len
        data[i] += sigma * noise[i]
    end
    return data
end

