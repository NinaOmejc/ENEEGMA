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
    brain_source_indices = args.brain_source_indices
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
    
    # Extract transient duration from data_settings if available
    transient_duration = data_settings.psd.transient_period_duration
    fs_actual = 1.0 / solver_kwargs[:saveat]  # Infer sampling frequency from time step
    keep_idx = ENEEGMA.get_indices_after_transient_removal(sol.t, transient_duration, tspan[1], fs_actual)
    
    # Validate simulation success and extract per-node model predictions
    success, error_msg, model_predictions = ENEEGMA.validate_simulation_success(
        sol, brain_source_indices, keep_idx, tspan[1], tspan[2], transient_duration
    )
    
    if !success
        return 1e9
    end

    # Demean each node's model prediction
    for node_name in keys(model_predictions)
        model_predictions[node_name] .-= Statistics.mean(model_predictions[node_name])
    end

    loss = metric_fun(model_predictions, data, 1/solver_kwargs[:saveat], loss_settings, data_settings)
    
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
        brain_source_indices::Dict{String, Int},
        keep_idx,
        tspan_start::Float64,
        tspan_end::Float64,
        transient_duration::Float64
    )::Tuple{Bool, String, Dict{String, Vector{Float64}}}

Comprehensive validation of simulation success with per-node model predictions.

Extracts model predictions for all nodes and performs safety checks:
1. ODE solver completed successfully (retcode)
2. Valid keep indices after transient removal
3. All model predictions are finite
4. Simulation wasn't terminated early (longer than expected based on tspan)
5. No individual state variables grew explosively
6. Model prediction max absolute values within bounds
7. Model predictions don't show exponential growth (RMS growth check)

# Returns
- `(success::Bool, error_msg::String, model_predictions::Dict{String, Vector{Float64}})`
"""
function validate_simulation_success(
    sol,
    brain_source_indices::Dict{String, Int},
    keep_idx,
    tspan_start::Float64,
    tspan_end::Float64,
    transient_duration::Float64
)::Tuple{Bool, String, Dict{String, Vector{Float64}}}
    
    # Check 1: Solver succeeded
    if !SciMLBase.successful_retcode(sol)
        return false, "Simulation failed with retcode: $(sol.retcode)", Dict{String, Vector{Float64}}()
    end
    
    # Check 2: Keep indices valid
    if isempty(keep_idx)
        return false, "No simulation time points after transient period", Dict{String, Vector{Float64}}()
    end
    
    # Extract predictions for all nodes
    model_predictions = Dict{String, Vector{Float64}}()
    try
        for (node_name, brain_idx) in brain_source_indices
            model_predictions[node_name] = Vector{Float64}(sol[brain_idx, keep_idx])
        end
    catch e
        return false, "Failed to extract brain sources: $e", Dict{String, Vector{Float64}}()
    end
    
    # Check 3: Finite values in all predictions
    for (node_name, pred) in model_predictions
        if any(!isfinite, pred)
            return false, "Model prediction for node $node_name contains non-finite values (Inf or NaN)", Dict{String, Vector{Float64}}()
        end
    end
    
    # Check 4: Simulation terminated early (check if got enough time points)
    # Calculate expected number of samples for full tspan
    expected_duration_sec = (tspan_end - tspan_start) / 1000.0  # Convert ms to seconds
    actual_duration_sec = (sol.t[end] - sol.t[1])
    if actual_duration_sec < 0.9 * expected_duration_sec  # Allow 10% tolerance
        return false, "Simulation terminated early (covered $(round(actual_duration_sec, digits=2))s of $(round(expected_duration_sec, digits=2))s)", Dict{String, Vector{Float64}}()
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
                return false, "State variable $var_idx exploded (RMS grew from $(round(early_rms, digits=2)) to $(round(late_rms, digits=2)))", Dict{String, Vector{Float64}}()
            end
        end
    end
    
    # Check 6: Max absolute value within bounds for each node
    max_abs_signal_threshold = 100.0
    for (node_name, pred) in model_predictions
        max_abs = maximum(abs, pred)
        if max_abs > max_abs_signal_threshold
            return false, "Signal for node $node_name exceeded threshold: max(|signal|) = $(round(max_abs, digits=2)) > $max_abs_signal_threshold", Dict{String, Vector{Float64}}()
        end
    end
    
    # Check 7: Model predictions don't show exponential growth (RMS check)
    max_rms_growth_threshold = 100.0
    for (node_name, pred) in model_predictions
        n = length(pred)
        w = min(200, max(50, n ÷ 5))
        if n ≥ 2w
            head_sum_sq = 0.0
            tail_sum_sq = 0.0
            @inbounds for i in 1:w
                head_sum_sq += pred[i]^2
                tail_sum_sq += pred[n - w + i]^2
            end
            head_rms = sqrt(head_sum_sq / w)
            tail_rms = sqrt(tail_sum_sq / w)
            if isfinite(head_rms) && head_rms > 0 && tail_rms > max_rms_growth_threshold * head_rms
                growth_factor = round(tail_rms / head_rms, digits=1)
                return false, "Prediction for node $node_name grew explosively (RMS growth $growth_factor x > threshold)", Dict{String, Vector{Float64}}()
            end
        end
    end
    
    return true, "", model_predictions
end


"""
    validate_simulation_success(sol, brain_source_idx::Int, keep_idx, tspan_start, tspan_end, transient_duration)

Backward-compatibility overload for single-node evaluation.
Extracts the single prediction corresponding to the given brain_source_idx.

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
    # Create a single-element dict and call the dict version
    brain_source_indices = Dict("node_1" => brain_source_idx)
    success, error_msg, model_predictions = validate_simulation_success(
        sol, brain_source_indices, keep_idx, tspan_start, tspan_end, transient_duration
    )
    
    if success
        # Extract the single prediction from the dict
        model_prediction = first(values(model_predictions))
        return true, "", model_prediction
    else
        return false, error_msg, Float64[]
    end
end


# ============================================================================
# LOSS FUNCTION IMPLEMENTATIONS
# ============================================================================

"""
    weighted_mae(model_predictions::Dict{String, Vector{Float64}}, target::Data, fs, loss_settings::LossSettings)

Unified frequency-domain MAE loss with per-node, region-based weighting.

For each node:
1. Extracts the node's model prediction from the dict
2. Computes model PSD from that prediction
3. Computes MAE between model and target PSDs
4. Applies region-based weighting (ROI vs background)
5. Averages losses across all nodes with equal weighting

Masks and weights are pre-computed and stored in each node's freq_peak_metadata during data preparation.
If no metadata is available, defaults to uniform (unweighted) MAE.

# Arguments
- `model_predictions::Dict{String, Vector{Float64}}`: Per-node time-domain model predictions
- `target::Data`: Target data with node_data dict containing per-node freq_peak_metadata
- `fs::Float64`: Sampling frequency
- `loss_settings::LossSettings`: Contains PSD settings and region weights

# Returns
- `Float64`: Averaged weighted MAE loss across all nodes
"""
function weighted_mae(model_predictions::Dict{String, Vector{Float64}}, target::Data, fs, loss_settings::LossSettings, 
                      data_settings::Union{Nothing, DataSettings}=nothing)
    
    if isempty(target.node_data)
        error("weighted_mae: Data has no node data loaded")
    end
    
    node_losses = Float64[]
    
    # Compute loss for each node using its own model prediction
    for (node_name, node_info) in target.node_data
        if !haskey(model_predictions, node_name)
            error("weighted_mae: No model prediction for node $node_name. Available: $(keys(model_predictions))")
        end
        
        model_pred = model_predictions[node_name]
        
        # Compute model PSD for this node's prediction
        model_freqs, modeled_powers = ENEEGMA.compute_noisy_preprocessed_welch_psd(model_pred, fs, loss_settings, data_settings)
        aligned_model = ENEEGMA._interpolate_psd(model_freqs, modeled_powers, node_info.freqs)
        
        # Compute loss for this node
        if node_info.freq_peak_metadata === nothing
            # Uniform MAE
            node_loss = Statistics.mean(abs.(node_info.powers .- aligned_model))
        else
            pm = node_info.freq_peak_metadata
            roi_mask = pm.roi_mask
            bg_mask = pm.bg_mask
            
            # Compute MAE in each region
            roi_diff = abs.(node_info.powers[roi_mask] .- aligned_model[roi_mask])
            bg_diff = abs.(node_info.powers[bg_mask] .- aligned_model[bg_mask])
            
            roi_mae = isempty(roi_diff) ? 0.0 : Statistics.mean(roi_diff)
            bg_mae = isempty(bg_diff) ? 0.0 : Statistics.mean(bg_diff)
            
            # Weighted combination
            roi_weight = pm.roi_weight
            bg_weight = pm.bg_weight
            total_weight = roi_weight + bg_weight
            
            if total_weight <= 0
                node_loss = 0.5 * (roi_mae + bg_mae)
            else
                node_loss = (roi_weight * roi_mae + bg_weight * bg_mae) / total_weight
            end
        end
        
        push!(node_losses, node_loss)
    end
    
    # Combine per-node losses: average (equal weighting for all nodes)
    combined_loss = Statistics.mean(node_losses)
    
    return combined_loss
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

