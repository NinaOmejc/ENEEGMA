"""
-------------
LOSSES - Module containing loss functions and 
         metrics used in optimization.
-------------

"""

# Main function that takes the loss function as an argument
"""
    get_metric_function(metric_name::Union{String, Symbol}="weighted_mae")::Function

Returns the specified metric function.

Supports:
- "weighted_mae": Unified region-weighted mean absolute error loss
- "weighted_iae": Integrated absolute error with region weighting

# Arguments
- `metric_name::Union{String, Symbol}`: Metric name

# Returns
- `metric_function`: The requested metric function
"""
function _canonical_metric_name(metric_name::Union{String, Symbol})::String
    metric_key = lowercase(String(metric_name))

    if metric_key == "weighted_mae"
        return "weighted_mae"
    elseif metric_key == "weighted_iae"
        return "weighted_iae"
    end

    error("Unknown metric: $metric_name. Supported: weighted_mae, weighted_iae")
end

function get_metric_function(metric_name::Union{String, Symbol}="weighted_mae")::Function
    canonical_metric_name = _canonical_metric_name(metric_name)

    if canonical_metric_name == "weighted_mae"
        return weighted_mae
    elseif canonical_metric_name == "weighted_iae"
        return weighted_iae
    end
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
    net           = args.net
    settings      = args.settings
    prob          = args.prob
    data          = args.data
    setter        = args.setter
    all_params    = args.all_params
    tspan         = args.tspan
    solver        = args.solver
    solver_kwargs = args.solver_kwargs
    loss_settings = args.loss_settings
    data_settings = haskey(args, :data_settings) ? args.data_settings : nothing

    n_inits = length(prob.u0)
    n_param_block = length(new_params) - n_inits
    @assert n_param_block >= 0 "Decision vector smaller than expected"

    θ = new_params[1:n_param_block]
    iv = new_params[n_param_block + 1 : n_param_block + n_inits]
    updated_all_params = setter(all_params, θ)
    new_prob = remake(prob; p=updated_all_params, u0=iv, tspan=tspan)
    sol = safe_solve(new_prob, solver; solver_kwargs=solver_kwargs)
    
    success, error_msg, _, model_predictions = ENEEGMA.extract_validated_model_predictions(
        sol,
        net,
        settings;
        demean=true
    )
    
    if !success
        return 1e9
    end

    loss = metric_fun(model_predictions, data, loss_settings, data_settings)
    
    if !isfinite(loss)
        return 1e9
    end
    return loss
end

function _source_dataframe_to_dict(df_sources::DataFrame)::Dict{String, Vector{Float64}}
    model_predictions = Dict{String, Vector{Float64}}()
    for col in names(df_sources)
        col_name = String(col)
        col_name == "time" && continue
        model_predictions[col_name] = Vector{Float64}(df_sources[!, col])
    end
    return model_predictions
end

function _demean_model_predictions!(model_predictions::Dict{String, Vector{Float64}})
    for predictions in values(model_predictions)
        predictions .-= Statistics.mean(predictions)
    end
    return model_predictions
end

function model_predictions_to_dataframe(times::AbstractVector{<:Real},
                                        model_predictions::Dict{String, Vector{Float64}};
                                        node_names::Union{Nothing, AbstractVector}=nothing)::DataFrame
    df_sources = DataFrame(time=Float64.(times))
    ordered_node_names = node_names === nothing ? collect(keys(model_predictions)) : String.(node_names)

    for node_name in ordered_node_names
        haskey(model_predictions, node_name) || error("No model prediction for node $node_name")
        df_sources[!, Symbol(node_name)] = model_predictions[node_name]
    end

    return df_sources
end

function prepare_source_dataframe_for_analysis(df_sources::DataFrame,
                                               fs::Real,
                                               transient_duration::Real;
                                               demean::Bool=true,
                                               keep_idx=nothing,
                                               source_names::Union{Nothing, AbstractVector}=nothing)
    isempty(df_sources) && return DataFrame(), Dict{String, Vector{Float64}}()

    times = Vector{Float64}(df_sources.time)
    keep_idx_use = keep_idx === nothing ?
        ENEEGMA.get_indices_after_transient_removal(times, Float64(transient_duration), times[1], Float64(fs)) :
        keep_idx
    isempty(keep_idx_use) && return DataFrame(), Dict{String, Vector{Float64}}()

    model_predictions = ENEEGMA._source_dataframe_to_dict(df_sources)
    for (node_name, predictions) in model_predictions
        model_predictions[node_name] = Vector{Float64}(predictions[keep_idx_use])
    end

    if demean
        ENEEGMA._demean_model_predictions!(model_predictions)
    end

    processed_df = ENEEGMA.model_predictions_to_dataframe(times[keep_idx_use],
                                                          model_predictions;
                                                          node_names=source_names)
    return processed_df, model_predictions
end

function extract_validated_model_predictions(sol,
                                             brain_source_indices::Dict{String, Int},
                                             tspan,
                                             saveat::Real,
                                             transient_duration::Real;
                                             demean::Bool=true
                                             )::Tuple{Bool, String, Vector{Float64}, Dict{String, Vector{Float64}}}
    fs_actual = 1.0 / Float64(saveat)
    keep_idx = ENEEGMA.get_indices_after_transient_removal(sol.t,
                                                           Float64(transient_duration),
                                                           Float64(tspan[1]),
                                                           fs_actual)

    success, error_msg, model_predictions = ENEEGMA.validate_simulation_success(
        sol,
        brain_source_indices,
        keep_idx,
        Float64(tspan[1]),
        Float64(tspan[2]),
        Float64(transient_duration)
    )

    if !success
        return false, error_msg, Float64[], Dict{String, Vector{Float64}}()
    end

    if demean
        ENEEGMA._demean_model_predictions!(model_predictions)
    end

    return true, "", Vector{Float64}(sol.t[keep_idx]), model_predictions
end

function extract_validated_model_predictions(sol,
                                             net::Network,
                                             settings::Settings;
                                             demean::Bool=true
                                             )::Tuple{Bool, String, Vector{Float64}, Dict{String, Vector{Float64}}}
    simulation_settings = settings.simulation_settings
    transient_duration = settings.data_settings.psd.transient_period_duration
    fs_actual = 1.0 / simulation_settings.saveat
    keep_idx = ENEEGMA.get_indices_after_transient_removal(sol.t,
                                                           Float64(transient_duration),
                                                           Float64(simulation_settings.tspan[1]),
                                                           fs_actual)

    success, error_msg = ENEEGMA._validate_solution_after_transient(sol,
                                                                    keep_idx,
                                                                    Float64(simulation_settings.tspan[1]),
                                                                    Float64(simulation_settings.tspan[2]))
    if !success
        return false, error_msg, Float64[], Dict{String, Vector{Float64}}()
    end

    node_names = [String(node.name) for node in net.nodes]
    model_predictions = ENEEGMA.extract_brain_sources(settings,
                                                      net,
                                                      sol;
                                                      node_names=node_names,
                                                      keep_idx=keep_idx,
                                                      output_column_style=:node_name,
                                                      return_type=:dict)
    isempty(model_predictions) && return false, "Failed to extract any brain source signals", Float64[], Dict{String, Vector{Float64}}()

    missing_node_names = [node_name for node_name in node_names if !haskey(model_predictions, node_name)]
    isempty(missing_node_names) || return false, "Failed to extract brain sources for nodes: $(join(missing_node_names, ", "))", Float64[], Dict{String, Vector{Float64}}()

    if demean
        ENEEGMA._demean_model_predictions!(model_predictions)
    end

    success, error_msg = ENEEGMA._validate_model_predictions(model_predictions)
    if !success
        return false, error_msg, Float64[], Dict{String, Vector{Float64}}()
    end

    return true, "", Vector{Float64}(sol.t[keep_idx]), model_predictions
end


function loss_empty(new_params, args)
    return NaN  # Or some default high loss value like 1e9
end


# ============================================================================
# HELPER FUNCTIONS FOR VALIDATION
# ============================================================================

function _validate_solution_after_transient(sol,
                                            keep_idx,
                                            tspan_start::Float64,
                                            tspan_end::Float64)::Tuple{Bool, String}
    if !SciMLBase.successful_retcode(sol)
        return false, "Simulation failed with retcode: $(sol.retcode)"
    end

    if isempty(keep_idx)
        return false, "No simulation time points after transient period"
    end

    expected_duration_sec = (tspan_end - tspan_start) / 1000.0
    actual_duration_sec = sol.t[end] - sol.t[1]
    if actual_duration_sec < 0.9 * expected_duration_sec
        return false, "Simulation terminated early (covered $(round(actual_duration_sec, digits=2))s of $(round(expected_duration_sec, digits=2))s)"
    end

    n_steps = length(sol.t)
    if n_steps >= 100
        n_sample = min(50, n_steps ÷ 5)

        for var_idx in 1:length(sol.u[1])
            early_sum_sq = 0.0
            for i in 1:n_sample
                early_sum_sq += abs(sol.u[i][var_idx])^2
            end
            early_rms = sqrt(early_sum_sq / n_sample)

            late_sum_sq = 0.0
            for i in (n_steps - n_sample + 1):n_steps
                late_sum_sq += abs(sol.u[i][var_idx])^2
            end
            late_rms = sqrt(late_sum_sq / n_sample)

            if early_rms > 0 && late_rms > 100.0 * early_rms
                return false, "State variable $var_idx exploded (RMS grew from $(round(early_rms, digits=2)) to $(round(late_rms, digits=2)))"
            end
        end
    end

    return true, ""
end

function _validate_model_predictions(model_predictions::Dict{String, Vector{Float64}})::Tuple{Bool, String}
    for (node_name, pred) in model_predictions
        if any(!isfinite, pred)
            return false, "Model prediction for node $node_name contains non-finite values (Inf or NaN)"
        end
    end

    max_abs_signal_threshold = 100.0
    for (node_name, pred) in model_predictions
        max_abs = maximum(abs, pred)
        if max_abs > max_abs_signal_threshold
            return false, "Signal for node $node_name exceeded threshold: max(|signal|) = $(round(max_abs, digits=2)) > $max_abs_signal_threshold"
        end
    end

    max_rms_growth_threshold = 100.0
    for (node_name, pred) in model_predictions
        n = length(pred)
        w = min(200, max(50, n ÷ 5))
        if n >= 2w
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
                return false, "Prediction for node $node_name grew explosively (RMS growth $growth_factor x > threshold)"
            end
        end
    end

    return true, ""
end

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
    
    success, error_msg = ENEEGMA._validate_solution_after_transient(sol, keep_idx, tspan_start, tspan_end)
    if !success
        return false, error_msg, Dict{String, Vector{Float64}}()
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

#= 
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
=#

# ============================================================================
# LOSS FUNCTION IMPLEMENTATIONS
# ============================================================================

function _compute_node_model_psd(model_prediction::AbstractVector{<:Real},
                                 node_info::NodeData,
                                 fs,
                                 loss_settings::LossSettings,
                                 data_settings::Union{Nothing, DataSettings}=nothing)
    if data_settings === nothing
        return ENEEGMA.compute_preprocessed_welch_psd(model_prediction, fs;
                                                      loss_settings=loss_settings,
                                                      data_settings=data_settings)
    end

    return ENEEGMA.compute_noisy_preprocessed_welch_psd(model_prediction,
                                                        fs,
                                                        loss_settings,
                                                        data_settings,
                                                        node_info)
end

function _metric_sampling_rate(data::Data,
                               data_settings::Union{Nothing, DataSettings}=nothing)::Float64
    return Float64(data.sampling_rate)
end

function _compute_model_psd_dict(model_predictions::Dict{String, Vector{Float64}},
                                 data::Data,
                                 loss_settings::LossSettings,
                                 data_settings::Union{Nothing, DataSettings}=nothing)
    fs = ENEEGMA._metric_sampling_rate(data, data_settings)
    return ENEEGMA._compute_model_psd_dict(model_predictions, data, fs, loss_settings, data_settings)
end

function _compute_model_psd_dict(model_predictions::Dict{String, Vector{Float64}},
                                 data::Data,
                                 fs,
                                 loss_settings::LossSettings,
                                 data_settings::Union{Nothing, DataSettings}=nothing)
    isempty(data.node_data) && error("Metric computation requires non-empty data.node_data")

    psd_dict = Dict{String, Tuple{Vector{Float64}, Vector{Float64}}}()
    aligned_psd_dict = Dict{String, Vector{Float64}}()

    for (node_name, node_info) in data.node_data
        if !haskey(model_predictions, node_name)
            error("No model prediction for node $node_name. Available: $(collect(keys(model_predictions)))")
        end

        model_freqs, modeled_powers = _compute_node_model_psd(model_predictions[node_name],
                                                              node_info,
                                                              fs,
                                                              loss_settings,
                                                              data_settings)
        psd_dict[node_name] = (model_freqs, modeled_powers)
        aligned_psd_dict[node_name] = ENEEGMA._interpolate_psd(model_freqs, modeled_powers, node_info.freqs)
    end

    return psd_dict, aligned_psd_dict
end

function _combine_region_metrics(roi_metric::Real, bg_metric::Real, pm)
    roi_weight = pm.roi_weight
    bg_weight = pm.bg_weight
    total_weight = roi_weight + bg_weight

    if total_weight <= 0
        return 0.5 * (Float64(roi_metric) + Float64(bg_metric))
    end

    return (roi_weight * Float64(roi_metric) + bg_weight * Float64(bg_metric)) / total_weight
end

function _weighted_node_mae(aligned_model::AbstractVector{<:Real}, node_info::NodeData)
    abs_diff = abs.(node_info.powers .- aligned_model)
    pm = node_info.freq_peak_metadata

    if pm === nothing
        return Statistics.mean(abs_diff)
    end

    roi_mask = Bool.(vec(pm.roi_mask))
    bg_mask = Bool.(vec(pm.bg_mask))
    length(roi_mask) == length(abs_diff) || error("ROI mask length must match PSD length for node $(node_info.channel)")
    length(bg_mask) == length(abs_diff) || error("Background mask length must match PSD length for node $(node_info.channel)")

    roi_mae = any(roi_mask) ? Statistics.mean(abs_diff[roi_mask]) : 0.0
    bg_mae = any(bg_mask) ? Statistics.mean(abs_diff[bg_mask]) : 0.0

    return _combine_region_metrics(roi_mae, bg_mae, pm)
end

function _weighted_node_iae(aligned_model::AbstractVector{<:Real}, node_info::NodeData)
    freqs = node_info.freqs
    if length(freqs) <= 1
        return 0.0
    end

    abs_diff = abs.(node_info.powers .- aligned_model)
    trap = 0.5 .* (abs_diff[1:end-1] .+ abs_diff[2:end]) .* diff(freqs)
    pm = node_info.freq_peak_metadata

    if pm === nothing
        return sum(trap)
    end

    roi_mask = Bool.(vec(pm.roi_mask))
    bg_mask = Bool.(vec(pm.bg_mask))
    length(roi_mask) == length(freqs) || error("ROI mask length must match frequency length for node $(node_info.channel)")
    length(bg_mask) == length(freqs) || error("Background mask length must match frequency length for node $(node_info.channel)")

    roi_seg = roi_mask[1:end-1] .& roi_mask[2:end]
    bg_seg = bg_mask[1:end-1] .& bg_mask[2:end]
    roi_iae = any(roi_seg) ? sum(trap[roi_seg]) : 0.0
    bg_iae = any(bg_seg) ? sum(trap[bg_seg]) : 0.0

    return _combine_region_metrics(roi_iae, bg_iae, pm)
end

"""
    weighted_mae(model_predictions::Dict{String, Vector{Float64}}, data::Data, loss_settings::LossSettings)

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
- `data::Data`: Target data with node_data dict containing per-node freq_peak_metadata
- Sampling frequency is resolved from `data.sampling_rate`
- `loss_settings::LossSettings`: Contains PSD settings and region weights

# Returns
- `Float64`: Averaged weighted MAE loss across all nodes
"""
function weighted_mae(model_predictions::Dict{String, Vector{Float64}}, data::Data, loss_settings::LossSettings,
                      data_settings::Union{Nothing, DataSettings}=nothing)
    _, aligned_psd_dict = _compute_model_psd_dict(model_predictions, data, loss_settings, data_settings)

    node_losses = Float64[]
    for (node_name, node_info) in data.node_data
        push!(node_losses, _weighted_node_mae(aligned_psd_dict[node_name], node_info))
    end

    return Statistics.mean(node_losses)
end

function weighted_mae(model_predictions::Dict{String, Vector{Float64}}, data::Data, fs, loss_settings::LossSettings, 
                      data_settings::Union{Nothing, DataSettings}=nothing)
    return weighted_mae(model_predictions, data, loss_settings, data_settings)
end

function weighted_mae(df_sources::DataFrame, data::Data, loss_settings::LossSettings,
                      data_settings::Union{Nothing, DataSettings}=nothing)
    return weighted_mae(_source_dataframe_to_dict(df_sources), data, loss_settings, data_settings)
end

function weighted_mae(df_sources::DataFrame, data::Data, fs, loss_settings::LossSettings,
                      data_settings::Union{Nothing, DataSettings}=nothing)
    return weighted_mae(df_sources, data, loss_settings, data_settings)
end

function weighted_iae(model_predictions::Dict{String, Vector{Float64}}, data::Data, loss_settings::LossSettings,
                      data_settings::Union{Nothing, DataSettings}=nothing)
    _, aligned_psd_dict = _compute_model_psd_dict(model_predictions, data, loss_settings, data_settings)

    node_iaes = Float64[]
    for (node_name, node_info) in data.node_data
        push!(node_iaes, _weighted_node_iae(aligned_psd_dict[node_name], node_info))
    end

    return Statistics.mean(node_iaes)
end

function weighted_iae(model_predictions::Dict{String, Vector{Float64}}, data::Data, fs, loss_settings::LossSettings,
                      data_settings::Union{Nothing, DataSettings}=nothing)
    return weighted_iae(model_predictions, data, loss_settings, data_settings)
end

function weighted_iae(df_sources::DataFrame, data::Data, loss_settings::LossSettings,
                      data_settings::Union{Nothing, DataSettings}=nothing)
    return weighted_iae(_source_dataframe_to_dict(df_sources), data, loss_settings, data_settings)
end

function weighted_iae(df_sources::DataFrame, data::Data, fs, loss_settings::LossSettings,
                      data_settings::Union{Nothing, DataSettings}=nothing)
    return weighted_iae(df_sources, data, loss_settings, data_settings)
end

function weighted_iae(model_psd::Vector{Float64}, target_psd::Vector{Float64},
                      freqs::Vector{Float64}, pm::Union{Nothing, NamedTuple}=nothing)
    length(model_psd) == length(target_psd) || error("model_psd and target_psd must have the same length")
    length(freqs) == length(target_psd) || error("freqs and target_psd must have the same length")
    length(freqs) <= 1 && return 0.0

    abs_diff = abs.(model_psd .- target_psd)
    trap = 0.5 .* (abs_diff[1:end-1] .+ abs_diff[2:end]) .* diff(freqs)

    if pm === nothing
        return sum(trap)
    end

    roi_mask = Bool.(vec(pm.roi_mask))
    bg_mask = Bool.(vec(pm.bg_mask))
    length(roi_mask) == length(freqs) || error("ROI mask length must match freqs length")
    length(bg_mask) == length(freqs) || error("Background mask length must match freqs length")

    roi_seg = roi_mask[1:end-1] .& roi_mask[2:end]
    bg_seg = bg_mask[1:end-1] .& bg_mask[2:end]
    roi_iae = any(roi_seg) ? sum(trap[roi_seg]) : 0.0
    bg_iae = any(bg_seg) ? sum(trap[bg_seg]) : 0.0

    return _combine_region_metrics(roi_iae, bg_iae, pm)
end

function r2(psd_model::AbstractVector{<:Real}, psd_data::AbstractVector{<:Real})
    ss_res = sum((psd_data .- psd_model).^2)
    ss_tot = sum((psd_data .- Statistics.mean(psd_data)).^2)
    if iszero(ss_tot)
        return iszero(ss_res) ? 1.0 : -Inf
    end
    return 1 - ss_res / ss_tot
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

