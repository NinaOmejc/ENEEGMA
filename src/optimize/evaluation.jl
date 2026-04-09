"""
Centralized evaluation utilities for model simulation and metric computation.

This module provides a single source of truth for:
- Simulating models with given parameters
- Extracting and processing predictions
- Computing PSDs
- Calculating metrics

This ensures consistency across optimization, evaluation, and analysis scripts.
"""

"""
    simulate_and_extract(net::Network, params, inits, settings::Settings; demean=true)

Simulate the network with given parameters and initial conditions, then extract and process
the brain source prediction.

# Arguments
- `net::Network`: The network to simulate
- `params`: Parameter values (can be NamedTuple or Vector)
- `inits`: Initial condition values (can be NamedTuple or Vector)
- `settings::Settings`: Settings object containing simulation configuration
- `demean::Bool=true`: Whether to subtract the mean from the prediction

# Returns
- `sol`: ODE solution object
- `model_prediction::Vector{Float64}`: Extracted and processed brain source time series
- `success::Bool`: Whether simulation succeeded
- `error_msg::String`: Error message if simulation failed, empty otherwise
"""

function simulate_and_extract(
    net::Network,
    params,
    inits,
    settings::Settings;
    demean::Bool=true
)
    simulation_settings = settings.simulation_settings
    
    # Get solver and kwargs
    solver = get_solver(net.problem, simulation_settings)
    solver_kwargs = get_solver_kwargs(net.problem, simulation_settings)
    
    # Simulate
    try
        sol = simulate_network(
            net.problem;
            new_params=params,
            new_inits=inits,
            solver=solver,
            solver_kwargs=solver_kwargs
        )
        
        # Check solution status
        if !SciMLBase.successful_retcode(sol)
            error_msg = "Simulation failed with retcode: $(sol.retcode)"
            vprint("Warning: $(error_msg)", level=2)
            return sol, Float64[], false, error_msg
        end
        
        # Extract brain source
        brain_source_idx = get_brain_source_idx(net)
        
        # Filter time points (skip transient period)
        tspan = simulation_settings.tspan
        if tspan === nothing
            error("SimulationSettings.tspan must be specified")
        end
        transient_time = tspan[1] + 2.0  # Skip first 2 seconds after tspan start
        keep_idx = findall(t -> t >= transient_time, sol.t)
        
        if isempty(keep_idx)
            error_msg = "No simulation time points after transient period (t=$(transient_time)s)"
            vprint("Warning: $(error_msg)", level=2)
            return sol, Float64[], false, error_msg
        end
        
        # Extract and process prediction
        model_prediction = Vector{Float64}(sol[brain_source_idx, keep_idx])
        
        # Demean if requested (critical for PSD computation consistency)
        if demean
            model_prediction .-= mean(model_prediction)
        end
        
        return sol, model_prediction, true, ""
        
    catch e
        error_msg = "Simulation error: $(e)"
        vprint("Warning: $(error_msg)", level=2)
        # Return empty solution-like object
        return nothing, Float64[], false, error_msg
    end
end


"""
    evaluate_model(net::Network, params, inits, data::Data, settings::Settings; 
                   evaluation_metrics=["loss", "r2", "iae"], demean=true)

Complete model evaluation: simulate, compute PSD, and calculate metrics.

# Arguments
- `net::Network`: The network to evaluate
- `params`: Parameter values (NamedTuple or Vector)
- `inits`: Initial condition values (NamedTuple or Vector)
- `data::Data`: Target data for comparison
- `settings::Settings`: Settings object
- `evaluation_metrics`: Which metrics to compute (can be: "loss", "r2", "iae"). Default: ["loss", "r2", "iae"]
- `demean::Bool=true`: Whether to demean prediction before PSD computation

# Returns
A NamedTuple with fields:
- `sol`: ODE solution (or nothing if failed)
- `model_prediction::Vector{Float64}`: Time series prediction
- `freqs::Vector{Float64}`: Frequency vector for PSD
- `model_psd::Vector{Float64}`: Model power spectral density
- `success::Bool`: Whether simulation succeeded
- `error_msg::String`: Error message if failed
- `metrics::Dict{String, Float64}`: Computed metrics (loss, r2, iae)

"""
function evaluate_model(
    net::Network,
    params,
    inits,
    data::Data,
    settings::Settings;
    evaluation_metrics::AbstractVector=["loss", "r2", "iae"],
    demean::Bool=true
)
    data_settings = settings.data_settings
    loss_settings = settings.optimization_settings.loss_settings
    
    # Simulate and extract prediction
    sol, model_prediction, success, error_msg = simulate_and_extract(
        net, params, inits, settings; demean=demean)
    
    # Normalize metric names (allows empty vector like [])
    metric_names = String.(evaluation_metrics)

    # Initialize results
    metrics = Dict{String, Float64}()
    freqs = Float64[]
    model_psd = Float64[]
    
    if !success || isempty(model_prediction)
        # Return zeros/empty results for failed simulation
        vprint("Returning empty results due to simulation failure", level=2)
        model_psd = zeros(length(data.powers))
        freqs = copy(data.freqs)
        
        # Set metrics to sentinel values
        for metric_name in metric_names
            metrics[metric_name] = Inf  # or NaN, depending on preference
        end
        
        return (
            sol=sol,
            model_prediction=model_prediction,
            freqs=freqs,
            model_psd=model_psd,
            success=false,
            error_msg=error_msg,
            metrics=metrics
        )
    end
    
    # Compute PSD
    freqs, model_psd = _compute_noisy_psd_avg(
        model_prediction,
        data_settings.fs,
        loss_settings
    )
    
    # Compute requested metrics
    for metric_name in metric_names
        metric_value = try
            if metric_name == "loss"
                # Use the unified region-weighted loss function
                loss_function = get_metric_function()
                loss_function(model_prediction, data, data_settings.fs, loss_settings)
            elseif metric_name == "r2"
                r2(model_psd, data.powers)
            elseif metric_name == "iae"
                # Integrated absolute error using unified region weighting from freq_peak_metadata
                weighted_iae(model_psd, data.powers, freqs, data)
            else
                error("Unknown evaluation metric: $metric_name. Supported metrics: loss, r2, iae")
            end
        catch e
            vprint("Warning: Error computing metric '$(metric_name)': $(e)", level=2)
            NaN
        end
        
        metrics[metric_name] = metric_value
    end
    
    return (
        sol=sol,
        model_prediction=model_prediction,
        freqs=freqs,
        model_psd=model_psd,
        success=true,
        error_msg="",
        metrics=metrics
    )
end





"""
    prepare_params_and_inits(net::Network, param_values, init_values, 
                             tunable_params_symbols::Vector{Symbol}, setter::Function)

Convert parameter and initial condition values to the format needed for simulation.

# Arguments
- `net::Network`: The network
- `param_values`: Parameter values (can be Dict, NamedTuple, OrderedDict, or Vector)
- `init_values`: Initial condition values (can be Dict, NamedTuple, OrderedDict, or Vector)
- `tunable_params_symbols::Vector{Symbol}`: Symbols of tunable parameters
- `setter::Function`: Function to set parameters in the full parameter NamedTuple

# Returns
- `params`: NamedTuple of all parameters (tunable + fixed)
- `inits`: Vector or NamedTuple of initial conditions
"""
function prepare_params_and_inits(
    net::Network,
    param_values,
    init_values,
    tunable_params_symbols::Vector{Symbol},
    setter::Function
)
    # Convert param_values to vector if needed
    if param_values isa Dict || param_values isa OrderedDict
        param_vec = [Float64(param_values[String(sym)]) for sym in tunable_params_symbols]
    elseif param_values isa NamedTuple
        param_vec = [Float64(param_values[sym]) for sym in tunable_params_symbols]
    elseif param_values isa AbstractVector
        param_vec = Float64.(param_values)
    else
        error("Unsupported param_values type: $(typeof(param_values))")
    end
    
    # Use setter to create full parameter NamedTuple
    params = setter(net.problem.p, param_vec)
    
    # Convert init_values to vector if needed
    if init_values isa Dict || init_values isa OrderedDict
        inits = collect(values(init_values))
    elseif init_values isa NamedTuple
        inits = collect(values(init_values))
    elseif init_values isa AbstractVector
        inits = init_values
    else
        error("Unsupported init_values type: $(typeof(init_values))")
    end
    
    return params, inits
end



# Compute weighted integrated absolute error using unified freq_peak_metadata
function weighted_iae(model_psd::Vector{Float64}, target_psd::Vector{Float64},
                      freqs::Vector{Float64}, data::Data)
    """
    Compute integrated absolute error (area between curves) using region weighting from freq_peak_metadata.
    Weights ROI regions differently from background, consistent with loss computation.
    """
    e = abs.(model_psd .- target_psd)
    df = diff(freqs)
    trap = 0.5 .* (e[1:end-1] .+ e[2:end]) .* df
    
    if data.freq_peak_metadata === nothing
        # Fallback: unweighted total IAE if no metadata
        return sum(trap)
    end
    
    pm = data.freq_peak_metadata
    roi_mask = pm.roi_mask
    bg_mask = pm.bg_mask
    
    # Compute IAE in each region
    roi_iae = sum(trap .* roi_mask[1:end-1] .& roi_mask[2:end])
    bg_iae = sum(trap .* bg_mask[1:end-1] .& bg_mask[2:end])
    
    # Weighted combination (same as loss: ROI 2x stronger than background)
    roi_weight = pm.roi_weight
    bg_weight = pm.bg_weight
    total_weight = roi_weight + bg_weight
    
    if total_weight <= 0
        # Fallback to equal weighting
        return 0.5 * (roi_iae + bg_iae)
    end
    
    return (roi_weight * roi_iae + bg_weight * bg_iae) / total_weight
end



function r2(psd_model, psd_data)
    ss_res = sum((psd_data .- psd_model).^2)
    ss_tot = sum((psd_data .- mean(psd_data)).^2)
    return 1 - ss_res/ss_tot
end
