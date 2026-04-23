"""
Centralized evaluation utilities for model simulation and metric computation.

This module provides a single source of truth for:
- Simulating models with given parameters
- Extracting and processing predictions
- Computing PSDs
- Calculating metrics

This ensures consistency across optimization, evaluation, and analysis scripts.
"""

# ============================================================================
# PRIMARY FUNCTIONS
# ============================================================================

"""
    simulate_and_extract(net::Network, params, inits, settings::Settings; demean=true, remove_transient_period=true)

Simulate the network with given parameters and initial conditions, then extract and process
the brain source signals.

# Arguments
- `net::Network`: The network to simulate
- `params`: Parameter values (can be NamedTuple or Vector)
- `inits`: Initial condition values (can be NamedTuple or Vector)
- `settings::Settings`: Settings object containing simulation configuration
- `demean::Bool=true`: Whether to subtract the mean from the prediction
- `remove_transient_period::Bool=true`: Whether to skip the configured transient period from the simulation

# Returns
- `sol`: ODE solution object
- `df_sources::DataFrame`: Extracted and processed brain source time series for all nodes
- `success::Bool`: Whether simulation succeeded
- `error_msg::String`: Error message if simulation failed, empty otherwise

# Notes
When `remove_transient_period=true`, the transient period is discarded to allow
the system to settle into steady-state before analysis. Set to `false` to use the entire simulation
window (useful when comparing directly with observed data that doesn't have this offset).
"""

function simulate_and_extract(
    net::Network,
    params,
    inits,
    settings::Settings;
    demean::Bool=true,
    transient_period_duration::Float64=2.0
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
        df = sol2df(sol, net)

        # Extract all source signals and compute PSD for each
        df_sources, _ = ENEEGMA.extract_brain_sources(settings, net, df; return_source_expressions=true)
        
        # Calculate sampling frequency
        fs = 1.0 / simulation_settings.saveat
        
        # Remove transient period
        transient_duration = settings.data_settings.psd.transient_period_duration
        keep_idx = ENEEGMA.get_indices_after_transient_removal(df_sources.time, transient_duration, df_sources.time[1], fs)
        
        if !isempty(keep_idx)
            df_sources = df_sources[keep_idx, :]
        end

        if demean
            for col in names(df_sources)
                if col != :time
                    df_sources[!, col] .-= mean(df_sources[!, col])
                end
            end
        end
#=      # Extract brain source indices for all nodes
        brain_source_indices = ENEEGMA.get_eeg_output_indices(net, settings)
        
        # Extract transient duration from data_settings if available
        tspan = solver_kwargs[:tspan]
        transient_duration = settings.data_settings.psd.transient_period_duration
        fs_actual = 1.0 / solver_kwargs[:saveat]  # Infer sampling frequency from time step
        keep_idx = ENEEGMA.get_indices_after_transient_removal(sol.t, transient_duration, tspan[1], fs_actual)
        
        # Validate simulation success and extract per-node model predictions
        success, error_msg, model_predictions = ENEEGMA.validate_simulation_success(
            sol, brain_source_indices, keep_idx, tspan[1], tspan[2], transient_duration
        )
        
        if !success
            vwarn("$(error_msg)"; level=2)
            return sol, Float64[], false, error_msg
        end
        
        # Demean if requested (critical for PSD computation consistency)
        if demean
            for (node_name, model_pred) in model_predictions
                model_predictions[node_name] .-= mean(model_pred)
            end
        end =#
        
        return sol, df_sources, true, ""
        
    catch e
        error_msg = "Simulation error: $(e)"
        vwarn("$(error_msg)"; level=2)
        return nothing, DataFrame(), false, error_msg
    end
end

function _normalize_evaluation_metric_name(metric_name)::String
    metric_key = lowercase(String(metric_name))
    if metric_key == "loss"
        return "weighted_mae"
    elseif metric_key == "iae"
        return "weighted_iae"
    elseif metric_key == "weighted_mae" || metric_key == "weighted_iae" || metric_key == "r2"
        return metric_key
    end

    error("Unknown evaluation metric: $metric_name. Supported metrics: loss, weighted_mae, iae, weighted_iae, r2")
end

function _evaluation_node_names(data::Data)::Vector{String}
    return collect(keys(data.node_data))
end

function _node_metric_values_from_aligned_psds(metric_name::String,
                                               aligned_psd_dict::Dict{String, Vector{Float64}},
                                               data::Data)::Vector{Float64}
    metric_values = Float64[]

    for node_name in _evaluation_node_names(data)
        node_info = data.node_data[node_name]
        aligned_model = aligned_psd_dict[node_name]

        if metric_name == "weighted_mae"
            push!(metric_values, ENEEGMA._weighted_node_mae(aligned_model, node_info))
        elseif metric_name == "weighted_iae"
            push!(metric_values, ENEEGMA._weighted_node_iae(aligned_model, node_info))
        elseif metric_name == "r2"
            push!(metric_values, ENEEGMA.r2(aligned_model, node_info.powers))
        else
            error("Unknown evaluation metric: $metric_name. Supported metrics: weighted_mae, weighted_iae, r2")
        end
    end

    return metric_values
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
- `evaluation_metrics`: Which metrics to compute. Accepts `"loss"`/`"weighted_mae"`, `"iae"`/`"weighted_iae"`, and `"r2"`. Default: `["loss", "r2", "iae"]`
- `demean::Bool=true`: Whether to demean prediction before PSD computation

# Returns
A NamedTuple with fields:
- `sol`: ODE solution (or nothing if failed)
- `node_names::Vector{String}`: Node order used for per-node metric vectors
- `df_sources::DataFrame`: Time series predictions for all evaluated nodes
- `model_predictions::Dict{String, Vector{Float64}}`: Per-node time-domain predictions
- `psd_dict::Dict{String, Tuple{Vector{Float64}, Vector{Float64}}}`: Per-node model PSDs
- `success::Bool`: Whether simulation succeeded
- `error_msg::String`: Error message if failed
- `metrics::Dict{String, Vector{Float64}}`: Per-node metric values keyed by normalized names (`"weighted_mae"`, `"weighted_iae"`, `"r2"`)

"""
function evaluate_model(
    net::Network,
    params,
    inits,
    data::Data,
    settings::Settings;
    evaluation_metrics::AbstractVector=["loss", "r2", "iae"],
    demean::Bool=true,
    transient_period_duration::Float64=2.0
)
    data_settings = settings.data_settings
    loss_settings = settings.optimization_settings.loss_settings
    
    # Validate required settings
    if data_settings === nothing
        error("evaluate_model requires data_settings to be configured in settings object")
    end
    
    if loss_settings === nothing
        error("evaluate_model requires loss_settings to be configured in optimization_settings")
    end
    
    # Simulate and extract prediction
    sol, df_sources, success, error_msg = ENEEGMA.simulate_and_extract(
        net, params, inits, settings; demean=demean, transient_period_duration=transient_period_duration)
    
    metric_names = [ENEEGMA._normalize_evaluation_metric_name(metric_name) for metric_name in evaluation_metrics]
    node_names = ENEEGMA._evaluation_node_names(data)

    metrics = Dict{String, Vector{Float64}}()
    empty_psd_dict = Dict{String, Tuple{Vector{Float64}, Vector{Float64}}}()
    
    if !success || isempty(df_sources)
        vinfo("Returning empty results due to simulation failure"; level=2)
        for metric_name in metric_names
            metrics[metric_name] = fill(Inf, length(node_names))
        end
        
        return (
            sol=sol,
            node_names=node_names,
            df_sources=DataFrame(),
            model_predictions=Dict{String, Vector{Float64}}(),
            psd_dict=empty_psd_dict,
            success=false,
            error_msg=error_msg,
            metrics=metrics
        )
    end

    brain_source_indices = ENEEGMA.get_eeg_output_indices(net, settings)

    solver_kwargs = get_solver_kwargs(net.problem, settings.simulation_settings)
    tspan = solver_kwargs[:tspan]
    transient_duration = data_settings.psd.transient_period_duration
    fs_actual = 1.0 / solver_kwargs[:saveat]  # Infer sampling frequency from time step
    keep_idx = ENEEGMA.get_indices_after_transient_removal(sol.t, transient_duration, tspan[1], fs_actual)
    
    # Validate simulation success and extract per-node model predictions
    success, error_msg, model_predictions = ENEEGMA.validate_simulation_success(
        sol, brain_source_indices, keep_idx, tspan[1], tspan[2], transient_duration
    )

    if !success
        vinfo("Returning empty results due to validation failure"; level=2)
        for metric_name in metric_names
            metrics[metric_name] = fill(Inf, length(node_names))
        end

        return (
            sol=sol,
            node_names=node_names,
            df_sources=DataFrame(),
            model_predictions=Dict{String, Vector{Float64}}(),
            psd_dict=empty_psd_dict,
            success=false,
            error_msg=error_msg,
            metrics=metrics
        )
    end
    
    psd_dict, aligned_psd_dict = ENEEGMA._compute_model_psd_dict(
        model_predictions,
        data,
        data_settings.fs,
        loss_settings,
        data_settings
    )

    for metric_name in metric_names
        metric_value = try
            ENEEGMA._node_metric_values_from_aligned_psds(metric_name, aligned_psd_dict, data)
        catch e
            vwarn("Error computing metric '$(metric_name)': $(e)"; level=2)
            fill(NaN, length(node_names))
        end
        
        metrics[metric_name] = metric_value
    end
    
    return (
        sol=sol,
        node_names=node_names,
        df_sources=df_sources,
        model_predictions=model_predictions,
        psd_dict=psd_dict,
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





