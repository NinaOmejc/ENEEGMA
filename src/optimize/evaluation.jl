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
    simulate_and_extract(net::Network, params, inits, settings::Settings; demean=true)

Simulate the network with given parameters and initial conditions, then extract and process
the brain source signals.

# Arguments
- `net::Network`: The network to simulate
- `params`: Parameter values (can be NamedTuple or Vector)
- `inits`: Initial condition values (can be NamedTuple or Vector)
- `settings::Settings`: Settings object containing simulation configuration
- `demean::Bool=true`: Whether to subtract the mean from the prediction
- The transient period is taken from `settings.data_settings.psd.transient_period_duration`

# Returns
- `sol`: ODE solution object
- `df_sources::DataFrame`: Extracted and processed brain source time series for all nodes
- `success::Bool`: Whether simulation succeeded
- `error_msg::String`: Error message if simulation failed, empty otherwise

# Notes
The configured transient period is discarded to allow the system to settle into steady-state before analysis.
"""

function simulate_and_extract_predictions(
    net::Network,
    params,
    inits,
    settings::Settings;
    demean::Bool=true
)
    simulation_settings = settings.simulation_settings
    solver = get_solver(net.problem, simulation_settings)
    solver_kwargs = get_solver_kwargs(net.problem, simulation_settings)

    try
        sol = simulate_network(
            net.problem;
            new_params=params,
            new_inits=inits,
            solver=solver,
            solver_kwargs=solver_kwargs
        )

        success, error_msg, times, model_predictions = ENEEGMA.extract_validated_model_predictions(
            sol,
            net,
            settings;
            demean=demean
        )

        return sol, times, model_predictions, success, error_msg
    catch e
        error_msg = "Simulation error: $(e)"
        vwarn("$(error_msg)"; level=2)
        return nothing, Float64[], Dict{String, Vector{Float64}}(), false, error_msg
    end
end

function simulate_and_extract(
    net::Network,
    params,
    inits,
    settings::Settings;
    demean::Bool=true
)
    sol, times, model_predictions, success, error_msg = ENEEGMA.simulate_and_extract_predictions(
        net,
        params,
        inits,
        settings;
        demean=demean
    )

    if !success
        return sol, DataFrame(), false, error_msg
    end

    node_names = [String(node.name) for node in net.nodes if haskey(model_predictions, String(node.name))]
    df_sources = ENEEGMA.model_predictions_to_dataframe(times, model_predictions; node_names=node_names)
    return sol, df_sources, true, ""
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

function _empty_node_metrics(node_names::Vector{String},
                             metric_names::Vector{String},
                             fill_value::Float64)
    metric_syms = Tuple(Symbol.(metric_names))
    metric_template = NamedTuple{metric_syms}(Tuple(fill(fill_value, length(metric_names))))
    metrics = Dict{String, typeof(metric_template)}()
    for node_name in node_names
        metrics[node_name] = metric_template
    end
    return metrics
end

function _node_metrics_from_metric_vectors(node_names::Vector{String},
                                           metric_names::Vector{String},
                                           metric_vectors::Dict{String, Vector{Float64}})
    metric_syms = Tuple(Symbol.(metric_names))
    metric_template = NamedTuple{metric_syms}(Tuple(fill(NaN, length(metric_names))))
    metrics = Dict{String, typeof(metric_template)}()

    for (node_idx, node_name) in enumerate(node_names)
        metric_tuple = NamedTuple{metric_syms}(Tuple(metric_vectors[metric_name][node_idx] for metric_name in metric_names))
        metrics[node_name] = metric_tuple
    end

    return metrics
end

function _evaluation_metric_vector(metrics_by_node,
                                   node_names::Vector{String},
                                   metric_name::String)::Vector{Float64}
    metric_sym = Symbol(metric_name)
    values = Float64[]

    for node_name in node_names
        if haskey(metrics_by_node, node_name) && hasproperty(metrics_by_node[node_name], metric_sym)
            push!(values, Float64(getproperty(metrics_by_node[node_name], metric_sym)))
        end
    end

    return values
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
- `metrics::Dict{String, NamedTuple}`: Per-node metric values keyed by node name

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
    
    # Validate required settings
    if data_settings === nothing
        error("evaluate_model requires data_settings to be configured in settings object")
    end
    
    if loss_settings === nothing
        error("evaluate_model requires loss_settings to be configured in optimization_settings")
    end
    
    metric_names = [ENEEGMA._normalize_evaluation_metric_name(metric_name) for metric_name in evaluation_metrics]
    node_names = ENEEGMA._evaluation_node_names(data)

    metrics = ENEEGMA._empty_node_metrics(node_names, metric_names, Inf)
    empty_psd_dict = Dict{String, Tuple{Vector{Float64}, Vector{Float64}}}()

    sol, times, model_predictions, success, error_msg = ENEEGMA.simulate_and_extract_predictions(
        net,
        params,
        inits,
        settings;
        demean=demean
    )

    if !success || isempty(model_predictions)
        vinfo("Returning empty results due to simulation failure"; level=2)
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

    df_sources = ENEEGMA.model_predictions_to_dataframe(times, model_predictions)
    psd_dict, aligned_psd_dict = ENEEGMA._compute_model_psd_dict(
        model_predictions,
        data,
        loss_settings,
        data_settings
    )

    metric_vectors = Dict{String, Vector{Float64}}()
    for metric_name in metric_names
        metric_value = try
            ENEEGMA._node_metric_values_from_aligned_psds(metric_name, aligned_psd_dict, data)
        catch e
            vwarn("Error computing metric '$(metric_name)': $(e)"; level=2)
            fill(NaN, length(node_names))
        end
        
        metric_vectors[metric_name] = metric_value
    end
    metrics = ENEEGMA._node_metrics_from_metric_vectors(node_names, metric_names, metric_vectors)
    
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





