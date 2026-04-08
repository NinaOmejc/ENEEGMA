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
        sol = simulate_problem(
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
    evaluate_model(net::Network, params, inits, data::TargetPSD, settings::Settings; 
                   compute_metrics=["loss", "r2"], demean=true)

Complete model evaluation: simulate, compute PSD, and calculate metrics.

# Arguments
- `net::Network`: The network to evaluate
- `params`: Parameter values (NamedTuple or Vector)
- `inits`: Initial condition values (NamedTuple or Vector)
- `data::TargetPSD`: Target data for comparison
- `settings::Settings`: Settings object
- `compute_metrics`: Which metrics to compute (can be empty)
- `demean::Bool=true`: Whether to demean prediction before PSD computation

# Returns
A NamedTuple with fields:
- `sol`: ODE solution (or nothing if failed)
- `model_prediction::Vector{Float64}`: Time series prediction
- `freqs::Vector{Float64}`: Frequency vector for PSD
- `model_psd::Vector{Float64}`: Model power spectral density
- `success::Bool`: Whether simulation succeeded
- `error_msg::String`: Error message if failed
- `metrics::Dict{String, Float64}`: Computed metrics (loss, r2, etc.)

"""
function evaluate_model(
    net::Network,
    params,
    inits,
    data::TargetPSD,
    settings::Settings;
    compute_metrics::AbstractVector=["loss", "r2"],
    demean::Bool=true
)
    data_settings = settings.data_settings
    loss_settings = settings.optimization_settings.loss_settings
    
    # Simulate and extract prediction
    sol, model_prediction, success, error_msg = simulate_and_extract(
        net, params, inits, settings; demean=demean)
    
    # Normalize metric names (allows empty vector like [])
    metric_names = String.(compute_metrics)

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
    iae_parts = nothing
    for metric_name in metric_names
        metric_value = try
            if metric_name == "loss"
                # Use the configured loss function from settings
                loss_function = get_metric_function(settings.optimization_settings.loss)
                loss_function(model_prediction, data, data_settings.fs, loss_settings)
            elseif metric_name == "r2"
                metric_r2(model_psd, data.powers)
            elseif metric_name == "iae" || metric_name == "iaep" || metric_name == "iaeb"
                iae_parts === nothing && (iae_parts = iae_decomposed(model_psd, data.powers, freqs,
                                                                        data_settings.task_type, loss_settings))
                metric_name == "iae"  ? iae_parts.full :
                metric_name == "iaep" ? iae_parts.peak :
                                        iae_parts.back            
            elseif metric_name == "maer"
                # Weighted MAE for resting state (peak weighted 2x)
                # Use peak_metadata if available, otherwise create simple alpha band mask
                if data.peak_metadata !== nothing
                    peak_mask = data.peak_metadata.peak_mask
                else
                    # Fallback to alpha band if no metadata
                    peak_mask = (freqs .>= 8.0) .& (freqs .<= 12.0)
                end
                metric_maer(model_psd, data.powers, freqs, peak_mask)
            elseif metric_name == "maes"
                # Weighted MAE for SSVEP (harmonics weighted 2x)
                # Extract stimulation frequency from loss_settings
                f0 = loss_settings.ssvep_stim_freq_hz
                H = loss_settings.ssvep_n_harmonics
                bw = loss_settings.ssvep_bandwidth_hz
                harmonic_mask = _harmonic_mask(freqs, f0, H, bw; fmin=loss_settings.fmin, fmax=loss_settings.fmax)
                metric_maes(model_psd, data.powers, freqs, harmonic_mask)
            elseif metric_name == "maef"
                # Weighted MAE that automatically switches between rest (maer) and SSVEP (maes)
                metric_maef(model_psd, data, freqs, data_settings.task_type, loss_settings)
            else
                # For any other metric name, try to get the function dynamically
                metric_function = get_metric_function(metric_name)
                metric_function(model_prediction, data, data_settings.fs, loss_settings)
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
    metric_maer(model_psd, target_psd, freqs, peak_mask)

Weighted Mean Absolute Error for resting state task.
Peak region (from metadata or fallback to 8-12 Hz) is weighted 2x stronger than background.
Reuses compute_peak_score and compute_background_score from losses.jl

# Arguments
- `model_psd`: Model power spectral density
- `target_psd`: Target power spectral density  
- `freqs`: Frequency vector
- `peak_mask`: Boolean mask indicating peak regions (from peak_metadata or manual)
"""
function metric_maer(model_psd::Vector{Float64}, target_psd::Vector{Float64}, freqs::Vector{Float64}, peak_mask::AbstractVector{Bool})
    # Use existing helper functions from losses.jl
    peak_score = compute_peak_score(model_psd, target_psd, peak_mask)
    background_score = compute_background_score(model_psd, target_psd, peak_mask)
    
    # Weight peak 2x stronger than background (2:1 ratio)
    return 1.0 * peak_score + 0.5 * background_score
end


"""
    metric_maes(model_psd, target_psd, freqs, harmonic_mask)

Weighted Mean Absolute Error for SSVEP task.
Harmonics of stimulation frequency are weighted 2x stronger than background.
Reuses compute_peak_score and compute_background_score from losses.jl

# Arguments
- `model_psd`: Model power spectral density
- `target_psd`: Target power spectral density
- `freqs`: Frequency vector
- `harmonic_mask`: Boolean mask indicating harmonic regions (computed using _harmonic_mask from losses.jl)
"""
function metric_maes(model_psd::Vector{Float64}, target_psd::Vector{Float64}, freqs::Vector{Float64}, harmonic_mask::AbstractVector{Bool})
    # Use existing helper functions from losses.jl
    harmonic_score = compute_peak_score(model_psd, target_psd, harmonic_mask)
    background_score = compute_background_score(model_psd, target_psd, harmonic_mask)
    
    # Weight harmonics 2x stronger than background (2:1 ratio)
    return 1.0 * harmonic_score + 0.5 * background_score
end


"""
    metric_maef(model_psd, target_psd, freqs, data, loss_settings)

Weighted Mean Absolute Error that automatically switches between rest and SSVEP tasks.
For rest tasks, uses maer (peak weighted 2x).
For SSVEP tasks, uses maes (harmonics weighted 2x).

# Arguments
- `model_psd`: Model power spectral density
- `target_psd`: Target power spectral density
- `freqs`: Frequency vector
- `data`: TargetPSD object containing powers and peak_metadata  
- `task_type`: String indicating the task type ("rest" or "ssvep")
- `loss_settings`: Loss settings containing SSVEP parameters
"""
function metric_maef(model_psd::Vector{Float64}, data::TargetPSD, freqs::Vector{Float64}, task_type::String, loss_settings)
    if task_type == "rest"
        # Use maer: peak region weighted 2x
        if data.peak_metadata !== nothing
            peak_mask = data.peak_metadata.peak_mask
        else
            # Fallback to alpha band if no metadata
            peak_mask = (freqs .>= 8.0) .& (freqs .<= 12.0)
        end
        return metric_maer(model_psd, data.powers, freqs, peak_mask)
    elseif task_type == "ssvep"
        # Use maes: harmonics weighted 2x
        f0 = loss_settings.ssvep_stim_freq_hz
        H = loss_settings.ssvep_n_harmonics
        bw = loss_settings.ssvep_bandwidth_hz
        harmonic_mask = _harmonic_mask(freqs, f0, H, bw; fmin=loss_settings.fmin, fmax=loss_settings.fmax)
        return metric_maes(model_psd, data.powers, freqs, harmonic_mask)
    else
        error("Unknown task_type: $(task_type). Expected 'rest' or 'ssvep'")
    end
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



# calculate area between curves Integrated absolute error (area between curves)
function l1_integral(a::Vector{Float64}, b::Vector{Float64}, freqs::Vector{Float64})
    e = abs.(a .- b)
    df = diff(freqs)
    area = sum(0.5 .* (e[1:end-1] .+ e[2:end]) .* df)
    return area
end

# Core: computes everything once (additive by construction)
function iae_decomposed(a::Vector{Float64}, b::Vector{Float64},
                        freqs::Vector{Float64}, task_type::String,
                        loss_settings::LossSettings)

    e    = abs.(a .- b)
    df   = diff(freqs)
    trap = 0.5 .* (e[1:end-1] .+ e[2:end]) .* df

    peak_point = if task_type == "rest"
        (freqs .>= loss_settings.peak_min_frequency_hz) .&
        (freqs .<= loss_settings.peak_max_frequency_hz)
    elseif task_type == "ssvep"
        f0 = loss_settings.ssvep_stim_freq_hz
        H  = loss_settings.ssvep_n_harmonics
        bw = loss_settings.ssvep_bandwidth_hz
        _harmonic_mask(freqs, f0, H, bw; fmin=loss_settings.fmin, fmax=loss_settings.fmax)
    else
        error("Unknown task_type: $task_type")
    end

    peak_interval = peak_point[1:end-1] .& peak_point[2:end]

    fit_point    = (freqs .>= loss_settings.fmin) .& (freqs .<= loss_settings.fmax)
    fit_interval = fit_point[1:end-1] .& fit_point[2:end]

    peak_interval .&= fit_interval
    back_interval = fit_interval .& .!peak_interval

    iae_peak = sum(trap .* peak_interval)
    iae_back = sum(trap .* back_interval)
    iae_full = sum(trap .* fit_interval)

    return (full = iae_full, peak = iae_peak, back = iae_back)
end

# Optional wrappers, if you like keeping old names:
l1_integral_peak(a,b,freqs,task,ls) = iae_decomposed(a,b,freqs,task,ls).peak
l1_integral_background(a,b,freqs,task,ls) = iae_decomposed(a,b,freqs,task,ls).back