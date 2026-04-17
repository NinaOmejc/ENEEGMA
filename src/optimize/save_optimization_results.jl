# MAIN FUNCTION
function save_optimization_results(optsol::SciMLBase.OptimizationSolution, 
                                    optlogger::Vector{OptLogEntry},       
                                    setter::Function,  
                                    net::Network, 
                                    data::Data,
                                    settings::Settings;
                                    blocks::Union{Nothing, Any}=nothing,
                                    restart_idx::Union{Int, Nothing}=nothing,
                                    hyperparam_combo::Union{Tuple, Nothing}=nothing,
                                    hyperparam_idx::Union{Int, Nothing}=nothing,
                                    hyperparam_keys::Union{Vector{String}, Nothing}=nothing,
                                    true_parameters::Union{OrderedDict{Num, Float64}, Dict{String, Tuple{Vararg{Float64}}}, OrderedDict{String, Tuple{Vararg{Float64}}}, Nothing}=nothing
                                    )::Nothing

    general_settings = settings.general_settings
    simulation_settings = settings.simulation_settings
    data_settings = settings.data_settings
    optimization_settings = settings.optimization_settings
    loss_settings = optimization_settings.loss_settings

    blocks = blocks === nothing ? ENEEGMA.prepare_optimization_blocks(net, optimization_settings) : blocks
    params_native, inits_native, min_optlogger_loss = ENEEGMA.best_params_inits(optsol, optlogger, blocks)
    param_range_entries, init_range_entries = ENEEGMA.native_range_entries(net, blocks)
    best_loss = min(optsol.minimum, min_optlogger_loss)

    # Use provided optimization output directory, or construct from settings
    if settings.optimization_settings.output_dir === nothing
        candidate_path = ENEEGMA.construct_output_dir(general_settings, settings.network_settings)
        optimization_path = ENEEGMA.find_next_numbered_folder(candidate_path, "optimization")
        mkpath(optimization_path)
    else
        optimization_path = settings.optimization_settings.output_dir 
    end
    
    # Create figures subdirectory for all plots
    figures_dir = joinpath(optimization_path, "figures")
    mkpath(figures_dir)

    restart_str = restart_idx === nothing ? "" : "_r$(restart_idx)"
    hyperparam_str = hyperparam_idx === nothing ? "" : "_h$(hyperparam_idx)"
    # Include exp_name at the beginning for all output files
    base_prefix = "$(general_settings.exp_name)_$(net.name)$(hyperparam_str)$(restart_str)"

    # Save loss evolution plot to figures folder
    fname_loss = "$(base_prefix)_loss_evolution_over_iterations.png"
    path_loss = joinpath(figures_dir, fname_loss)
    ENEEGMA.plot_loss_over_iterations(optlogger, general_settings, path_loss)
    
    # Save optlogger CSV to main optimization folder
    ENEEGMA.save_optlogger(optlogger, settings; fullfname_csv=joinpath(optimization_path, "$(base_prefix)_optlogger.csv"))

    # Simulate and evaluate model with best parameters using centralized evaluate_model function
    opt_params = setter(net.problem.p, params_native);

    result = ENEEGMA.evaluate_model(
        net, opt_params, inits_native, data, settings;
        evaluation_metrics=["loss", "r2", "iae"],
        demean=true,
        transient_period_duration=settings.data_settings.psd.transient_period_duration
    );
    
    if !result.success
        vwarn("Failed to simulate model with best parameters: $(result.error_msg)"; level=2)
        vwarn("Attempting to save results with fallback values..."; level=2)
    end
    
    times = data.times
    model_prediction = result.model_prediction
    freqs = result.freqs
    modeled_powers = result.model_psd
    
    ENEEGMA.save_modeled_psd(modeled_powers, freqs, settings; 
                            fullfname_csv=joinpath(optimization_path, "$(base_prefix)_modeled_psd.csv"))

    fname_obs_ts = "$(base_prefix)_observed_vs_modeled_timeseries.png"
    path_obs_ts = joinpath(figures_dir, fname_obs_ts)
    # Use unified visualization function from visualization.jl
    ENEEGMA.plot_timeseries_windows(times, model_prediction;
                                     observed_signal=data.signal,
                                     zoom_window=(2.0, 5.0),
                                     title_prefix="Observed vs Modeled Timeseries",
                                     fullfname_fig=path_obs_ts,
                                     general_settings=general_settings);

    fname_obs_freq = "$(base_prefix)_observed_vs_modeled_spectrum.png"
    path_obs_freq = joinpath(figures_dir, fname_obs_freq)
    # Use unified visualization function from visualization.jl
    ENEEGMA.plot_psd_comparison(freqs, modeled_powers, data.powers;
                                title="Power Spectral Density Comparison",
                                fullfname_fig=path_obs_freq,
                                general_settings=general_settings)

    # PLOT: Parameter exploration
    fname_param = "$(base_prefix)_parameter_exploration.png"
    path_param = joinpath(figures_dir, fname_param)
    ENEEGMA.plot_param_exploration(optlogger, net; 
                                true_parameters=true_parameters,
                                loss_settings=loss_settings,
                                fullfname_fig=path_param,
                                general_settings=general_settings)

    # PLOT: Frequency spectra evolution
    fname_freq_plot = "$(base_prefix)_spectrum_evolution.png"
    fname_freq_anim = "$(base_prefix)_spectrum_evolution.gif"
    freq_plot_path = joinpath(figures_dir, fname_freq_plot)
    freq_anim_path = joinpath(figures_dir, fname_freq_anim)
    ENEEGMA.plot_psd_spetra_evolution(optlogger, net, setter;
                    data=data,
                    settings=settings,
                    fullfname_fig=freq_plot_path,
                    fullfname_anim=freq_anim_path)

    # Use metrics from evaluate_model for consistency with local re-runs
    recomputed_loss = result.success ? get(result.metrics, "loss", best_loss) : best_loss
    r2_metric = result.success ? get(result.metrics, "r2", NaN) : NaN
    iae_metric = result.success ? get(result.metrics, "iae", NaN) : NaN
    
    # Build best_params dict with normalized names
    best_opt_params_named = OrderedDict{String, Any}()
    if !isempty(blocks.tunable_params_symbols)
        for (sym, val) in zip(blocks.tunable_params_symbols, params_native)
            normalized_name = normalize_parameter_name(String(sym))
            best_opt_params_named[normalized_name] = val
        end
    end
    
    # Build initial_states dict with normalized state variable names
    init_names = if net.problem.u0 isa NamedTuple
        String.(keys(net.problem.u0))
    else
        # Get actual state variable names from the network
        state_vars = get_state_vars(net.vars)
        string.(get_symbols(state_vars))
    end
    initial_states_dict = OrderedDict{String, Any}()
    for (init_name, init_val) in zip(init_names, inits_native)
        normalized_name = normalize_parameter_name(init_name)
        initial_states_dict[normalized_name] = init_val
    end

    duration_sec = (optlogger[end].time - optlogger[1].time).value
    
    # Build metadata section with settings file reference
    timestamp_str = Dates.format(Dates.now(), "yyyy-mm-ddTHH:MM:SSZ")
    # Settings file is saved in the experiment folder (exp_name subfolder)
    settings_path = joinpath(general_settings.path_out, general_settings.exp_name, "settings.json")
    # Convert to forward slashes for cross-platform JSON compatibility
    settings_path_normalized = replace(settings_path, "\\" => "/")
    
    metadata = OrderedDict(
        "settings_file" => settings_path_normalized,
        "timestamp" => timestamp_str,
        "duration_seconds" => duration_sec,
        "restart_idx" => restart_idx,
        "hyperparam_idx" => hyperparam_idx
    )
    
    # Build optimization_results section with all loss metrics
    optimization_results_dict = OrderedDict(
        "best_loss" => best_loss,
        "loss_final_recomputed" => recomputed_loss,
        "r2" => r2_metric,
        "iae" => iae_metric,
        "retcode" => string(optsol.retcode),
        "iterations_completed" => length(optlogger)
    )
    
    # Build hyperparam_adaptation section (empty dict if no hyperparams)
    hyperparam_adaptation_dict = OrderedDict{String, Any}()
    if hyperparam_combo !== nothing
        # Map hyperparam keys to values
        for (key, val) in zip(hyperparam_keys, hyperparam_combo)
            hyperparam_adaptation_dict[key] = val
        end
    end

    # Build main results structure with optimization_results first, then metadata and other sections
    results = OrderedDict{String, Any}(
        "optimization_results" => optimization_results_dict,
        "metadata" => metadata,
        "best_parameters" => best_opt_params_named,
        "initial_states" => initial_states_dict,
        "hyperparam_adaptation" => hyperparam_adaptation_dict,
    )
    
    # Add hyperparam_idx to results if present
    if hyperparam_idx !== nothing
        results["hyperparam_idx"] = hyperparam_idx
    end

    if restart_idx === nothing
        results["parameter_ranges"] = param_range_entries
        results["initial_state_ranges"] = init_range_entries
    end
    
    # Add full configuration if requested
    if optimization_settings.include_settings_in_results_output
        results["configuration"] = settings_to_dict(settings)
    end

    fname_results = "$(base_prefix)_optimization_results.json"
    results_path = joinpath(optimization_path, fname_results)
    write_compact_json(results_path, results)

end

function best_params_inits(optsol::SciMLBase.OptimizationSolution,
                           optlogger::Vector{OptLogEntry},
                           blocks)
    n_state_vars = length(blocks.initial_values_native)
    n_params = length(optsol.minimizer) - n_state_vars
    n_params < 0 && error("Optimization solution length is inconsistent with state/measurement-noise counts.")
    length(blocks.param_spec) == n_params || error("Mismatch between tunable parameter count and optimization solution length.")
    length(blocks.init_spec) == n_state_vars || error("Mismatch between initial-state count and optimization solution length.")

    decoded_solution = materialize_logged_params(optsol.minimizer, blocks.param_spec, blocks.init_spec)
    params_native = decoded_solution[1:n_params]
    inits_native = decoded_solution[n_params + 1 : n_params + n_state_vars]
    
    return params_native, inits_native, optsol.minimum
end    
#=     
    if isempty(optlogger)
        return params_native, inits_native, optsol.minimum
    end 

    best_optlog_iter = optlogger[argmin([optlogger[i].loss for i in 1:length(optlogger)])]
    if best_optlog_iter.params !== nothing &&
       length(best_optlog_iter.params) == n_params + n_state_vars &&
       round(best_optlog_iter.loss, digits=6) < round(optsol.minimum, digits=6)
        params_native = best_optlog_iter.params[1:n_params]
        inits_native = best_optlog_iter.params[n_params + 1 : n_params + n_state_vars]
        vinfo("Using best logged parameters (loss=$(round(best_optlog_iter.loss, digits=6))) instead of optsol.minimizer (loss=$(round(optsol.minimum, digits=6)))"; level=2)
    end
    return params_native, inits_native, round(best_optlog_iter.loss, digits=6)
end
=#

function save_optlogger(optlogger, settings; fullfname_csv::String="optimization_history.csv")
    if settings.optimization_settings.save_optimization_history
        CSV.write(fullfname_csv, DataFrame(optlogger))
    end
end

function save_modeled_psd(modeled_powers::Vector{Float64},
                          freqs::Vector{Float64},
                          settings::Settings; 
                          fullfname_csv::String="modeled_psd.csv")::Nothing
    if settings.optimization_settings.save_modeled_psd
        CSV.write(fullfname_csv, DataFrame(freq = freqs, psd = modeled_powers))
    end
    return nothing
end

const JSON_SETTINGS_EXCLUDE_FIELDS = Set(["reference_freqs", "reference_psd", "workspace"])

function write_compact_json(path::AbstractString, data; digits::Int=3)
    prepared = _prepare_json_value(data, digits)
    open(path, "w") do io
        JSON.print(io, prepared, 2)
        println(io)
    end
    vinfo("Saved optimization results to JSON: $path"; level=2)
end

function _prepare_json_value(value, digits::Int)
    return _prepare_json_value(value, digits, IdDict{Any, Bool}())
end

function _prepare_json_value(value, digits::Int, seen::IdDict{Any, Bool})
    if value isa AbstractDict
        container = OrderedDict{Any, Any}()
        for (k, v) in pairs(value)
            if string(k) in JSON_SETTINGS_EXCLUDE_FIELDS
                continue
            end
            container[k] = _prepare_json_value(v, digits, seen)
        end
        return container
    elseif value isa NamedTuple
        container = OrderedDict{Any, Any}()
        for (k, v) in pairs(value)
            if string(k) in JSON_SETTINGS_EXCLUDE_FIELDS
                continue
            end
            container[String(k)] = _prepare_json_value(v, digits, seen)
        end
        return container
    elseif value isa AbstractVector
        return [_prepare_json_value(value[i], digits, seen) for i in eachindex(value)]
    elseif value isa AbstractMatrix
        rows, cols = size(value)
        out = Vector{Any}(undef, rows)
        for i in 1:rows
            row = Vector{Any}(undef, cols)
            for j in 1:cols
                row[j] = _prepare_json_value(value[i, j], digits, seen)
            end
            out[i] = row
        end
        return out
    elseif value isa AbstractArray
        return _prepare_json_value(collect(value), digits, seen)
    elseif value isa Tuple
        return [_prepare_json_value(val, digits, seen) for val in value]
    elseif value isa Number
        return isfinite(value) ? round(value; digits=digits) : 1e9
    elseif value isa AbstractString || value isa Bool || value === nothing
        return value
    elseif value isa Symbol
        return String(value)
    elseif Base.isstructtype(typeof(value))
        track = Base.ismutabletype(typeof(value))
        if track
            if haskey(seen, value)
                return "[[Circular $(typeof(value))]]"
            end
            seen[value] = true
        end
        container = OrderedDict{Any, Any}()
        for field in fieldnames(typeof(value))
            if String(field) in JSON_SETTINGS_EXCLUDE_FIELDS
                continue
            end
            container[String(field)] = _prepare_json_value(getfield(value, field), digits, seen)
        end
        track && delete!(seen, value)
        return container
    else
        return string(value)
    end
end


function plot_observed_vs_modelled_ts(data_predicted::Matrix{Float64},
                                      data_observed::Matrix{Float64}; 
                                      times::Vector{Float64}=Float64[],
                                      xlims::Tuple{<:Real, <:Real}=(0.0, 10.0),
                                      fullfname_fig::String="obs_vs_mod_ts.png",
                                      general_settings::GeneralSettings=nothing
    )::Nothing

    try
        titles = ["Var $i" for i in 1:size(data_observed, 2)]        
        p = plot(layout = (size(data_observed, 2), 2), xlabel="", ylabel="", size=(800, 800))

        # Loop through each subplot and add the observed and predicted data
        for i in 1:size(data_observed, 2)
            if i == 1
                plot!(p[i, 1], times, data_observed[:, i], label="Observed data", ylabel=titles[i], legend=true, xlabel="")
            else
                plot!(p[i, 1], times, data_observed[:, i], label="Observed data", ylabel=titles[i], legend=false, xlabel="")
            end
            plot!(p[i, 1], times, data_predicted[:, i], label="Model prediction", legend=false, xlabel="")
            plot!(p[i, 2], times, data_observed[:, i], label="Observed data", legend=false, xlabel="", xlims=xlims)
            plot!(p[i, 2], times, data_predicted[:, i], label="Model prediction", legend=false, xlabel="", xlims=xlims)
        end
        # add legend to the first subplot

        savefig(p, fullfname_fig);        
    catch
        vwarn("Simulation failed. Could not plot observed vs modelled time series."; level=2)
    end

    return nothing
end


"""
plot_observed_vs_modellepsd(sol, data_observed; 
        fullfname_fig, fmin, fmax, 
        use_log_scale, titles)

Plot observed vs. modeled data in frequency domain using periodograms.

# Arguments
- `sol`: The ODE solution
- `data_observed`: Matrix of observed time series data
- `fullfname_fig`: Full file path for saving the figure
- `fmin`: Minimum frequency to display (Hz)
- `fmax`: Maximum frequency to display (Hz)
- `use_log_scale`: Whether to use logarithmic scales (default: true)
- `titles`: Vector of variable titles for subplots

# Returns
- `Nothing`: Saves the plot to the specified path
"""
function plot_observed_vs_modelled_psd(data_modeled::Union{Vector{Float64}, Matrix{Float64}},
                                        data_observed::Union{Vector{Float64}, Matrix{Float64}},
                                        freqs::Vector{Float64};
                                        fullfname_fig::String="obs_vs_mod_freq.png",
                                        general_settings::Union{Nothing, GeneralSettings}=nothing
    )::Nothing
    if general_settings !== nothing && !general_settings.make_plots
        return nothing
    end

    # Check for dimension mismatch (compare frequency axis against first dimension)
    n_freqs = length(freqs)
    n_modeled = data_modeled isa AbstractMatrix ? size(data_modeled, 1) : length(data_modeled)
    n_observed = data_observed isa AbstractMatrix ? size(data_observed, 1) : length(data_observed)

    if n_modeled != n_observed || n_modeled != n_freqs
        vwarn("Cannot plot frequency comparison: dimension mismatch (modeled=$n_modeled, observed=$n_observed, freqs=$n_freqs)"; level=2)
        vwarn("  This usually means the simulation was too short or failed. Skipping plot."; level=2)
        return nothing
    end
    # Create a figure
    p = plot(layout = (size(data_observed, 2), 1), size=(800, 800),
                legend=:topright, xminorgrid=true, yminorgrid=true)

    plot!(p, freqs, data_modeled, 
            label="Model", 
            xlabel="Frequency (Hz)",
            ylabel="Log Power",
            linewidth=1,
            alpha=1.)

    plot!(p, freqs, data_observed, 
            label="Observed", 
            linewidth=1,
            color=:black, 
            alpha=1.)
    
    plot!(p, plot_title="Frequency Domain Comparison", titlefontsize=12)

    try
        savefig(p, fullfname_fig)
    catch e
        vwarn("Failed to save frequency domain comparison plot to $fullfname_fig: $e"; level=2)
    end

    return nothing
end

function plot_observed_vs_modelled_psd(data_modeled::Union{Vector{Float64}, Matrix{Float64}},
                                        data::Data;
                                        fullfname_fig::String="obs_vs_mod_freq.png",
                                        general_settings::Union{Nothing, GeneralSettings}=nothing
    )::Nothing
    return plot_observed_vs_modelled_psd(data_modeled, data.powers, data.freqs;
                                          fullfname_fig=fullfname_fig,
                                          general_settings=general_settings)
end



function plot_param_exploration(optlogger::Vector{OptLogEntry},
                                net::Network;
                                true_parameters::Union{OrderedDict{Num, Float64}, Dict{String, Tuple{Vararg{Float64}}}, OrderedDict{String, Tuple{Vararg{Float64}}}, Nothing}=nothing,
                                fullfname_fig::String="param_exploration.png",
                                loss_settings::Union{Nothing, LossSettings}=nothing,
                                general_settings::Union{Nothing, GeneralSettings}=nothing)::Nothing
    if general_settings !== nothing && !general_settings.make_plots
        return nothing
    end

    # Get ALL tunable parameters from network (includes tscale and any other added params)
    tunable_params = get_symbols(get_tunable_params(net.params); sort=true)
    tunable_syms = Symbol.(tunable_params)
    tunable_params_lb, tunable_params_ub = get_param_minmax_values(net.params; p_subset=tunable_params, return_type="vector")

    # if true_parameters is given, use those; otherwise just use midpoints of the bounds
    tunable_params_default_vals =
        true_parameters === nothing ?
            0.5 .* (tunable_params_lb .+ tunable_params_ub) :
            Float64[first(t) for t in values(true_parameters)]

    # Get initial conditions (state variables)
    state_vars = get_state_vars(net.vars)
    inits_lb, inits_ub = get_var_minmax_values(state_vars; return_type="vector")
    inits_default_vals = zeros(length(state_vars.vars))  # assume default inits are zero

    # Combine parameters and initial conditions
    tunables_lb = vcat(tunable_params_lb, inits_lb)
    tunables_ub = vcat(tunable_params_ub, inits_ub)
    tunables_default_vals = vcat(tunable_params_default_vals, inits_default_vals)
    tunable_syms = vcat(tunable_syms, [Symbol(state_vars.vars[i].symbol) for i in 1:length(state_vars.vars)])

    # Collect all sampled tunable params from logger
    has_params = [e.params !== nothing for e in optlogger]
    if !any(has_params)
        vinfo("Optlogger does not contain parameter samples; skipping param exploration plot."; level=2)
        return nothing
    end

    samples_matrix = reduce(hcat, [e.params for e in optlogger if e.params !== nothing])'
    # rows: samples, cols: parameters
    n_params = size(samples_matrix, 2)
    
    # Verify dimensions match
    expected_n_params = length(tunables_lb)
    if n_params != expected_n_params
        vwarn("Parameter dimension mismatch: optlogger has $n_params params, but network has $expected_n_params tunable params + inits. Plot may be incorrect."; level=2)
        # Use minimum to avoid index errors
        n_params = min(n_params, expected_n_params)
    end

    # one subplot per parameter
    p = plot(layout=(n_params, 1), size=(900, max(250, 200*n_params)),
             legend=:topright, xminorgrid=true, yminorgrid=true)

    unique_restarts = sort(unique(e.irestart for e in optlogger))

    # choose distinct base hues for each restart (avoid black/white)
    base_hues = range(0, stop=360, length=length(unique_restarts)+1)[1:end-1]
    restart_colors = Dict(
        r => HSV(h, 0.8, 0.9) for (r, h) in zip(unique_restarts, base_hues)
    )


    for (j, sym) in enumerate(tunable_syms)
        sp = p[j]

        # scatter (rug-like) of sampled values, colored by restart and iteration
        for r in unique_restarts
            idx = findall(e -> e.irestart == r && e.params !== nothing, optlogger)
            if !isempty(idx)
                xr = [optlogger[i].params[j] for i in idx]
                iter_vals = [optlogger[i].iter for i in idx]

                # normalize iterations to [0, 1] within this restart
                it_min = minimum(iter_vals)
                it_max = maximum(iter_vals)
                it_range = max(it_max - it_min, 1)
                it_norm = [(it - it_min) / it_range for it in iter_vals]

                # alpha: early = 0.2 (very light), late = 1.0 (fully opaque)
                alphas = 0.2 .+ 0.6 .* it_norm
                base_c = restart_colors[r]

                # jitter y so points don't overlap
                yr = 0.0 .+ 0.02 .* (rand(length(xr)) .- 0.5)

                # Use mean alpha and color from restart
                scatter!(sp, xr, yr;
                         markersize=5,
                         markerstrokewidth=0,
                         alpha=mean(alphas),
                         color=base_c,
                         label = j == 1 ? "restart $r" : "")
            end
        end

        # draw min/max (allowed range)
        vline!(sp, [tunables_lb[j]]; color=:black, linestyle=:dash, linewidth=1.5, label = j == 1 ? "min/max" : "")
        vline!(sp, [tunables_ub[j]]; color=:black, linestyle=:dash, linewidth=1.5, label="")

        # draw default value
        vline!(sp, [tunables_default_vals[j]]; color=:red, linestyle=:solid, linewidth=2, label = j == 1 ? "true" : "")

        # cosmetics
        ylims!(sp, (-0.15, 0.15))
        xlabel!(sp, "value")
        ylabel!(sp, "")
        title!(sp, string(sym))
    end

    plot!(p, plot_title = "Parameter exploration across optimization")
    try
        savefig(p, fullfname_fig)
    catch e
        vwarn("Failed to save parameter exploration plot to $fullfname_fig: $e"; level=2)
    end

    return nothing
end


"""
    plot_psd_spetra_evolution(optlogger, net, setter; kwargs...)

Simulate and plot the power spectrum associated with each logged parameter
vector (typically from the best restart) so that spectrum evolution across the
optimization can be inspected or animated. Respects the optional σ_meas
measurement-noise parameter by injecting noise prior to PSD computation.
"""
function plot_psd_spetra_evolution(optlogger::Vector{OptLogEntry},
                              net::Network,
                              setter::Function;
                              data::Union{Data, Vector{Float64}, Matrix{Float64}, DataFrame}=Float64[],
                              settings::Union{Nothing, Settings}=nothing,
                              fullfname_fig::Union{Nothing, String}=nothing,
                              fullfname_anim::Union{Nothing, String}=nothing,
                              max_entries::Int=typemax(Int))::Nothing

    if settings !== nothing && !settings.general_settings.make_plots
        return nothing
    end

    entries = [entry for entry in optlogger if entry.params !== nothing]
    if isempty(entries)
        vinfo("Optlogger does not contain parameter samples; skipping param exploration plot."; level=2)
        return nothing
    end

    gs = settings.general_settings
    ss = settings.simulation_settings
    ls = settings.optimization_settings.loss_settings
    ds = settings.data_settings

    solver = get_solver(net.problem, ss)
    solver_kwargs = get_solver_kwargs(net.problem, ss)
    n_state_vars = length(net.problem.u0)
    n_params = length(entries[1].params) - n_state_vars

    spectra = NamedTuple{(:iter, :loss, :restart, :freqs, :powers, :sigma)}[]
    for entry in Iterators.take(entries, min(max_entries, length(entries)))
        params_vec = entry.params
        if length(params_vec) != n_params + n_state_vars
            continue
        end

        param_block = params_vec[1:n_params]
        init_block = params_vec[n_params + 1 : n_params + n_state_vars]
        sigma_val = ds.measurement_noise_std
        new_params = setter(net.problem.p, param_block)
        try
            sol = simulate_network(net.problem; new_params=new_params, new_inits=init_block,
                                   solver=solver, solver_kwargs=solver_kwargs)
            df = DataFrame(sol)
            if size(df, 2) < 2
                continue
            end
            model_prediction = Matrix(df)[:, 2]
            sampling_rate = length(sol.t) > 1 ? 1.0 / (sol.t[2] - sol.t[1]) : 1.0 / ss.saveat
            freqs, modeled_powers = compute_noisy_preprocessed_welch_psd(model_prediction, sampling_rate, ls, ds)
            # Skip entries with empty or invalid power spectra
            if !isempty(modeled_powers) && all(isfinite, modeled_powers)
                vinfo("Restart $(entry.irestart), Iter $(entry.iter): signal_len=$(length(model_prediction)), freq_points=$(length(freqs))"; level=2)
                push!(spectra, (iter=entry.iter, loss=entry.loss, restart=entry.irestart,
                                freqs=freqs, powers=modeled_powers, sigma=sigma_val))
            end
        catch err
            vwarn("Unable to compute spectrum for iter $(entry.iter): $(err)"; level=2)
        end
    end

    if isempty(spectra)
        vinfo("No spectra computed from optlogger; skipping spectrum sweep plot."; level=2)
        return nothing
    end

    # Filter out any spectra with empty powers (safety check)
    spectra = filter(spec -> !isempty(spec.powers) && all(isfinite, spec.powers), spectra)
    if isempty(spectra)
        vinfo("All computed spectra had empty or invalid power values; skipping spectrum sweep plot."; level=2)
        return nothing
    end

    # Standardize frequency grid across all spectra to finest resolution
    # Find spectrum with most frequency points (finer resolution)
    finest_freq_idx = argmax(length(spec.freqs) for spec in spectra)
    finest_freqs = spectra[finest_freq_idx].freqs
    
    # Interpolate all spectra to common frequency grid
    spectra_standardized = []
    for spec in spectra
        if length(spec.freqs) == length(finest_freqs) && isapprox(spec.freqs, finest_freqs)
            # Already on standard grid
            push!(spectra_standardized, spec)
        else
            # Interpolate to standard grid
            try
                interp_func = linear_interpolation(spec.freqs, spec.powers; extrapolation_bc=Flat())
                standardized_powers = interp_func.(finest_freqs)
                push!(spectra_standardized, (iter=spec.iter, loss=spec.loss, restart=spec.restart,
                                            freqs=finest_freqs, powers=standardized_powers, sigma=spec.sigma))
            catch
                # If interpolation fails, keep original
                push!(spectra_standardized, spec)
            end
        end
    end
    spectra = spectra_standardized

    target_freqs, target_curve = data isa Data ? (data.freqs, data.powers) : (nothing, nothing)
    
    # Interpolate target to model's frequency resolution (lower resolution) for plotting efficiency
    plot_freqs = nothing
    plot_target_curve = nothing
    if target_freqs !== nothing && !isempty(target_curve) && length(target_freqs) != length(spectra[1].freqs)
        # Downsample target to match model spectrum frequency grid via linear interpolation
        if !isempty(spectra[1].freqs) && all(isfinite, target_curve)
            interp_func = linear_interpolation(target_freqs, target_curve; extrapolation_bc=Flat())
            plot_target_curve = interp_func.(spectra[1].freqs)
            plot_freqs = spectra[1].freqs
        else
            plot_freqs = target_freqs
            plot_target_curve = target_curve
        end
    else
        plot_freqs = target_freqs
        plot_target_curve = target_curve
    end
    
    matches_target = plot_target_curve !== nothing && !isempty(plot_target_curve) && length(plot_target_curve) == length(spectra[1].freqs)
    overlay_freqs = plot_freqs === nothing ? spectra[1].freqs : plot_freqs

    power_min = minimum(minimum(spec.powers) for spec in spectra)
    power_max = maximum(maximum(spec.powers) for spec in spectra)
    if matches_target
        power_min = min(power_min, minimum(target_curve))
        power_max = max(power_max, maximum(target_curve))
    end
    span = max(power_max - power_min, eps(Float64))
    lower_pad = 0.05 * span
    upper_pad = 0.1 * span
    ylims_tuple = (power_min - lower_pad, power_max + upper_pad)

    fig_path = isnothing(fullfname_fig) ? "$(gs.path_out)\\$(gs.exp_name)_$(net.name)_freq_sweep.png" : fullfname_fig
    anim_path = isnothing(fullfname_anim) ? "$(gs.path_out)\\$(gs.exp_name)_$(net.name)_freq_sweep.gif" : fullfname_anim

    p = plot(title="Frequency spectra evolution", xlabel="Frequency (Hz)", ylabel="Norm Log10 Power")
    for (idx, spec) in enumerate(spectra)
        label = idx == length(spectra) ? "Iter $(spec.iter)" : ""
        plot!(p, spec.freqs, spec.powers; alpha=idx == length(spectra) ? 0.9 : 0.25, label=label)
    end
    # Plot target data last so it appears on top (black solid line, thicker)
    if matches_target
        plot!(p, overlay_freqs, plot_target_curve; label="Target", color=:black, linewidth=4, linestyle=:solid)
    end
    ylims!(p, ylims_tuple)
    try
        savefig(p, fig_path)
    catch e
        vwarn("Failed to save spectrum evolution plot to $fig_path: $e"; level=2)
    end

    if length(spectra) > 1
        anim = @animate for spec in spectra
            plt = plot(spec.freqs, spec.powers;
                       label="Iter $(spec.iter)",
                       color=:dodgerblue,
                       linewidth=2,
                       xlabel="Frequency (Hz)",
                       ylabel="Norm Log10 Power",
                       title="Restart $(spec.restart) • Iter $(spec.iter) • Loss $(round(spec.loss, digits=4)) • σ=$(round(spec.sigma, digits=4))",
                       ylims=ylims_tuple)
            # Plot target data last so it appears on top (black solid line, thicker)
            if matches_target
                plot!(plt, overlay_freqs, plot_target_curve; label="Target", color=:black, linewidth=4, linestyle=:solid)
            end
        end
        try
            gif(anim, anim_path, fps=min(12, max(2, length(spectra) ÷ 10)), verbose=false)
        catch err
            vwarn("Unable to save spectrum animation: $(err)"; level=2)
        end
    end

    return nothing
end


function compute_r2_for_params(net::Network,
                               data::Data,
                               param_symbols::Vector{Symbol},
                               params_native::Vector{Float64},
                               inits_native::Vector{Float64})::Float64
    ss = net.settings.simulation_settings
    ls = net.settings.optimization_settings.loss_settings
    ds = net.settings.data_settings
    solver = get_solver(net.problem, ss)
    solver_kwargs = get_solver_kwargs(net.problem, ss)

    setter = make_namedtuple_setter(Tuple(param_symbols))
    opt_params = setter(net.problem.p, params_native)
    sol = simulate_network(net.problem;
                                  new_params=opt_params,
                                  new_inits=inits_native,
                                  solver=solver,
                                  solver_kwargs=solver_kwargs)
    sol.retcode == :Success || return NaN

    df = DataFrame(sol)
    if size(df, 2) < 2
        return NaN
    end
    model_prediction = Matrix(df)[:, 2]
    fs = length(sol.t) > 1 ? 1.0 / (sol.t[2] - sol.t[1]) : 1.0 / ss.saveat
    _, modeled_powers = compute_noisy_preprocessed_welch_psd(model_prediction, fs, ls, ds)
    target_freqs, target_curve = data.freqs, data.powers
    if target_curve === nothing || length(target_curve) != length(modeled_powers)
        return NaN
    end
    return r2(modeled_powers, target_curve)
end

function native_range_entries(net::Network, blocks)
    param_ranges = OrderedDict{String, Any}[]
    for (idx, sym) in enumerate(blocks.tunable_params_symbols)
        param_type = String(get_param_by_symbol(net.params, sym).type)
        push!(param_ranges, OrderedDict(
            "name" => String(sym),
            "native_bounds" => [blocks.tunable_params_lb[idx], blocks.tunable_params_ub[idx]],
            "param_type" => param_type
        ))
    end

    init_names = if net.problem.u0 isa NamedTuple
        String.(keys(net.problem.u0))
    else
        # Get actual state variable names from the network
        state_vars = get_state_vars(net.vars)
        string.(get_symbols(state_vars))
    end
    init_ranges = OrderedDict{String, Any}[]
    for i in eachindex(blocks.init_lb)
        push!(init_ranges, OrderedDict(
            "name" => init_names[i],
            "native_bounds" => [blocks.init_lb[i], blocks.init_ub[i]],
        ))
    end
    return param_ranges, init_ranges
end

function plot_loss_over_iterations(optlogger::Vector{OptLogEntry},
                                   general_settings::GeneralSettings,
                                   fullfname_fig::AbstractString)
    general_settings.make_plots || return nothing
    try
        plot([optlogger[i].iter for i in eachindex(optlogger)],
             [optlogger[i].loss for i in eachindex(optlogger)],
             xlabel="Iteration", ylabel="Loss")
        savefig(fullfname_fig)
    catch e
        vwarn("Failed to plot loss evolution to $fullfname_fig: $e"; level=2)
    end
    return nothing
end

function plot_observed_vs_modelled_ts_windows(sol::SciMLBase.AbstractODESolution,
                                              data::Union{Data, Vector{Float64}, Matrix{Float64}, DataFrame},
                                              settings::Union{Nothing, Settings},
                                              net::Network;
                                              zoom_window::Tuple{Float64, Float64}=(2.0, 5.0),
                                              fullfname_fig::Union{Nothing, AbstractString}=nothing)
    gen_settings = settings === nothing ? net.settings.general_settings : settings.general_settings
    gen_settings.make_plots || return nothing
    data isa Data || return nothing

    df = DataFrame(sol)
    size(df, 2) >= 2 || return nothing

    model_times = sol.t
    model_ts = Vector(df[:, 2])
    obs_times = data.times
    obs_ts = data.signal

    p = plot(layout=(2, 1), size=(900, 500), legend=:topright)
    plot!(p[1], obs_times, obs_ts, label="Observed", xlabel="", ylabel="")
    plot!(p[1], model_times, model_ts, label="Model", xlabel="", ylabel="")
    plot!(p[2], obs_times, obs_ts, label="Observed", xlabel="Time (s)", ylabel="", xlims=zoom_window)
    plot!(p[2], model_times, model_ts, label="Model", xlabel="Time (s)", ylabel="", xlims=zoom_window)

    outpath = fullfname_fig === nothing ?
              joinpath(gen_settings.path_out,
                       "$(net.name)_$(gen_settings.exp_name)_plot_obs_vs_mod_ts.png") :
              String(fullfname_fig)
    try
        savefig(p, outpath)
    catch e
        vwarn("Failed to save observed vs modelled timeseries plot to $outpath: $e"; level=2)
    end
    return nothing
end


