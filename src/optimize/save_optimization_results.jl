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
        evaluation_metrics=["weighted_mae", "r2", "weighted_iae"],
        demean=true,
        transient_period_duration=settings.data_settings.psd.transient_period_duration
    );
    
    if !result.success
        vwarn("Failed to simulate model with best parameters: $(result.error_msg)"; level=2)
        vwarn("Attempting to save results with fallback values..."; level=2)
    end

    observed_signals = Dict(node_name => node_info.signal for (node_name, node_info) in data.node_data)

    # Extract data from first node for single-panel outputs that still expect one PSD curve
    first_node_key = first(keys(data.node_data))

    times = result.success ? result.df_sources.time : data.times
    freqs, modeled_powers = get(result.psd_dict, first_node_key, (Float64[], Float64[]))
    
    ENEEGMA.save_modeled_psd(modeled_powers, freqs, settings; 
                            fullfname_csv=joinpath(optimization_path, "$(base_prefix)_modeled_psd.csv"))

    fname_obs_ts = "$(base_prefix)_observed_vs_modeled_timeseries.png"
    path_obs_ts = joinpath(figures_dir, fname_obs_ts)
    # Use unified visualization function from visualization.jl
    ENEEGMA.plot_timeseries_windows(times, result.model_predictions;
                                     observed_signals=observed_signals,
                                     zoom_window=(2.0, 5.0),
                                     title_prefix="Observed vs Modeled Timeseries",
                                     fullfname_fig=path_obs_ts,
                                     general_settings=general_settings);

    fname_obs_freq = "$(base_prefix)_observed_vs_modeled_spectrum.png"
    path_obs_freq = joinpath(figures_dir, fname_obs_freq)
    # Use unified visualization function from visualization.jl
    ENEEGMA.plot_psd_comparison(result.psd_dict, data;
                                title="Power Spectral Density Comparison",
                                data_settings=settings.data_settings,
                                fullfname_fig=path_obs_freq,
                                general_settings=general_settings)

    # PLOT: Parameter exploration
    fname_param = "$(base_prefix)_parameter_exploration.png"
    path_param = joinpath(figures_dir, fname_param)
    ENEEGMA.plot_param_exploration(optlogger, net; 
                                true_parameters=true_parameters,
                                node_names=result.node_names,
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

    # Average node-wise metrics from evaluate_model for summary outputs
    loss_values = result.success ? get(result.metrics, "weighted_mae", Float64[]) : Float64[]
    r2_values = result.success ? get(result.metrics, "r2", Float64[]) : Float64[]
    iae_values = result.success ? get(result.metrics, "weighted_iae", Float64[]) : Float64[]

    recomputed_loss = !isempty(loss_values) ? Statistics.mean(loss_values) : best_loss
    r2_metric = !isempty(r2_values) ? Statistics.mean(r2_values) : NaN
    iae_metric = !isempty(iae_values) ? Statistics.mean(iae_values) : NaN
    
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

    per_node_metrics_dict = OrderedDict(
        "node_names" => result.node_names,
        "weighted_mae" => loss_values,
        "r2" => r2_values,
        "weighted_iae" => iae_values
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
        "per_node_metrics" => per_node_metrics_dict,
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
    first_node_key = first(keys(data.node_data))
    first_node_data = data.node_data[first_node_key]
    return plot_observed_vs_modelled_psd(data_modeled, first_node_data.powers, first_node_data.freqs;
                                          fullfname_fig=fullfname_fig,
                                          general_settings=general_settings)
end

function _group_param_indices_by_node(tunable_syms::Vector{Symbol}, node_names::Vector{String})::Vector{Vector{Int}}
    if isempty(node_names)
        return [collect(eachindex(tunable_syms))]
    end

    shared_indices = Int[]
    grouped_indices = [Int[] for _ in node_names]

    for (idx, sym) in enumerate(tunable_syms)
        sym_lower = lowercase(String(sym))
        matched = false
        for (node_idx, node_name) in enumerate(node_names)
            if occursin(lowercase(node_name), sym_lower)
                push!(grouped_indices[node_idx], idx)
                matched = true
            end
        end
        matched || push!(shared_indices, idx)
    end

    return [unique(vcat(shared_indices, grouped_indices[node_idx])) for node_idx in eachindex(node_names)]
end

function _plot_param_exploration_panel!(sp,
                                        param_idx::Int,
                                        sym::Symbol,
                                        optlogger::Vector{OptLogEntry},
                                        unique_restarts,
                                        restart_colors,
                                        tunables_lb::Vector{Float64},
                                        tunables_ub::Vector{Float64},
                                        tunables_default_vals::Vector{Float64};
                                        show_legend::Bool=false,
                                        panel_title::String=string(sym))
    for r in unique_restarts
        idx = findall(e -> e.irestart == r && e.params !== nothing, optlogger)
        if !isempty(idx)
            xr = [optlogger[i].params[param_idx] for i in idx]
            iter_vals = [optlogger[i].iter for i in idx]

            it_min = minimum(iter_vals)
            it_max = maximum(iter_vals)
            it_range = max(it_max - it_min, 1)
            it_norm = [(it - it_min) / it_range for it in iter_vals]
            alphas = 0.2 .+ 0.6 .* it_norm
            yr = 0.0 .+ 0.02 .* (rand(length(xr)) .- 0.5)

            scatter!(sp, xr, yr;
                     markersize=5,
                     markerstrokewidth=0,
                     alpha=mean(alphas),
                     color=restart_colors[r],
                     label=show_legend ? "restart $r" : "")
        end
    end

    vline!(sp, [tunables_lb[param_idx]]; color=:black, linestyle=:dash, linewidth=1.5, label=show_legend ? "min/max" : "")
    vline!(sp, [tunables_ub[param_idx]]; color=:black, linestyle=:dash, linewidth=1.5, label="")
    vline!(sp, [tunables_default_vals[param_idx]]; color=:red, linestyle=:solid, linewidth=2, label=show_legend ? "true" : "")

    ylims!(sp, (-0.15, 0.15))
    xlabel!(sp, "value")
    ylabel!(sp, "")
    title!(sp, panel_title)
end

function _blank_subplot!(sp)
    plot!(sp;
          framestyle=:none,
          grid=false,
          xticks=nothing,
          yticks=nothing,
          label="")
    return nothing
end

function plot_param_exploration(optlogger::Vector{OptLogEntry},
                                net::Network;
                                true_parameters::Union{OrderedDict{Num, Float64}, Dict{String, Tuple{Vararg{Float64}}}, OrderedDict{String, Tuple{Vararg{Float64}}}, Nothing}=nothing,
                                node_names::Union{Nothing, Vector{String}}=nothing,
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

    unique_restarts = sort(unique(e.irestart for e in optlogger))

    # choose distinct base hues for each restart (avoid black/white)
    base_hues = range(0, stop=360, length=length(unique_restarts)+1)[1:end-1]
    restart_colors = Dict(
        r => HSV(h, 0.8, 0.9) for (r, h) in zip(unique_restarts, base_hues)
    )

    if node_names === nothing || length(node_names) <= 1
        p = plot(layout=(n_params, 1), size=(900, max(250, 200*n_params)),
                 legend=:topright, xminorgrid=true, yminorgrid=true)

        for (j, sym) in enumerate(tunable_syms)
            _plot_param_exploration_panel!(p[j], j, sym, optlogger, unique_restarts, restart_colors,
                                           tunables_lb, tunables_ub, tunables_default_vals;
                                           show_legend=j == 1,
                                           panel_title=string(sym))
        end
    else
        param_indices_by_node = _group_param_indices_by_node(tunable_syms[1:n_params], node_names)
        n_cols = length(node_names)
        n_rows = maximum(max(length(indices), 1) for indices in param_indices_by_node)
        p = plot(layout=(n_rows, n_cols),
                 size=(420 * n_cols, max(250, 180 * n_rows)),
                 legend=:topright,
                 xminorgrid=true,
                 yminorgrid=true)

        for (col_idx, node_name) in enumerate(node_names)
            param_indices = param_indices_by_node[col_idx]
            for row_idx in 1:n_rows
                sp = p[(row_idx - 1) * n_cols + col_idx]
                if row_idx <= length(param_indices)
                    param_idx = param_indices[row_idx]
                    sym = tunable_syms[param_idx]
                    panel_title = row_idx == 1 ? "$(node_name)\n$(sym)" : string(sym)
                    _plot_param_exploration_panel!(sp, param_idx, sym, optlogger, unique_restarts, restart_colors,
                                                   tunables_lb, tunables_ub, tunables_default_vals;
                                                   show_legend=col_idx == 1 && row_idx == 1,
                                                   panel_title=panel_title)
                else
                    _blank_subplot!(sp)
                end
            end
        end
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

    spectra_frames = NamedTuple{(:iter, :loss, :restart, :sigma, :psd_dict)}[]
    node_names = data isa Data ? collect(keys(data.node_data)) : String[]

    for entry in Iterators.take(entries, min(max_entries, length(entries)))
        params_vec = entry.params
        length(params_vec) == n_params + n_state_vars || continue

        param_block = params_vec[1:n_params]
        init_block = params_vec[n_params + 1 : n_params + n_state_vars]
        sigma_val = ds.measurement_noise_std
        new_params = setter(net.problem.p, param_block)

        try
            sol = simulate_network(net.problem; new_params=new_params, new_inits=init_block,
                                   solver=solver, solver_kwargs=solver_kwargs)
            df = sol2df(sol, net)
            nrow(df) == 0 && continue

            df_sources, _ = extract_brain_sources(settings, net, df; return_source_expressions=true)
            nrow(df_sources) == 0 && continue

            fs = 1.0 / ss.saveat
            transient_duration = ds.psd.transient_period_duration
            keep_idx = get_indices_after_transient_removal(df_sources.time, transient_duration, df_sources.time[1], fs)
            if !isempty(keep_idx)
                df_sources = df_sources[keep_idx, :]
            end

            for col in names(df_sources)
                col == :time && continue
                df_sources[!, col] .-= mean(df_sources[!, col])
            end

            psd_fs = ds.fs === nothing ? fs : ds.fs
            frame_psd_dict = compute_psd_for_all_sources(df_sources, psd_fs;
                                                         data_settings=ds,
                                                         loss_settings=ls)
            isempty(frame_psd_dict) && continue

            if isempty(node_names)
                node_names = collect(keys(frame_psd_dict))
            end

            valid_psd_dict = Dict{String, Tuple{Vector{Float64}, Vector{Float64}}}()
            for node_name in node_names
                haskey(frame_psd_dict, node_name) || continue
                freqs, powers = frame_psd_dict[node_name]
                if !isempty(powers) && all(isfinite, powers)
                    valid_psd_dict[node_name] = (freqs, powers)
                end
            end

            isempty(valid_psd_dict) && continue
            push!(spectra_frames, (iter=entry.iter,
                                   loss=entry.loss,
                                   restart=entry.irestart,
                                   sigma=sigma_val,
                                   psd_dict=valid_psd_dict))
        catch err
            vwarn("Unable to compute spectrum for iter $(entry.iter): $(err)"; level=2)
        end
    end

    isempty(spectra_frames) && return nothing

    node_names = [node_name for node_name in node_names if any(haskey(frame.psd_dict, node_name) for frame in spectra_frames)]
    isempty(node_names) && return nothing

    ylims_by_node = Dict{String, Tuple{Float64, Float64}}()
    for node_name in node_names
        all_powers = Float64[]
        for frame in spectra_frames
            haskey(frame.psd_dict, node_name) || continue
            append!(all_powers, frame.psd_dict[node_name][2])
        end
        if data isa Data && haskey(data.node_data, node_name)
            append!(all_powers, data.node_data[node_name].powers)
        end

        if isempty(all_powers)
            ylims_by_node[node_name] = (0.0, 1.0)
        else
            power_min = minimum(all_powers)
            power_max = maximum(all_powers)
            span = max(power_max - power_min, eps(Float64))
            ylims_by_node[node_name] = (power_min - 0.05 * span, power_max + 0.1 * span)
        end
    end

    fig_path = isnothing(fullfname_fig) ? "$(gs.path_out)\\$(gs.exp_name)_$(net.name)_freq_sweep.png" : fullfname_fig
    anim_path = isnothing(fullfname_anim) ? "$(gs.path_out)\\$(gs.exp_name)_$(net.name)_freq_sweep.gif" : fullfname_anim

    n_nodes = length(node_names)
    p = plot(layout=(1, n_nodes), size=(450 * n_nodes, 400), legend=:topright)
    for (col_idx, node_name) in enumerate(node_names)
        sp = p[col_idx]
        for (frame_idx, frame) in enumerate(spectra_frames)
            haskey(frame.psd_dict, node_name) || continue
            freqs, powers = frame.psd_dict[node_name]
            label = frame_idx == length(spectra_frames) ? "Iter $(frame.iter)" : ""
            plot!(sp, freqs, powers;
                  alpha=frame_idx == length(spectra_frames) ? 0.9 : 0.25,
                  label=label,
                  linewidth=1.5)
        end
        if data isa Data && haskey(data.node_data, node_name)
            node_info = data.node_data[node_name]
            plot!(sp, node_info.freqs, node_info.powers;
                  label="Target",
                  color=:black,
                  linewidth=4,
                  linestyle=:solid)
        end
        xlabel!(sp, "Frequency (Hz)")
        ylabel!(sp, col_idx == 1 ? "Norm Log10 Power" : "")
        title!(sp, node_name)
        ylims!(sp, ylims_by_node[node_name])
    end
    plot!(p, plot_title="Frequency spectra evolution")
    try
        savefig(p, fig_path)
    catch e
        vwarn("Failed to save spectrum evolution plot to $fig_path: $e"; level=2)
    end

    if length(spectra_frames) > 1
        anim = @animate for frame in spectra_frames
            plt = plot(layout=(1, n_nodes), size=(450 * n_nodes, 400), legend=:topright)
            for (col_idx, node_name) in enumerate(node_names)
                sp = plt[col_idx]
                if haskey(frame.psd_dict, node_name)
                    freqs, powers = frame.psd_dict[node_name]
                    plot!(sp, freqs, powers;
                          label="Iter $(frame.iter)",
                          color=:dodgerblue,
                          linewidth=2)
                end
                if data isa Data && haskey(data.node_data, node_name)
                    node_info = data.node_data[node_name]
                    plot!(sp, node_info.freqs, node_info.powers;
                          label="Target",
                          color=:black,
                          linewidth=4,
                          linestyle=:solid)
                end
                xlabel!(sp, "Frequency (Hz)")
                ylabel!(sp, col_idx == 1 ? "Norm Log10 Power" : "")
                title!(sp, node_name)
                ylims!(sp, ylims_by_node[node_name])
            end
            plot!(plt, plot_title="Restart $(frame.restart) | Iter $(frame.iter) | Loss $(round(frame.loss, digits=4)) | sigma=$(round(frame.sigma, digits=4))")
        end
        try
            gif(anim, anim_path, fps=min(12, max(2, Int(floor(length(spectra_frames) / 10)))), verbose=false)
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
    first_node_key = first(keys(data.node_data))
    first_node_data = data.node_data[first_node_key]
    target_freqs, target_curve = first_node_data.freqs, first_node_data.powers
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
    first_node_key = first(keys(data.node_data))
    first_node_data = data.node_data[first_node_key]
    obs_ts = first_node_data.signal

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


