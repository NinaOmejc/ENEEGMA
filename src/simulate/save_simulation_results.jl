# ============================================================================
# Save Simulation Results
# ============================================================================
# Saves simulation output (timeseries DataFrame, plots, and metadata) to a 
# numbered simulation_N/ folder with organized figure/ subfolder.
# Mirrors the structure and approach used in save_optimization_results.jl.

using CSV, DataFrames, Dates, OrderedCollections

"""
    save_simulation_results(sol::SciMLBase.AbstractODESolution,
                            net::Network,
                            settings::Settings;
                            simulation_output_dir::Union{String, Nothing}=nothing)::String

Save simulation results from a solved simulation using the same post-processing
pipeline as optimization and evaluation.

"""
function save_simulation_results(sol::SciMLBase.AbstractODESolution,
                                 net::Network,
                                 settings::Settings;
                                 simulation_output_dir::Union{String, Nothing}=nothing)::String
    general_settings = settings.general_settings
    simulation_settings = settings.simulation_settings

    if simulation_output_dir === nothing
        candidate_path = ENEEGMA.construct_output_dir(general_settings, settings.network_settings)
        simulation_output_dir = ENEEGMA.find_next_numbered_folder(candidate_path, "simulation")
        mkpath(simulation_output_dir)
    end

    df = sol2df(sol, net)
    _, expr_map = ENEEGMA.extract_brain_sources(settings, net, df; return_source_expressions=true)
    success, error_msg, times, model_predictions = ENEEGMA.extract_validated_model_predictions(sol, net, settings; demean=true)

    if success
        node_names = [String(node.name) for node in net.nodes if haskey(model_predictions, String(node.name))]
        df_sources = ENEEGMA.model_predictions_to_dataframe(times, model_predictions; node_names=node_names)
        psd_dict = ENEEGMA.compute_psd_for_all_sources(df_sources, 1.0 / simulation_settings.saveat;
            data_settings=settings.data_settings,
            loss_settings=settings.optimization_settings.loss_settings)
    else
        vwarn("Could not prepare canonical simulation outputs: $(error_msg)"; level=2)
        df_sources = DataFrame()
        psd_dict = Dict{String, Tuple{Vector{Float64}, Vector{Float64}}}()
    end

    return ENEEGMA._write_simulation_results(
        df_sources,
        expr_map,
        length(sol.t),
        sol.t[end] - sol.t[1],
        net,
        settings;
        simulation_output_dir=simulation_output_dir,
        psd_dict=psd_dict
    )
end

"""
    save_simulation_results(df::DataFrame, 
                            net::Network, 
                            settings::Settings;
                            simulation_output_dir::Union{String, Nothing}=nothing)::String

Save simulation results including timeseries plot and CSV data file.

Creates a numbered simulation_N/ folder in the experiment directory with:
- figures/ subfolder containing the composite plot
- CSV file with simulation timeseries data
- Optional settings.json reference

The composite plot includes 3 subplots:
  1. Full timeseries view
  2. Zoomed 2-second window
  3. Power spectral density (PSD)

# Arguments
- `df::DataFrame`: Simulation output DataFrame with time column
- `net::Network`: Network object for extracting name and parameters
- `settings::Settings`: Settings object containing general and simulation config
- `simulation_output_dir::Union{String, Nothing}`: Custom output directory (optional)

# Returns
String: Path to the created simulation output folder

# Example
```julia
df = simulate_network(net)
simulation_path = save_simulation_results(df, net, settings)
# Creates: ./results/my_exp/simulation_1/
#   └─ figures/
#     └─ my_exp_net_simulated.png (3-panel composite plot)
#   └─ my_exp_net_simulated.csv
```
"""
function save_simulation_results(df::DataFrame,
                                 net::Network,
                                 settings::Settings;
                                 simulation_output_dir::Union{String, Nothing}=nothing)::String
    simulation_settings = settings.simulation_settings
    data_settings = settings.data_settings

    df_sources_raw, expr_map = ENEEGMA.extract_brain_sources(settings, net, df; return_source_expressions=true)
    source_names = [String(col) for col in names(df_sources_raw) if String(col) != "time"]
    df_sources, _ = ENEEGMA.prepare_source_dataframe_for_analysis(
        df_sources_raw,
        1.0 / simulation_settings.saveat,
        data_settings.psd.transient_period_duration;
        demean=true,
        source_names=source_names
    )

    psd_dict = ENEEGMA.compute_psd_for_all_sources(df_sources, 1.0 / simulation_settings.saveat;
        data_settings=settings.data_settings,
        loss_settings=settings.optimization_settings.loss_settings)

    return ENEEGMA._write_simulation_results(
        df_sources,
        expr_map,
        length(df.time),
        df.time[end] - df.time[1],
        net,
        settings;
        simulation_output_dir=simulation_output_dir,
        psd_dict=psd_dict
    )
end

function _write_simulation_results(df_sources::DataFrame,
                                   expr_map::AbstractDict,
                                   time_points::Integer,
                                   duration_seconds::Real,
                                   net::Network,
                                   settings::Settings;
                                   simulation_output_dir::Union{String, Nothing}=nothing,
                                   psd_dict::Union{Nothing, Dict{String, Tuple{Vector{Float64}, Vector{Float64}}}}=nothing)::String
    general_settings = settings.general_settings
    simulation_settings = settings.simulation_settings

    if simulation_output_dir === nothing
        candidate_path = ENEEGMA.construct_output_dir(general_settings, settings.network_settings)
        simulation_output_dir = ENEEGMA.find_next_numbered_folder(candidate_path, "simulation")
        mkpath(simulation_output_dir)
    end

    base_prefix = "$(general_settings.exp_name)_$(net.name)_simulated"

    fname_plot = "$(base_prefix).png"
    path_plot = joinpath(simulation_output_dir, fname_plot)
    ENEEGMA.plot_simulation_results(df_sources;
                                     psd_dict=psd_dict,
                                     zoom_window=(2.0, 5.0),
                                     fullfname_fig=path_plot,
                                     data_settings=settings.data_settings,
                                     general_settings=general_settings)

    fname_csv = "$(base_prefix).csv"
    path_csv = joinpath(simulation_output_dir, fname_csv)
    CSV.write(path_csv, df_sources)
    vinfo("Saved brain source signals to CSV: $path_csv"; level=2)

    timestamp_str = Dates.format(Dates.now(), "yyyy-mm-ddTHH:MM:SSZ")
    settings_path = joinpath(general_settings.path_out, general_settings.exp_name, "settings.json")
    settings_path_normalized = replace(settings_path, "\\" => "/")

    metadata = OrderedDict(
        "settings_file" => settings_path_normalized,
        "timestamp" => timestamp_str,
        "eeg_output" => expr_map,
        "time_points" => time_points,
        "duration_seconds" => duration_seconds
    )

    param_defaults = ENEEGMA.get_param_default_values(net.params; return_type="named_tuple", sort=true)
    current_params_dict = OrderedDict{String, Any}()
    for (param_name, param_default) in pairs(param_defaults)
        normalized_name = ENEEGMA.normalize_parameter_name(String(param_name))
        current_params_dict[normalized_name] = param_default
    end

    init_names = if net.problem.u0 isa NamedTuple
        String.(keys(net.problem.u0))
    else
        state_vars = ENEEGMA.get_state_vars(net.vars)
        string.(ENEEGMA.get_symbols(state_vars))
    end

    current_inits_dict = OrderedDict{String, Any}()
    init_values = net.problem.u0 isa NamedTuple ? collect(values(net.problem.u0)) : net.problem.u0
    for (init_name, init_val) in zip(init_names, init_values)
        normalized_name = ENEEGMA.normalize_parameter_name(init_name)
        current_inits_dict[normalized_name] = init_val
    end

    results = OrderedDict{String, Any}(
        "metadata" => metadata,
        "parameters" => current_params_dict,
        "initial_states" => current_inits_dict
    )

    if hasfield(typeof(simulation_settings), :include_settings_in_results_output) &&
       simulation_settings.include_settings_in_results_output
        results["simulation_settings"] = ENEEGMA.settings_to_dict(settings)
    end

    fname_results = "$(base_prefix)_info.json"
    results_path = joinpath(simulation_output_dir, fname_results)
    ENEEGMA.write_compact_json(results_path, results)

    vinfo("Simulation results saved to: $simulation_output_dir"; level=1)
    return simulation_output_dir
end

