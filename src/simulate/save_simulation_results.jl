# ============================================================================
# Save Simulation Results
# ============================================================================
# Saves simulation output (timeseries DataFrame, plots, and metadata) to a 
# numbered simulation_N/ folder with organized figure/ subfolder.
# Mirrors the structure and approach used in save_optimization_results.jl.

using CSV, DataFrames, Dates, OrderedCollections

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

    general_settings = settings.general_settings
    simulation_settings = settings.simulation_settings

    # Create numbered simulation_N/ folder if not provided
    if simulation_output_dir === nothing
        candidate_path = ENEEGMA.construct_output_dir(general_settings, settings.network_settings)
        simulation_output_dir = ENEEGMA.find_next_numbered_folder(candidate_path, "simulation")
        mkpath(simulation_output_dir)
    end

    # Extract signal and compute PSD
    signal_name = ENEEGMA.get_eeg_signal(settings, df)
    signal = df[!, Symbol(signal_name)]
    
    # Calculate sampling frequency
    fs = 1.0 / simulation_settings.saveat
    
    # Compute PSD using centralized function
    freqs, powers = ENEEGMA.compute_preprocessed_welch_psd(signal, fs)

    # Build output file names using standardized format: exp_name_net_name_simulated
    base_prefix = "$(general_settings.exp_name)_$(net.name)_simulated"

    # Save composite plot (3 panels: full ts, zoomed ts, PSD) to figures folder
    fname_plot = "$(base_prefix).png"
    path_plot = joinpath(simulation_output_dir, fname_plot)
    ENEEGMA.plot_simulation_results(df.time, signal, freqs, powers;
                                     zoom_window=(2.0, 5.0),
                                     signal_name=signal_name,
                                     fullfname_fig=path_plot,
                                     general_settings=general_settings)

    # Save timeseries DataFrame to CSV in simulation folder root
    fname_csv = "$(base_prefix).csv"
    path_csv = joinpath(simulation_output_dir, fname_csv)
    CSV.write(path_csv, df)
    vinfo("Saved simulation timeseries to CSV: $path_csv"; level=2)

    # Build metadata section
    timestamp_str = Dates.format(Dates.now(), "yyyy-mm-ddTHH:MM:SSZ")
    settings_path = joinpath(general_settings.path_out, general_settings.exp_name, "settings.json")
    settings_path_normalized = replace(settings_path, "\\" => "/")

    metadata = OrderedDict(
        "settings_file" => settings_path_normalized,
        "timestamp" => timestamp_str,
        "eeg_output" => signal_name,
        "time_points" => length(df.time),
        "duration_seconds" => df.time[end] - df.time[1]
    )

    # Extract current parameter values from network
    current_params_dict = OrderedDict{String, Any}()
    for param in net.params.params
        param_name = string(param.symbol)
        normalized_name = ENEEGMA.normalize_parameter_name(param_name)
        current_params_dict[normalized_name] = param.default
    end

    # Extract initial condition values from network problem
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

    # Build main results structure with simulation_results first, then metadata and other sections
    results = OrderedDict{String, Any}(
        "metadata" => metadata,
        "parameters" => current_params_dict,
        "initial_states" => current_inits_dict
    )

    # Add full simulation settings if available
    if hasfield(typeof(simulation_settings), :include_settings_in_results_output) && 
       simulation_settings.include_settings_in_results_output
        results["simulation_settings"] = ENEEGMA.settings_to_dict(settings)
    end

    # Save results JSON
    fname_results = "$(base_prefix)_info.json"
    results_path = joinpath(simulation_output_dir, fname_results)
    ENEEGMA.write_compact_json(results_path, results)

    vinfo("Simulation results saved to: $simulation_output_dir"; level=1)
    return simulation_output_dir
end

