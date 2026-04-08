"""
    load_data(ds::DataSettings)

Load data from CSV file specified in DataSettings.

# Arguments
- `ds::DataSettings`: Data settings containing data_file path

# Returns
DataFrame with loaded data
"""
function load_data(ds::DataSettings)
    # Validate inputs
    isnothing(ds.data_file) && error("data_file not specified in DataSettings")
    !isfile(ds.data_file) && error("Data file not found: $(ds.data_file)")
    
    data = CSV.read(ds.data_file, DataFrame)
    return data
end

"""
    load_settings_from_file(filepath::String)::Dict{String, Any}

Load settings from a JSON file with validation.

# Arguments
- `filepath::String`: Path to the JSON settings file (must end with .json)

# Returns
Loaded settings dictionary

# Errors
- Errors if file doesn't exist
- Errors if file doesn't have .json extension
"""
function load_settings_from_file(filepath::String)::Dict{String, Any}
    !isfile(filepath) && error("Settings file not found: $filepath")
    !endswith(filepath, ".json") && error("Settings file must be .json format")
    
    settings = JSON.parsefile(filepath; dicttype=Dict{String, Any})
    vinfo("Settings loaded from: $filepath")
    return settings
end



"""
    create_default_settings()::Settings

Create a Settings object with all default values from the constructors.

All defaults are defined in the Settings type constructors in src/types/settings.jl.
This function simply instantiates Settings with an empty dictionary, allowing
all sub-settings to apply their built-in defaults.

Since all Settings types are mutable, you can modify fields after creation:

# Example
```julia
settings = create_default_settings()
settings.general_settings.exp_name = "MyExperiment"
settings.network_settings.n_nodes = 3
settings.general_settings.verbosity_level = 2
```

# Returns
A fully initialized `Settings` object with default values for:
- General settings: exp_name="SimpleNetwork", path_out="./results", etc.
- Network settings: n_nodes=1, node_names=["N1"], node_models=["WC"], etc.
- Simulation settings: time span, solver parameters
- Optimization settings: loss functions, optimizer params (CMAES only)
- Data and sampling settings with sensible defaults
"""
function create_default_settings()::Settings
    return Settings(Dict{String, Any}())
end

"""
    settings_to_dict(settings::Settings)::Dict{String, Any}

Convert a Settings object back to its dictionary representation.
Useful for saving and inspecting configuration.

# Arguments
- `settings::Settings`: Settings object to convert

# Returns
Dictionary representation of the Settings object
"""
function settings_to_dict(settings::Settings)::Dict{String, Any}
    d = Dict{String, Any}()
    
    # General settings
    d["general_settings"] = Dict(
        "exp_name" => settings.general_settings.exp_name,
        "path_out" => settings.general_settings.path_out,
        "verbosity_level" => settings.general_settings.verbosity_level,
        "seed" => settings.general_settings.seed,
        "make_plots" => settings.general_settings.make_plots,
        "save_model_formats" => settings.general_settings.save_model_formats
    )
    
    # Network settings
    ns = settings.network_settings
    # Serialize node_models: convert RuleTree to string representation
    node_models_serialized = [m isa String ? m : serialize_rule_tree(m) for m in ns.node_models]
    d["network_settings"] = Dict(
        "n_nodes" => ns.n_nodes,
        "node_names" => ns.node_names,
        "node_models" => node_models_serialized,
        "node_coords" => ns.node_coords,
        "network_conn" => ns.network_conn,
        "network_delay" => ns.network_delay,
        "network_conn_funcs" => ns.network_conn_funcs,
        "sensory_input_conn" => ns.sensory_input_conn,
        "sensory_input_func" => ns.sensory_input_func,
        "sensory_seed" => ns.sensory_seed,
        "eeg_output" => ns.eeg_output
    )
    
    # Simulation settings
    ss = settings.simulation_settings
    d["simulation_settings"] = Dict(
        "n_runs" => ss.n_runs,
        "tspan" => collect(ss.tspan),
        "dt" => ss.dt,
        "solver" => ss.solver
    )
    
    # Data settings
    data_s = settings.data_settings
    d["data_settings"] = if data_s !== nothing
        Dict(
            "data_file" => data_s.data_file,
            "target_channel" => data_s.target_channel,
            "task_type" => data_s.task_type,
            "fs" => data_s.fs,
            "data_columns" => data_s.data_columns
        )
    else
        Dict()
    end
    
    # Optimization settings with nested loss and optimizer
    os = settings.optimization_settings
    ls = os.loss_settings
    opt_s = os.optimizer_settings
    
    d["optimization_settings"] = Dict(
        "method" => os.method,
        "loss" => os.loss,
        "loss_abstol" => os.loss_abstol,
        "loss_reltol" => os.loss_reltol,
        "abs_target_loss" => os.abs_target_loss,
        "param_range_level" => os.param_range_level,
        "save_optimization_history" => os.save_optimization_history,
        "save_all_optim_restarts_results" => os.save_all_optim_restarts_results,
        "save_modeled_psd" => os.save_modeled_psd,
        "reparametrize" => os.reparametrize,
        "n_restarts" => os.n_restarts,
        "maxiters" => os.maxiters,
        "time_limit_minutes" => os.time_limit_minutes,
        "loss_settings" => Dict(
            "fmin" => ls.fmin,
            "fmax" => ls.fmax,
            "fbands" => ls.fbands,
            "psd_preproc" => ls.psd_preproc,
            "psd_window_size" => ls.psd_window_size,
            "psd_poly_order" => ls.psd_poly_order,
            "psd_rel_eps" => ls.psd_rel_eps,
            "psd_smooth_sigma" => ls.psd_smooth_sigma,
            "psd_welch_window_sec" => ls.psd_welch_window_sec,
            "psd_welch_overlap" => ls.psd_welch_overlap,
            "psd_welch_nperseg" => ls.psd_welch_nperseg,
            "psd_welch_nfft" => ls.psd_welch_nfft,
            "psd_noise_avg_reps" => ls.psd_noise_avg_reps,
            "sigma_meas" => ls.sigma_meas,
            "auto_initialize_sigma_meas" => ls.auto_initialize_sigma_meas,
            "loss_noise_seed" => ls.loss_noise_seed,
            "peak_bandwidth_hz" => ls.peak_bandwidth_hz,
            "peak_prominence_db" => ls.peak_prominence_db,
            "max_peak_windows" => ls.max_peak_windows,
            "weight_background" => ls.weight_background,
            "fspb_enabled" => ls.fspb_enabled,
            "peak_detection_empty" => ls.peak_detection_empty,
            "peak_baseline_window_hz" => ls.peak_baseline_window_hz,
            "peak_baseline_quantile" => ls.peak_baseline_quantile,
            "peak_min_frequency_hz" => ls.peak_min_frequency_hz,
            "peak_max_frequency_hz" => ls.peak_max_frequency_hz,
            "weight_fspb" => ls.weight_fspb,
            "weight_ssvep" => ls.weight_ssvep,
            "ssvep_enabled" => ls.ssvep_enabled,
            "ssvep_stim_freq_hz" => ls.ssvep_stim_freq_hz,
            "ssvep_n_harmonics" => ls.ssvep_n_harmonics,
            "ssvep_bandwidth_hz" => ls.ssvep_bandwidth_hz,
            "ssvep_harmonic_decay" => ls.ssvep_harmonic_decay,
            "max_abs_signal" => ls.max_abs_signal,
            "max_rms_growth" => ls.max_rms_growth
        ),
        "optimizer_settings" => Dict(
            "population_size" => opt_s.population_size,
            "sigma0" => opt_s.sigma0,
            "K" => opt_s.K,
            "n_samples" => opt_s.n_samples,
            "learning_rate" => opt_s.learning_rate
        )
    )
    
    # Sampling settings
    samp_s = settings.sampling_settings
    d["sampling_settings"] = if samp_s !== nothing
        Dict(
            "grammar_file" => samp_s.grammar_file,
            "n_samples" => samp_s.n_samples,
            "only_unique" => samp_s.only_unique,
            "max_resample_attempts" => samp_s.max_resample_attempts,
            "grammar_seed" => samp_s.grammar_seed
        )
    else
        Dict()
    end
    
    return d
end

"""
    save_settings(settings::Union{Settings, Dict}, filepath::Union{String, Nothing}=nothing; overwrite::Bool=true)::String

Save settings object or dictionary to a JSON file.

For Settings objects: By default, saves to `<path_out>/<exp_name>/settings.json`. Can optionally specify custom filepath.
For Dict objects: filepath must be provided.

# Arguments
- `settings::Union{Settings, Dict}`: Settings object or dictionary to save
- `filepath::Union{String, Nothing}`: Custom filepath (optional for Settings, required for Dict). If nothing with Settings, uses path_out/exp_name/settings.json
- `overwrite::Bool`: Whether to overwrite if file exists (default=true)

# Returns
The full path to the saved file

# Example
```julia
settings = create_default_settings()
settings.general_settings.exp_name = "MyExperiment"
save_settings(settings)  # Saves to ./results/MyExperiment/settings.json

save_settings(settings, "custom/path/settings.json")  # Custom path

# For Dict
settings_dict = Dict(...)
save_settings(settings_dict, "config.json")  # filepath required for Dict
```
"""
function save_settings(settings::Union{Settings, Dict}, filepath::Union{String, Nothing}=nothing; overwrite::Bool=true)::String
    # Determine filepath if not provided
    if filepath === nothing
        if settings isa Settings
            out_dir = joinpath(settings.general_settings.path_out, settings.general_settings.exp_name)
            filepath = joinpath(out_dir, "settings.json")
        else
            error("filepath must be provided when saving Dict objects")
        end
    end
    
    # Convert Settings to dict if needed
    settings_dict = if settings isa Settings
        settings_to_dict(settings)
    else
        settings
    end
    
    # Ensure directory exists
    dir = dirname(filepath)
    isempty(dir) || mkpath(dir)
    
    # Write JSON
    open(filepath, "w") do f
        JSON.print(f, settings_dict, 2)
    end
    
    vinfo("Settings saved to: $filepath")
    return filepath
end

"""
    print_settings_summary(settings::Union{Settings, Dict}; section::String="all", format_type::String="short")

Pretty-print settings configuration with detailed field information.

# Arguments
- `settings::Union{Settings, Dict}`: Settings object or dictionary
- `section::String`: Which section to display: "all", "general_settings", "network_settings", 
                     "simulation_settings", "optimization_settings", or "data_settings"
- `format_type::String`: Output format - "short" (default) or "long"
  - "short": Compact format showing only `key: value`
  - "long": Detailed format with types and descriptions

# Example
```julia
settings = create_default_settings()
print_settings_summary(settings; section="network_settings")  # Short format
print_settings_summary(settings; section="optimization_settings", format_type="long")
print_settings_summary(settings; format_type="short")  # All sections, compact
```
"""
function print_settings_summary(settings::Union{Settings, Dict}; section::String="all", format_type::String="short")
    # Convert Settings to dict for consistent handling
    settings_dict = if settings isa Settings
        settings_to_dict(settings)
    else
        settings
    end
    
    print_section(section, settings_dict, format_type)
end

function print_section(section::String, settings_dict::Dict, format_type::String="short")
    should_print(s) = section == "all" || section == s
    
    if format_type == "short"
        print_section_short(section, settings_dict)
    else
        print_section_long(section, settings_dict)
    end
end

function print_section_short(section::String, settings_dict::Dict)
    should_print(s) = section == "all" || section == s
    
    if should_print("general_settings")
        gs = get(settings_dict, "general_settings", Dict())
        println("\n[General Settings]")
        println("  exp_name: $(get(gs, "exp_name", "N/A"))")
        println("  path_out: $(get(gs, "path_out", "N/A"))")
        println("  verbosity_level: $(get(gs, "verbosity_level", 1))")
        println("  seed: $(get(gs, "seed", nothing))")
        println("  make_plots: $(get(gs, "make_plots", false))")
        println("  save_model_formats: $(get(gs, "save_model_formats", ["tex"]))")
    end
    
    if should_print("network_settings")
        ns = get(settings_dict, "network_settings", Dict())
        println("\n[Network Settings]")
        println("  n_nodes: $(get(ns, "n_nodes", "N/A"))")
        println("  node_names: $(join(get(ns, "node_names", []), ", "))")
        # Handle both String and RuleTree node models
        node_models_raw = get(ns, "node_models", [])
        node_models_display = if all(x -> x isa String for x in node_models_raw)
            join(node_models_raw, ", ")
        else
            join([x isa String ? x : serialize_rule_tree(x) for x in node_models_raw], ", ")
        end
        println("  node_models: $(node_models_display)")
        
        # node_coords: Print each coordinate on separate line
        node_coords = get(ns, "node_coords", [])
        println("  node_coords:")
        if isempty(node_coords)
            println("    (empty)")
        else
            for (i, coord) in enumerate(node_coords)
                println("    Node $i: $coord")
            end
        end
        
        # network_conn: Print matrix with rows
        conn = get(ns, "network_conn", [])
        println("  network_conn:")
        if isempty(conn)
            println("    (empty)")
        elseif conn isa Vector
            println("    $(conn)")
        else
            rows = length(conn)
            cols = (rows > 0 && conn[1] isa Vector) ? length(conn[1]) : 1
            println("    $rows × $cols matrix")
            for (i, row) in enumerate(conn)
                println("      Row $i: $(row)")
            end
        end
        
        # network_delay: Print matrix with rows
        delay = get(ns, "network_delay", [])
        println("  network_delay:")
        if isempty(delay)
            println("    (empty)")
        elseif delay isa Vector
            println("    $(delay)")
        else
            rows = length(delay)
            cols = (rows > 0 && delay[1] isa Vector) ? length(delay[1]) : 1
            println("    $rows × $cols matrix")
            for (i, row) in enumerate(delay)
                println("      Row $i: $(row)")
            end
        end
        
        # network_conn_funcs: Print matrix with rows
        funcs = get(ns, "network_conn_funcs", [])
        println("  network_conn_funcs:")
        if isempty(funcs)
            println("    (empty)")
        elseif funcs isa Vector
            println("    $(funcs)")
        else
            rows = length(funcs)
            cols = (rows > 0 && funcs[1] isa Vector) ? length(funcs[1]) : 1
            println("    $rows × $cols matrix")
            for (i, row) in enumerate(funcs)
                println("      Row $i: $(row)")
            end
        end
        
        println("  sensory_input_conn: $(join(get(ns, "sensory_input_conn", []), ", "))")
        println("  sensory_input_func: $(get(ns, "sensory_input_func", "none"))")
        println("  sensory_seed: $(get(ns, "sensory_seed", nothing))")
        println("  eeg_output: $(get(ns, "eeg_output", "default"))")
    end
    
    if should_print("simulation_settings")
        ss = get(settings_dict, "simulation_settings", Dict())
        println("\n[Simulation Settings]")
        tspan_val = get(ss, "tspan", [0.0, 1000.0])
        t1 = (tspan_val isa Vector) ? tspan_val[1] : tspan_val[1]
        t2 = (tspan_val isa Vector) ? tspan_val[2] : tspan_val[2]
        println("  tspan: [$t1, $t2]")
        println("  n_runs: $(get(ss, "n_runs", 1))")
        println("  dt: $(get(ss, "dt", "N/A"))")
        println("  saveat: $(get(ss, "saveat", "N/A"))")
        println("  solver: $(get(ss, "solver", "Tsit5"))")
    end
    
    if should_print("data_settings")
        ds = get(settings_dict, "data_settings", Dict())
        println("\n[Data Settings]")
        println("  data_file: $(get(ds, "data_file", ""))")
        println("  target_channel: $(get(ds, "target_channel", "IC3"))")
        println("  task_type: $(get(ds, "task_type", nothing))")
        println("  fs: $(get(ds, "fs", nothing))")
    end
    
    if should_print("optimization_settings")
        os = get(settings_dict, "optimization_settings", Dict())
        println("\n[Optimization Settings]")
        println("  method: $(get(os, "method", "CMAES"))")
        println("  loss: $(get(os, "loss", "fspb"))")
        println("  abs_target_loss: $(get(os, "abs_target_loss", 0.01))")
        println("  reparametrize: $(get(os, "reparametrize", true))")
        println("  param_range_level: $(get(os, "param_range_level", "high"))")
        println("  n_restarts: $(get(os, "n_restarts", 1))")
        println("  maxiters: $(get(os, "maxiters", 2000))")
        println("  time_limit_minutes: $(get(os, "time_limit_minutes", 120))")
        
        ls = get(os, "loss_settings", Dict())
        println("  loss_settings.fmin: $(get(ls, "fmin", 1.0))")
        println("  loss_settings.fmax: $(get(ls, "fmax", 48.0))")
    end
    
    if should_print("sampling_settings")
        samp = get(settings_dict, "sampling_settings", Dict())
        println("\n[Sampling Settings]")
        println("  grammar_file: $(get(samp, "grammar_file", "grammars/default_grammar.cfg"))")
        println("  n_samples: $(get(samp, "n_samples", 10))")
        println("  only_unique: $(get(samp, "only_unique", true))")
        println("  max_resample_attempts: $(get(samp, "max_resample_attempts", 100))")
        println("  grammar_seed: $(get(samp, "grammar_seed", nothing))")
    end
end

function print_section_long(section::String, settings_dict::Dict)
    should_print(s) = section == "all" || section == s
    
    if should_print("general_settings")
        gs = get(settings_dict, "general_settings", Dict())
        println("\n[General Settings]")
        println("  exp_name              :: String   | Name of the experiment")
        println("    → $(get(gs, "exp_name", "N/A"))")
        println("  path_out              :: String      | Base output directory")
        println("    → $(get(gs, "path_out", "N/A"))")
        println("  verbosity_level       :: Int         | 0=silent, 1=minimal, 2=detailed")
        println("    → $(get(gs, "verbosity_level", 1))")
        println("  seed                  :: Int?        | Random seed (nothing=random)")
        println("    → $(get(gs, "seed", nothing))")
        println("  make_plots            :: Bool        | Generate plot output (PNG)")
        println("    → $(get(gs, "make_plots", false))")
        println("  save_model_formats    :: Vector      | Formats for saving models (tex only, tvb coming)")
        println("    → $(get(gs, "save_model_formats", ["tex"]))")
    end
    
    if should_print("network_settings")
        ns = get(settings_dict, "network_settings", Dict())
        println("\n[Network Settings]")
        println("  n_nodes               :: Int      | Number of nodes in network")
        println("    → $(get(ns, "n_nodes", "N/A"))")
        println("  node_names            :: Vector   | Names of each node")
        println("    → $(join(get(ns, "node_names", []), ", "))")
        println("  node_models           :: Vector   | Model per node (Unknown=grammar-sampled)")
        # Handle both String and RuleTree node models
        node_models_raw = get(ns, "node_models", [])
        node_models_display = if all(x -> x isa String for x in node_models_raw)
            join(node_models_raw, ", ")
        else
            join([x isa String ? x : serialize_rule_tree(x) for x in node_models_raw], ", ")
        end
        println("    → $(node_models_display)")
        
        # node_coords: Print actual coordinates
        node_coords = get(ns, "node_coords", [])
        println("  node_coords           :: Vector   | 3D coordinates of each node")
        if isempty(node_coords)
            println("    → (empty)")
        else
            for (i, coord) in enumerate(node_coords)
                println("    → Node $i: $coord")
            end
        end
        
        # network_conn: Print matrix dimensions and values
        conn = get(ns, "network_conn", [])
        println("  network_conn          :: Matrix   | Connection strengths (n_nodes × n_nodes)")
        if isempty(conn)
            println("    → (empty)")
        elseif conn isa Vector
            println("    → $(conn)")
        else
            rows = length(conn)
            cols = (rows > 0 && conn[1] isa Vector) ? length(conn[1]) : 1
            println("    → $rows × $cols matrix")
            for (i, row) in enumerate(conn)
                println("       Row $i: $(row)")
            end
        end
        
        # network_delay: Print matrix
        delay = get(ns, "network_delay", [])
        println("  network_delay         :: Matrix   | Transmission delays (n_nodes × n_nodes)")
        if isempty(delay)
            println("    → (empty)")
        elseif delay isa Vector
            println("    → $(delay)")
        else
            rows = length(delay)
            cols = (rows > 0 && delay[1] isa Vector) ? length(delay[1]) : 1
            println("    → $rows × $cols matrix")
            for (i, row) in enumerate(delay)
                println("       Row $i: $(row)")
            end
        end
        
        # network_conn_funcs: Print matrix
        funcs = get(ns, "network_conn_funcs", [])
        println("  network_conn_funcs    :: Matrix   | Connection function types")
        if isempty(funcs)
            println("    → (empty)")
        elseif funcs isa Vector
            println("    → $(funcs)")
        else
            rows = length(funcs)
            cols = (rows > 0 && funcs[1] isa Vector) ? length(funcs[1]) : 1
            println("    → $rows × $cols matrix")
            for (i, row) in enumerate(funcs)
                println("       Row $i: $(row)")
            end
        end
        
        println("  sensory_input_conn    :: Vector   | Which nodes receive sensory input")
        println("    → $(join(get(ns, "sensory_input_conn", []), ", "))")
        println("  sensory_input_func    :: String   | Input dynamics function")
        println("    → $(get(ns, "sensory_input_func", "none"))")
        println("  sensory_seed      :: Int?     | Seed for input randomization")
        println("    → $(get(ns, "sensory_seed", nothing))")
        println("  eeg_output            :: String   | EEG recording specification")
        println("    → $(get(ns, "eeg_output", "default"))")
    end
    
    if should_print("simulation_settings")
        ss = get(settings_dict, "simulation_settings", Dict())
        println("\n[Simulation Settings]")
        tspan_val = get(ss, "tspan", [0.0, 1000.0])
        t1 = (tspan_val isa Vector) ? tspan_val[1] : tspan_val[1]
        t2 = (tspan_val isa Vector) ? tspan_val[2] : tspan_val[2]
        println("  tspan                 :: Vector   | Time interval (ms): [start, end]")
        println("    → [$t1, $t2]")
        println("  n_runs                :: Int      | Number of independent runs")
        println("    → $(get(ss, "n_runs", 1))")
        println("  dt                    :: Float    | Time step (ms)")
        println("    → $(get(ss, "dt", "N/A"))")
        println("  saveat                :: Float    | Output save frequency (ms)")
        println("    → $(get(ss, "saveat", "N/A"))")
        println("  solver                :: String   | ODE solver (Tsit5, RK4, etc.)")
        println("    → $(get(ss, "solver", "Tsit5"))")
        skw = get(ss, "solver_kwargs", Dict())
        println("  solver_kwargs         :: Dict     | Solver configuration")
        println("    abstol → $(get(skw, "abstol", 1e-6))")
        println("    reltol → $(get(skw, "reltol", 1e-5))")
        println("    maxiters → $(get(skw, "maxiters", 1e5))")
    end
    
    if should_print("optimization_settings")
        os = get(settings_dict, "optimization_settings", Dict())
        println("\n[Optimization Settings]")
        println("  method                :: String   | Optimizer: CMAES (only option)")
        println("    → $(get(os, "method", "CMAES"))")
        println("  loss                  :: String   | Loss function: fspb, psd_iae, etc.")
        println("    → $(get(os, "loss", "fspb"))")
        println("  abs_target_loss       :: Float    | Absolute loss target")
        println("    → $(get(os, "abs_target_loss", 0.01))")
        println("  reparametrize         :: Bool     | Use reparametrization strategy")
        println("    → $(get(os, "reparametrize", true))")
        println("  param_range_level     :: String   | Parameter range level (high/medium/low)")
        println("    → $(get(os, "param_range_level", "high"))")
        println("  n_restarts            :: Int      | Number of optimization restarts")
        println("    → $(get(os, "n_restarts", 1))")
        println("  maxiters              :: Int      | Maximum iterations per restart")
        println("    → $(get(os, "maxiters", 2000))")
        println("  time_limit_minutes    :: Int      | Time limit for optimization")
        println("    → $(get(os, "time_limit_minutes", 120))")
        println("  loss_abstol           :: Float    | Absolute tolerance for loss")
        println("    → $(get(os, "loss_abstol", 1e-5))")
        println("  loss_reltol           :: Float    | Relative tolerance for loss")
        println("    → $(get(os, "loss_reltol", 1e-5))")
        println("  save_optimization_history :: Bool | Save optimization trajectory")
        println("    → $(get(os, "save_optimization_history", false))")
        println("  save_all_optim_restarts_results :: Bool | Automatically save all restart results")
        println("    → $(get(os, "save_all_optim_restarts_results", true))")
        println("  save_modeled_psd      :: Bool     | Save modeled PSD output")
        println("    → $(get(os, "save_modeled_psd", false))")
        
        ls = get(os, "loss_settings", Dict())
        println("\n  [Loss Settings]")
        println("    loss_fn             :: String   | fmin=1.0, fmax=48.0, fbands")
        println("      → fmin            → $(get(ls, "fmin", 1.0))")
        println("      → fmax            → $(get(ls, "fmax", 48.0))")
        println("      → fbands          → $(join(get(ls, "fbands", ["delta", "theta", "alpha", "betalow", "betahigh"]), ", "))")
        println("    psd_settings")
        println("      → preproc         → $(get(ls, "psd_preproc", "log10"))")
        println("      → window_size     → $(get(ls, "psd_window_size", 5))")
        println("      → poly_order      → $(get(ls, "psd_poly_order", 2))")
        println("      → rel_eps         → $(get(ls, "psd_rel_eps", 1e-12))")
        println("      → smooth_sigma    → $(get(ls, "psd_smooth_sigma", 1.0))")
        println("      → welch_window    → $(get(ls, "psd_welch_window_sec", 2.0)) sec")
        println("      → welch_overlap   → $(get(ls, "psd_welch_overlap", 0.5))")
        println("      → welch_nperseg   → $(get(ls, "psd_welch_nperseg", 0))")
        println("      → noise_avg_reps  → $(get(ls, "psd_noise_avg_reps", 1))")
        println("    fspb_settings (peak tracking)")
        println("      → enabled         → $(get(ls, "fspb_enabled", true))")
        println("      → min_frequency   → $(get(ls, "peak_min_frequency_hz", 5.0)) Hz")
        println("      → max_frequency   → $(get(ls, "peak_max_frequency_hz", 45.0)) Hz")
        println("      → bandwidth       → $(get(ls, "peak_bandwidth_hz", 6.0)) Hz")
        println("      → prominence      → $(get(ls, "peak_prominence_db", 0.5)) dB")
        println("      → max_windows     → $(get(ls, "max_peak_windows", 2))")
        println("      → baseline_window → $(get(ls, "peak_baseline_window_hz", 6.0)) Hz")
        println("      → baseline_quantile → $(get(ls, "peak_baseline_quantile", 0.2))")
        println("      → weight          → $(get(ls, "weight_fspb", 1.0))")
        println("    noise_settings")
        println("      → sigma_meas      → $(get(ls, "sigma_meas", 0.0))")
        println("      → auto_initialize → $(get(ls, "auto_initialize_sigma_meas", true))")
        println("      → seed            → $(get(ls, "loss_noise_seed", nothing))")
        println("    ssvep_settings")
        println("      → enabled         → $(get(ls, "ssvep_enabled", true))")
        println("      → stim_freq       → $(get(ls, "ssvep_stim_freq_hz", 5.0)) Hz")
        println("      → n_harmonics     → $(get(ls, "ssvep_n_harmonics", 3))")
        println("      → bandwidth       → $(get(ls, "ssvep_bandwidth_hz", 0.5)) Hz")
        println("      → harmonic_decay  → $(get(ls, "ssvep_harmonic_decay", 0.7))")
        println("      → weight          → $(get(ls, "weight_ssvep", 1.0))")
        println("    loss_weights")
        println("      → weight_background → $(get(ls, "weight_background", 0.4))")
        println("    signal_bounds")
        println("      → max_abs_signal  → $(get(ls, "max_abs_signal", 100.0))")
        println("      → max_rms_growth  → $(get(ls, "max_rms_growth", 100.0))")
        
        os_set = get(os, "optimizer_settings", Dict())
        println("\n  [Optimizer Settings (CMAES)]")
        println("    population_size     :: Int      | Population size for CMAES")
        println("      → $(get(os_set, "population_size", 50))")
        println("    sigma0              :: Float    | Initial step size")
        println("      → $(get(os_set, "sigma0", 1.0))")
        println("    K                   :: Float    | Learning rate factor")
        println("      → $(get(os_set, "K", 0.5))")
        println("    n_samples           :: Int      | Number of samples per generation")
        println("      → $(get(os_set, "n_samples", 100))")
        println("    learning_rate       :: Float    | Adam learning rate")
        println("      → $(get(os_set, "learning_rate", 0.1))")
    end
    
    if should_print("data_settings")
        ds = get(settings_dict, "data_settings", Dict())
        println("\n[Data Settings]")
        println("  data_file             :: String   | Path to data CSV file")
        println("    → $(get(ds, "data_file", ""))")
        println("  target_channel        :: String   | Channel/component to fit")
        println("    → $(get(ds, "target_channel", "IC3"))")
        println("  task_type             :: String?  | Type of task (optional)")
        println("    → $(get(ds, "task_type", nothing))")
        println("  fs                    :: Float?   | Sampling frequency (optional)")
        println("    → $(get(ds, "fs", nothing))")
    end
    
    if should_print("sampling_settings")
        samp = get(settings_dict, "sampling_settings", Dict())
        println("\n[Sampling Settings]")
        println("  grammar_file          :: String   | Path to grammar configuration file")
        println("    → $(get(samp, "grammar_file", "grammars/default_grammar.cfg"))")
        println("  n_samples             :: Int      | Number of models to sample")
        println("    → $(get(samp, "n_samples", 10))")
        println("  only_unique           :: Bool     | Keep only unique models")
        println("    → $(get(samp, "only_unique", true))")
        println("  max_resample_attempts :: Int      | Max attempts to resample for uniqueness")
        println("    → $(get(samp, "max_resample_attempts", 100))")
        println("  grammar_seed          :: Int?     | Random seed for sampling (nothing=random)")
        println("    → $(get(samp, "grammar_seed", nothing))")
    end
end

"""
    load_settings_file(settings_path::String)::Settings

Load and initialize settings from a JSON file with workflow setup.

This is the primary entry point for loading ENEEGMA configuration files.
It handles:
- Loading JSON from file
- Migrating legacy hyperparameter sweep settings
- Building Settings object (all defaults applied by constructors)
- Setting up task-local storage
- Configuring global verbosity level
- Displaying network configuration summary

# Arguments
- `settings_path::String`: Path to settings JSON file

# Returns
Fully initialized Settings object ready for use

# Example
```julia
settings = load_settings_file("experiments/my_settings.json")
```
"""
function load_settings_file(settings_path::String)::Settings
    # Load settings from file
    settings = load_settings_from_file(settings_path)

    # Ensure general_settings exists
    general_dict = nothing
    if haskey(settings, "general_settings") && settings["general_settings"] !== nothing
        general_dict = settings["general_settings"]
    end
    if general_dict === nothing
        general_dict = Dict{String, Any}()
        settings["general_settings"] = general_dict
    end

    # Ensure hyperparameter sweep settings live under optimization settings
    legacy = nothing
    if haskey(settings, "hyperparameter_sweep")
        legacy = settings["hyperparameter_sweep"]
        delete!(settings, "hyperparameter_sweep")
    elseif haskey(settings, "hyper_sweep")
        legacy = settings["hyper_sweep"]
        delete!(settings, "hyper_sweep")
    end
    if legacy !== nothing
        optdict = get!(settings, "optimization_settings", Dict{String, Any}())
        if !haskey(optdict, "hyperparameter_sweep")
            optdict["hyperparameter_sweep"] = legacy
        end
    end

    # Build Settings object from loaded settings dict
    # The Settings constructors handle all default values internally
    s = Settings(settings)
    set_task_settings(s)  # Store settings in task-local storage
    set_verbose(s.general_settings.verbosity_level) # Set global verbosity level based on settings

    print_settings_summary(s; section="network_settings")
    flush(stdout)

    return s
end