const DEFAULT_TARGET_CHANNEL = "IC3"

function manage_settings(settings_path::String;
                      settings_idx::Union{Int, Nothing}=nothing
                      )::Settings
    # Load settings and fixate to one combination (if multiple)
    settings = load_settings(settings_path)

    general_dict = nothing
    if haskey(settings, "general_settings") && settings["general_settings"] !== nothing
        general_dict = settings["general_settings"]
    end
    if general_dict === nothing
        general_dict = Dict{String, Any}()
        settings["general_settings"] = general_dict
    end

    override_idx = _coerce_settings_idx_value(get(general_dict, "settings_idx", nothing))
    effective_idx = settings_idx === nothing ? override_idx : Int(settings_idx)
    effective_idx = effective_idx === nothing ? 1 : effective_idx
    general_dict["settings_idx"] = effective_idx
    general_dict["settings_path"] = settings_path

    if haskey(settings, "network_settings") && settings["network_settings"] !== nothing
        netdict = settings["network_settings"]
        if !haskey(netdict, "seed_sensory_input")
            netdict["seed_sensory_input"] = nothing
        end
    end

    optdict = get!(settings, "optimization_settings", Dict{String, Any}())
    if !haskey(optdict, "reparametrize")
        optdict["reparametrize"] = false
    end
    if !haskey(optdict, "abs_target_loss")
        optdict["abs_target_loss"] = 0.1
    end
    if !haskey(optdict, "save_all_optim_restarts_results")
        optdict["save_all_optim_restarts_results"] = false
    end
    if !haskey(optdict, "save_modeled_psd")
        optdict["save_modeled_psd"] = false
    end
    if !haskey(optdict, "component_fit")
        optdict["component_fit"] = "all"
    end

    if !haskey(optdict, "optimizer_settings") || optdict["optimizer_settings"] === nothing
        if haskey(settings, "optimizer_settings") && settings["optimizer_settings"] !== nothing
            optdict["optimizer_settings"] = settings["optimizer_settings"]
            delete!(settings, "optimizer_settings")
        else
            optdict["optimizer_settings"] = Dict{String, Any}()
        end
    end

    lossdict = if haskey(optdict, "loss_settings") && optdict["loss_settings"] !== nothing
        optdict["loss_settings"]
    elseif haskey(settings, "loss_settings") && settings["loss_settings"] !== nothing
        section = settings["loss_settings"]
        optdict["loss_settings"] = section
        delete!(settings, "loss_settings")
        section
    else
        section = Dict{String, Any}()
        optdict["loss_settings"] = section
        section
    end
    if !(lossdict isa Dict{String, Any})
        lossdict = Dict{String, Any}(lossdict)
        optdict["loss_settings"] = lossdict
    end
    normalize_loss_settings_dict!(lossdict)
    lossdict["sigma_meas"] = get(lossdict, "sigma_meas", 0.0)
    if !haskey(lossdict, "noise_seed")
        lossdict["noise_seed"] = nothing
    end
    lossdict["psd_preproc"] = get(lossdict, "psd_preproc", nothing)
    lossdict["psd_window_size"] = get(lossdict, "psd_window_size", 5)
    lossdict["psd_poly_order"] = get(lossdict, "psd_poly_order", 2)
    lossdict["psd_rel_eps"] = get(lossdict, "psd_rel_eps", 1e-12)
    lossdict["psd_smooth_sigma"] = get(lossdict, "psd_smooth_sigma", 1.0)
    peak_defaults = Dict(
        "peak_bandwidth_hz" => 6.0,
        "peak_prominence_db" => 0.5,
        "max_peak_windows" => 2,
        "weight_background" => 0.4,
        "peak_baseline_window_hz" => 6.0,
        "peak_baseline_quantile" => 0.2,
        "peak_min_frequency_hz" => 5.0,
        "peak_max_frequency_hz" => 45.0,
        "max_abs_signal" => 100.0,
        "max_rms_growth" => 100.0
    )
    for (k, v) in peak_defaults
        lossdict[k] = get(lossdict, k, v)
    end

    if haskey(settings, "data_settings") && settings["data_settings"] !== nothing
        datadict = settings["data_settings"]
        datadict["target_channel"] = get(datadict, "target_channel", DEFAULT_TARGET_CHANNEL)
        datadict["task_type"] = get(datadict, "task_type", nothing)
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
    if legacy !== nothing && !haskey(optdict, "hyperparameter_sweep")
        optdict["hyperparameter_sweep"] = legacy
    end
    if !haskey(optdict, "hyperparameter_sweep") || optdict["hyperparameter_sweep"] === nothing
        optdict["hyperparameter_sweep"] = Dict{String, Any}()
    end
    sweep_section = optdict["hyperparameter_sweep"]
    if sweep_section isa AbstractDict && haskey(sweep_section, "raw")
        delete!(sweep_section, "raw")
    end

    # Build Settings via convenience constructor (handles optional sub-settings)
    s = Settings(settings, effective_idx)
    set_task_settings(s)  # Store settings in task-local storage
    set_verbose(s.general_settings.verbosity_level) # Set global verbosity level based on settings

    print_network_info(s.network_settings; verbosity_level=s.general_settings.verbosity_level)
    vprint("--- SETTINGS LOADED for $(s.network_settings.network_name)")
    flush(stdout)

    return s
end

"""
    load_settings(settings_path)

Load settings from a JSON file.
"""
function load_settings(settings_path::String)::Dict{String, Any}
    # Check if the settings_path ends with .json, otherwise give error 
    if !endswith(settings_path, ".json")
        println("Error: The path to the settings file must end with .json. Give the full path to the settings file.")
        exit(1)
    end
    return JSON.parsefile(settings_path; dicttype=Dict{String, Any})
end

function save_settings_file(settings_path::AbstractString, settings::Settings; overwrite::Bool=true)::String
    out_dir = settings.general_settings.path_out
    isempty(out_dir) && error("general_settings.path_out is empty; cannot save settings copy.")
    isdir(out_dir) || mkpath(out_dir)
    dest = joinpath(out_dir, basename(String(settings_path)))
    cp(String(settings_path), dest; force=overwrite)
    return dest
end


"""
    print_network_info(ns::NetworkSettings, settings_idx::Int, total_combinations::Int, combination_info::Tuple{String, String, String, String})

Print detailed information about the network configuration and combination settings in a formatted way.
"""
function print_network_info(ns::NetworkSettings; verbosity_level::Int=1)
    width = 40
    if verbosity_level <= 1
        return
    end
    vprint("\n"*create_border(width); level=2)
    vprint(create_header("NETWORK CONFIGURATION: $(ns.network_name)", width); level=2)
    vprint(create_border(width); level=2)
    vprint("\nNodes ($(ns.n_nodes)): $(join(ns.node_names, ", "))"; level=2)
    vprint("Models: $(join(ns.node_models, ", "))"; level=2)
    vprint("\nCoords:"; level=2)
    for (i,(x,y,z)) in enumerate(ns.node_coords)
        vprint("  $(ns.node_names[i]): ($(round(x,digits=2)),$(round(y,digits=2)),$(round(z,digits=2)))"; level=2)
    end
    vprint("\nConnectivity:"; level=2)
    print_matrix(ns.network_conn, ns.node_names; level=2)
    vprint("Delays:"; level=2)
    print_matrix(ns.network_delay, ns.node_names; level=2)
    if !isempty(ns.network_conn_funcs)
        vprint("\nConn funcs:"; level=2)
        n = min(length(ns.node_names), size(ns.network_conn_funcs, 1))
        for i in 1:n
            row = ns.network_conn_funcs[i, :]
            vprint("  $(ns.node_names[i]): " * join([s == "" ? "none" : s for s in row], ", "); level=2)
        end
    end
    # Sensory input summary: function and targets
    targets = [ns.node_names[i] for i in 1:length(ns.node_names) if ns.sensory_input_conn[i] != 0]
    vprint("\nSensory input:"; level=2)
    vprint("  funcs: " * (isempty(ns.sensory_input_func) ? "none" : ns.sensory_input_func); level=2)
    vprint("  targets: " * (isempty(targets) ? "none" : join(targets, ", ")); level=2)
    vprint("EEG output: " * (isempty(ns.eeg_output) ? "default" : ns.eeg_output); level=2)
    vprint(create_border(width); level=2)
end

# Helper function to create borders
function create_border(width::Int=80)
    return repeat("═", width)
end

# Helper to create centered headers
function create_header(text::String, width::Int=80)
    padding = max(0, width - length(text) - 2)
    left_pad = div(padding, 2)
    right_pad = padding - left_pad
    return "║" * repeat(" ", left_pad) * text * repeat(" ", right_pad) * "║"
end

"""
    print_matrix(matrix::Matrix, labels::Vector{String})

Print a matrix with row and column labels in a nicely formatted way.
"""
function print_matrix(matrix::Matrix, labels::Vector{String}; level::Int=2)
    n = size(matrix, 1)
    col_widths = [maximum([length(labels[j]), maximum(length(string(round(matrix[i, j], digits=2))) for i in 1:n)]) for j in 1:n]
    vprint("       │" * join([" " * rpad(label, col_widths[j]) * " │" for (j, label) in enumerate(labels)], ""); level=level)
    vprint("   ────┼" * join([repeat("─", col_widths[j] + 2) * "┼" for j in 1:n], ""); level=level)
    for i in 1:n
        row_str = "   " * rpad(labels[i], 4) * "│"
        for j in 1:n
            val = round(matrix[i, j], digits=2)
            row_str *= " " * rpad(string(val), col_widths[j]) * " │"
        end
        vprint(row_str; level=level)
    end
end

function generate_settings_file(settings_path::String; kwargs...)
    # Load default settings from example_settings_short.json
    default_settings_path = joinpath(@__DIR__, "..", "..", "examples", "example_settings_1node.json")
    settings = JSON.parsefile(default_settings_path; dicttype=Dict)

    # Helper to update nested fields
    function update_nested!(settings::Dict, key::String, value)
        # Try top-level keys first
        if haskey(settings, key)
            settings[key] = value
            return
        end
        # Try nested keys
        for (section, subdict) in settings
            if isa(subdict, Dict) && haskey(subdict, key)
                # println("Updating key '$key' in section '$section'... ")
                subdict[key] = value
                return
            end
        end
        # If not found, try to split key (e.g. "network_settings.node_models")
        if occursin(".", key)
            parts = split(key, ".")
            if length(parts) == 2 && haskey(settings, parts[1])
                settings[parts[1]][parts[2]] = value
                return
            end
        end
        # If not found, give a warning that the key does not exist
        vwarn("Key '$key' not found in settings. No update performed.")
    end

    # Update settings with kwargs
    for (k, v) in kwargs
        update_nested!(settings, string(k), v)
    end

    # Save to specified file
    if !isdir(dirname(settings_path))
        mkpath(dirname(settings_path))
    end

    open(settings_path, "w") do f
        JSON.print(f, settings, 4)
    end
end
