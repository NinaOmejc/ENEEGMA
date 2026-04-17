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
    vinfo("Data loaded from: $(ds.data_file)", level=2)
    return data
end

"""
    normalize_parameter_name(name::String)::String

Normalize parameter names for JSON output and display.
Converts naming convention: __[letter] → _x
Example: N1__c11 → N1_x11

This handles parameter names with double underscores followed by a single letter,
common in symbolically-generated parameter names.

# Arguments
- `name::String`: Parameter name to normalize

# Returns
String: Normalized parameter name
"""
function normalize_parameter_name(name::String)::String
    return replace(name, r"__[a-z]" => "_x")
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
    vinfo("Settings loaded from: $filepath", level=2)
    return settings
end

"""
    construct_output_dir(gs::GeneralSettings, ns::NetworkSettings)::String

Construct and create the output directory path based on GeneralSettings and NetworkSettings.

Builds hierarchical directory: `path_out / exp_name / network_name/`
If network_name is set to a value other than "net1", it creates a subdirectory for that network.
The directory is created if it doesn't already exist.

# Arguments
- `gs::GeneralSettings`: General settings containing path_out and exp_name
- `ns::NetworkSettings`: Network settings containing network name

# Returns
String: The full path to the output directory

# Example
```julia
gs = settings.general_settings
ns = settings.network_settings
output_dir = construct_output_dir(gs, ns)
# Returns: "./results/default_exp/net1/" or "./results/my_exp/my_network/"
```
"""
function construct_output_dir(gs::GeneralSettings, ns::NetworkSettings)::String
    output_dir = joinpath(gs.path_out, gs.exp_name)
    output_dir = joinpath(output_dir, ns.name)
    if !isdir(output_dir)
        mkpath(output_dir)
    end
    return output_dir
end

"""
    find_next_numbered_folder(base_path::String, prefix::String="optimization")::String

Find the next available numbered folder in a directory, creating it if needed.

Creates folders with numeric suffixes (e.g., `optimization_1/`, `optimization_2/`) 
incrementally. Used for organizing job outputs so each run gets its own numbered folder.

This function is reusable for both optimization and hyperparameter sweep jobs,
as well as grammar sampling results.

# Arguments
- `base_path::String`: Base directory where numbered folders should be created
- `prefix::String`: Prefix for folder names (default: "optimization")

# Returns
String: Full path to the next available numbered folder (e.g., "./results/exp/optimization_1")

# Example
```julia
output_dir = find_next_numbered_folder("./results/my_exp", "optimization")
# Returns: "./results/my_exp/optimization_1/" if it doesn't exist
# Returns: "./results/my_exp/optimization_2/" if optimization_1 already exists
```
"""
function find_next_numbered_folder(base_path::String, prefix::String="optimization")::String
    mkpath(base_path)
    idx = 1
    while isdir(joinpath(base_path, "$(prefix)_$idx"))
        idx += 1
    end
    return joinpath(base_path, "$(prefix)_$idx")
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
    s = Settings(Dict{String, Any}())
    set_task_settings(s)  # Store settings in task-local storage and set verbosity
    vinfo("Default settings created successfully."; level=1)
    return s
end

"""
    struct_to_ordered_dict(obj::T; exclude::Set{Symbol}=Set{Symbol}())::OrderedDict where T

Reflection-based serialization: Convert any struct to OrderedDict by introspecting its fields.
Automatically handles RuleTree objects by serializing them.

# Arguments
- `obj::T`: Any struct object to serialize
- `exclude::Set{Symbol}`: Field names to skip during serialization

# Returns
OrderedDict with field names as keys and field values as values
"""
function struct_to_ordered_dict(obj::T; exclude::Set{Symbol}=Set{Symbol}())::OrderedDict where T
    out = OrderedDict{String, Any}()
    
    for fname in fieldnames(T)
        fname in exclude && continue
        
        val = getfield(obj, fname)
        key = String(fname)
        
        # Handle RuleTree serialization
        if val isa RuleTree
            out[key] = serialize_rule_tree(val)
        # Handle nested AbstractSettings objects
        elseif val isa AbstractSettings && !isa(val, RuleTree)
            out[key] = struct_to_ordered_dict(val; exclude)
        else
            out[key] = val
        end
    end
    
    return out
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
function settings_to_dict(settings::Settings)::OrderedDict{String, Any}
    d = OrderedDict{String, Any}()
    
    # Serialize each settings component using reflection, with custom handling for complex fields
    
    # General settings - serialize all fields
    d["general_settings"] = struct_to_ordered_dict(settings.general_settings)
    
    # Network settings - needs custom RuleTree handling
    ns = settings.network_settings
    node_models_serialized = [m isa String ? m : serialize_rule_tree(m) for m in ns.node_models]
    d["network_settings"] = struct_to_ordered_dict(ns)
    d["network_settings"]["node_models"] = node_models_serialized
    
    # Sampling settings - serialize if not nothing
    samp_s = settings.sampling_settings
    d["sampling_settings"] = if samp_s !== nothing
        struct_to_ordered_dict(samp_s)
    else
        OrderedDict()
    end
    
    # Simulation settings - serialize all fields
    ss = settings.simulation_settings
    sim_dict = struct_to_ordered_dict(ss)
    # Convert tspan tuple to array for JSON
    sim_dict["tspan"] = collect(ss.tspan)
    d["simulation_settings"] = sim_dict
    
    # Data settings - serialize if not nothing
    data_s = settings.data_settings
    d["data_settings"] = if data_s !== nothing
        data_dict = struct_to_ordered_dict(data_s; exclude=Set([:workspace]))
        # Nested PSD settings are already serialized via reflection
        data_dict
    else
        OrderedDict()
    end
    
    # Optimization settings - requires custom handling for nested structures
    os = settings.optimization_settings
    d["optimization_settings"] = OrderedDict(
        "method" => os.method,
        "param_bound_scaling_level" => os.param_bound_scaling_level,
        "save_optimization_history" => os.save_optimization_history,
        "save_modeled_psd" => os.save_modeled_psd,
        "include_settings_in_results_output" => os.include_settings_in_results_output,
        "reparametrize" => os.reparametrize,
        "n_restarts" => os.n_restarts,
        "maxiters" => os.maxiters,
        "time_limit_minutes" => os.time_limit_minutes,
        "loss_settings" => struct_to_ordered_dict(os.loss_settings),
        "optimizer_settings" => struct_to_ordered_dict(os.optimizer_settings),
        "hyperparameter_sweep" => os.hyperparameter_sweep.hyperparameters
    )
    
    return d
end

"""
    format_json_with_indent(json_str::String; indent=2)

Format a compact JSON string with proper indentation while preserving key order.
"""
function format_json_with_indent(json_str::String; indent=2)
    result = String[]
    current_indent = 0
    i = 1
    in_array = false
    array_depth = 0
    
    while i <= length(json_str)
        c = json_str[i]
        
        if c == '{'
            push!(result, "{")
            current_indent += indent
            # Check if next non-whitespace is }
            j = i + 1
            while j <= length(json_str) && json_str[j] in (' ', '\t', '\n', '\r')
                j += 1
            end
            if j <= length(json_str) && json_str[j] != '}'
                push!(result, "\n" * " "^current_indent)
            end
        elseif c == '}'
            current_indent -= indent
            if !isempty(result) && result[end] != "{"
                push!(result, "\n" * " "^current_indent)
            end
            push!(result, "}")
        elseif c == '['
            push!(result, "[")
            array_depth += 1
        elseif c == ']'
            array_depth -= 1
            push!(result, "]")
        elseif c == ','
            push!(result, ",")
            # Only add newline if we're not inside an array
            if array_depth == 0
                j = i + 1
                while j <= length(json_str) && json_str[j] in (' ', '\t', '\n', '\r')
                    j += 1
                end
                if j <= length(json_str) && json_str[j] != '}'
                    push!(result, "\n" * " "^current_indent)
                end
            else
                # In array: add space after comma only if not followed by newline
                j = i + 1
                while j <= length(json_str) && json_str[j] in ('\t', '\n', '\r')
                    j += 1
                end
                if j <= length(json_str) && json_str[j] == ' '
                    push!(result, " ")
                    i = j
                    continue
                end
            end
        elseif c in (' ', '\t', '\n', '\r')
            # Skip whitespace, we'll add our own
            # Exception: preserve space after comma in arrays
            if !isempty(result) && result[end] == ","
                # Skip, we handled it above
            end
        else
            push!(result, string(c))
        end
        
        i += 1
    end
    
    return join(result)
end

"""
    save_settings(settings::Union{Settings, Dict}, filepath::Union{String, Nothing}=nothing)::String

Save settings object or dictionary to a JSON file.

For Settings objects: By default, saves to `<path_out>/<exp_name>/settings.json`. Can optionally specify custom filepath.
For Dict objects: filepath must be provided.

# Arguments
- `settings::Union{Settings, Dict}`: Settings object or dictionary to save
- `filepath::Union{String, Nothing}`: Custom filepath (optional for Settings, required for Dict). If nothing with Settings, uses path_out/exp_name/settings.json

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
function save_settings(settings::Union{Settings, Dict}, filepath::Union{String, Nothing}=nothing)::String
    # Set verbosity from Settings if available (so vinfo() calls work correctly)
    if settings isa Settings
        set_verbose(settings.general_settings.verbosity_level)
    end
    
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
    
    # Write JSON with pretty formatting while preserving OrderedDict order
    # Use JSON.print to preserve OrderedDict key order
    open(filepath, "w") do f
        JSON.print(f, settings_dict, 2)
    end
    
    vinfo("Settings saved to: $filepath", level=2)
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
    println()  # Add extra newline at end for readability
end

function print_section(section::String, settings_dict::AbstractDict, format_type::String="short")
    should_print(s) = section == "all" || section == s
    
    if format_type == "short"
        print_section_short(section, settings_dict)
    else
        print_section_long(section, settings_dict)
    end
end

"""
    format_value_short(value::Any)::String
    
Format a value for short display (single line).
"""
function format_value_short(value::Any)::String
    if value === nothing
        return "nothing"
    elseif value isa Bool
        return string(value)
    elseif value isa Number
        return string(value)
    elseif value isa AbstractString
        return value
    elseif value isa Vector
        if isempty(value)
            return "[]"
        elseif all(v -> v isa Number || v isa Bool || v isa Nothing, value)
            # Simple vector of numbers/bools
            return join(value, ", ")
        elseif all(v -> v isa AbstractString, value)
            # Vector of strings
            return join(value, ", ")
        else
            # Complex vector
            return string(length(value)) * " items"
        end
    elseif value isa AbstractDict
        return "{" * string(length(value)) * " keys}"
    elseif value isa Tuple
        return "(" * join(value, ", ") * ")"
    else
        return string(value)
    end
end

"""
    format_value_long(value::Any, indent::Int=0)::String
    
Format a value for long display (potentially multi-line).
"""
function format_value_long(value::Any, indent::Int=0)::String
    prefix = " "^indent
    
    if value === nothing
        return "nothing"
    elseif value isa Bool
        return string(value)
    elseif value isa Number
        return string(value)
    elseif value isa AbstractString
        return value
    elseif value isa Vector
        if isempty(value)
            return "[]"
        elseif length(value) <= 3 && all(v -> v isa Number || v isa Bool, value)
            return "[" * join(value, ", ") * "]"
        elseif all(v -> v isa AbstractString, value)
            return "[" * join(value, ", ") * "]"
        else
            return string(length(value)) * " items"
        end
    elseif value isa Matrix
        rows = size(value, 1)
        cols = size(value, 2)
        return "Matrix{$(rows)×$(cols)}"
    elseif value isa AbstractDict
        return "{$(length(value)) keys}"
    elseif value isa Tuple
        if length(value) <= 3
            return "(" * join(value, ", ") * ")"
        else
            return "Tuple{$(length(value)) elements}"
        end
    else
        return string(value)
    end
end

"""
    print_dict_short(dict::AbstractDict, section_name::String, indent::Int=0)
    
Print dictionary contents in short format, automatically iterating over keys.
"""
function print_dict_short(dict::AbstractDict, section_name::String, indent::Int=2)
    isempty(dict) && return
    
    prefix = " "^indent
    
    for (key, value) in dict
        # Skip certain internal keys
        key in ("workspace",) && continue
        
        # Format the value
        val_str = format_value_short(value)
        
        # Handle nested dicts separately for readability
        if value isa AbstractDict && !isempty(value)
            println("$prefix$key:")
            print_dict_short(value, "", indent + 2)
        else
            println("$prefix$key: $val_str")
        end
    end
end

"""
    print_dict_long(dict::AbstractDict, section_name::String, indent::Int=0)
    
Print dictionary contents in long format, automatically iterating over keys.
"""
function print_dict_long(dict::AbstractDict, section_name::String, indent::Int=2)
    isempty(dict) && return
    
    prefix = " "^indent
    
    for (key, value) in dict
        # Skip certain internal keys
        key in ("workspace",) && continue
        
        # Convert underscores to spaces for readability
        display_key = replace(key, "_" => " ")
        
        # Try to infer a description based on the key  
        type_str = if value isa Bool
            "Bool"
        elseif value isa Number
            "$(typeof(value))"
        elseif value isa AbstractString
            "String"
        elseif value isa Vector
            "Vector"
        elseif value isa Matrix
            "Matrix"
        elseif value isa AbstractDict
            "Dict"
        else
            "$(typeof(value))"
        end
        
        # Format value
        val_str = format_value_long(value, indent + 2)
        
        # Print with type info
        println("$prefix$display_key :: $type_str")
        
        if val_str != string(value) && (value isa AbstractDict && !isempty(value))
            # If it's a non-empty dict, print recursively
            print_dict_long(value, "", indent + 2)
        else
            println("$prefix  → $val_str")
        end
    end
end

"""
    print_section_short(section::String, settings_dict::AbstractDict)
    
Print settings in short format using generic reflection-based iteration.
Automatically prints all fields in settings_dict without manual enumeration.
"""
function print_section_short(section::String, settings_dict::AbstractDict)
    should_print(s) = section == "all" || section == s
    
    for (section_name, section_data) in settings_dict
        if should_print(section_name)
            # Format section name: "general_settings" → "General Settings"
            formatted_name = replace(section_name, "_" => " ") |> titlecase
            println("\n[$formatted_name]")
            print_dict_short(section_data, section_name, 2)
        end
    end
end

"""
    print_section_long(section::String, settings_dict::AbstractDict)
    
Print settings in long format using generic reflection-based iteration.
Automatically prints all fields with type information without manual enumeration.
"""
function print_section_long(section::String, settings_dict::AbstractDict)
    should_print(s) = section == "all" || section == s
    
    for (section_name, section_data) in settings_dict
        if should_print(section_name)
            # Format section name: "general_settings" → "General Settings"
            formatted_name = replace(section_name, "_" => " ") |> titlecase
            println("\n[$formatted_name]")
            print_dict_long(section_data, section_name, 2)
        end
    end
end

"""
    load_settings(settings_path::String)::Settings

Load and initialize settings from a JSON file.

This is the primary entry point for loading ENEEGMA configuration files.
It handles:
- Loading JSON from file
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
settings = load_settings("experiments/my_settings.json")
```
"""
function load_settings(settings_path::String)::Settings
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

    # Build Settings object from loaded settings dict
    # The Settings constructors handle all default values internally
    s = Settings(settings)
    set_task_settings(s)  # Store settings in task-local storage and set verbosity

    vinfo("Settings loaded and initialized from: $settings_path", level=1)
    flush(stdout)
    return s
end