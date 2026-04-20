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
    check_settings(settings::Settings)::Bool

Validate all settings for consistency and correctness.

This function is independent from settings construction, allowing you to validate
settings at any point in the workflow - after loading, after programmatic modifications,
or before building/optimizing. Does not modify settings, only checks them.

Validates:
- Network settings:
  * n_nodes must be > 0
  * node_names, node_models, node_coords lengths match n_nodes
  * network_conn, network_conn_funcs, network_delay are n_nodes × n_nodes
  * sensory_input_conn length matches n_nodes
  * Multi-node networks have inter-node connections (warns if not)
- Simulation settings (if present):
  * tspan: start < end
  * saveat: positive and ≤ total simulation time
  * solver: specified (uses default if missing)
- Optimization settings (always present):
  * method: only CMAES currently supported
  * frequency range: fmin < fmax
  * hyperparameter sweep: all sweep axes have values

# Arguments
- `settings::Settings`: Settings object to validate

# Returns
`true` if all validations pass

# Throws
- `ArgumentError`: If any validation fails with detailed error message

# Example
```julia
settings = load_settings("config.json")
settings.network_settings.n_nodes = 3  # User modification
check_settings(settings)  # Validate before building
model = build_network(settings)

# Or validate before optimizing
optimize_network(net, data, settings)  # Calls check_settings() automatically
```

# Notes
- OptimizationSettings is always created by Settings constructor (never None)
- SimulationSettings can be None for workflows that don't simulate
- Validation is non-destructive; settings are never modified
"""
function check_settings(settings::Settings)::Bool
    ns = settings.network_settings
    n_nodes = ns.n_nodes
    
    # Validate n_nodes
    n_nodes > 0 || throw(ArgumentError("n_nodes must be > 0, got $n_nodes"))
    
    # Validate node names match n_nodes
    length(ns.node_names) == n_nodes || throw(ArgumentError(
        "node_names length ($(length(ns.node_names))) must match n_nodes ($n_nodes)"
    ))
    
    # Validate node models match n_nodes
    length(ns.node_models) == n_nodes || throw(ArgumentError(
        "node_models length ($(length(ns.node_models))) must match n_nodes ($n_nodes)"
    ))
    
    # Validate node coordinates match n_nodes
    length(ns.node_coords) == n_nodes || throw(ArgumentError(
        "node_coords length ($(length(ns.node_coords))) must match n_nodes ($n_nodes)"
    ))
    
    # Validate connectivity matrix dimensions
    size(ns.network_conn) == (n_nodes, n_nodes) || throw(ArgumentError(
        "network_conn must be ($n_nodes × $n_nodes), got $(size(ns.network_conn))"
    ))
    
    # Validate connection functions matrix dimensions
    size(ns.network_conn_funcs) == (n_nodes, n_nodes) || throw(ArgumentError(
        "network_conn_funcs must be ($n_nodes × $n_nodes), got $(size(ns.network_conn_funcs))"
    ))
    
    # Validate delay matrix dimensions
    size(ns.network_delay) == (n_nodes, n_nodes) || throw(ArgumentError(
        "network_delay must be ($n_nodes × $n_nodes), got $(size(ns.network_delay))"
    ))
    
    # Validate sensory input connectivity length
    length(ns.sensory_input_conn) == n_nodes || throw(ArgumentError(
        "sensory_input_conn length ($(length(ns.sensory_input_conn))) must match n_nodes ($n_nodes)"
    ))
    
    # Validate multi-node networks have inter-node connections
    if n_nodes > 1
        has_zero_conn = all(ns.network_conn[i,j] == 0.0 for i=1:n_nodes for j=1:n_nodes if i != j)
        if has_zero_conn
            vwarn("Multi-node network has no inter-node connections (network_conn matrix is all zeros). Nodes will evolve independently."; level=1)
        end
    end
    
    # ========== SIMULATION SETTINGS VALIDATION ==========
    ss = settings.simulation_settings
    if !isnothing(ss)
        # Validate time span
        ss.tspan[1] < ss.tspan[2] || throw(ArgumentError(
            "Simulation tspan invalid: start ($(ss.tspan[1])) must be < end ($(ss.tspan[2]))"
        ))
        
        # Validate saveat is positive and less than tspan
        ss.saveat > 0.0 || throw(ArgumentError("saveat must be > 0, got $(ss.saveat)"))
        ss.saveat <= (ss.tspan[2] - ss.tspan[1]) || vwarn(
            "saveat ($(ss.saveat)) is larger than tspan ($(ss.tspan[2] - ss.tspan[1]))"; level=2
        )
        
        # Validate solver is specified
        isempty(ss.solver) && vwarn("solver not specified in simulation settings, will use default"; level=2)
    end
    
    # ========== OPTIMIZATION SETTINGS VALIDATION ==========
    os = settings.optimization_settings
    # Note: OptimizationSettings is ALWAYS created in Settings constructor, so no None check needed
    
    # Validate method
    os.method == "CMAES" || throw(ArgumentError(
        "Optimization method '$(os.method)' is not supported. Only 'CMAES' is currently available."
    ))
    
    # Validate frequency range
    os.loss_settings.fmin >= os.loss_settings.fmax && throw(ArgumentError(
        "fmin ($(os.loss_settings.fmin)) must be < fmax ($(os.loss_settings.fmax))"
    ))
    
    # Validate hyperparameter sweep if present
    hs = os.hyperparameter_sweep
    if !isempty(hs.hyperparameters)
        for (key, values) in hs.hyperparameters
            isempty(values) && throw(ArgumentError(
                "Hyperparameter '$key' has no values to sweep over"
            ))
        end
        vinfo("Hyperparameter sweep configured with $(length(hs.hyperparameters)) axes"; level=2)
    end
    
    # ========== DATA SETTINGS VALIDATION ==========
    ds = settings.data_settings
    if !isnothing(ds) && ds.target_channel isa Dict
        # Multi-node: validate that all dict keys exist in node_names
        dict_keys = Set(keys(ds.target_channel))
        node_names_set = Set(ns.node_names)
        
        missing_nodes = setdiff(dict_keys, node_names_set)
        !isempty(missing_nodes) && throw(ArgumentError(
            "target_channel dict contains keys that don't match node_names. " *
            "Extra keys: $(collect(missing_nodes)). Available node_names: $(ns.node_names)"
        ))
        
        # Warn if some nodes don't have data channels specified
        missing_data = setdiff(node_names_set, dict_keys)
        !isempty(missing_data) && vwarn(
            "Nodes without target_channel mapping: $(collect(missing_data)). " *
            "These nodes will not load data during prepare_data!()."; level=2
        )
        
        vinfo("Data settings validated: $(length(dict_keys)) node(s) mapped to data channel(s)"; level=2)
    end
    
    vinfo("All settings validated successfully."; level=2)
    return true
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
    
Format a value for short display (single line or multi-line for matrices/coordinates).
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
    elseif value isa Matrix
        return "MATRIX_MULTILINE"  # Signal for multi-line handling
    elseif value isa Vector
        if isempty(value)
            return "[]"
        elseif length(value) == 1
            # Single-item vector: print the item itself
            return string(value[1])
        elseif all(v -> v isa Number || v isa Bool || v isa Nothing, value)
            # Simple vector of numbers/bools
            return join(value, ", ")
        elseif all(v -> v isa AbstractString, value)
            # Vector of strings
            return join(value, ", ")
        elseif all(v -> v isa Tuple && length(v) == 3, value)
            # Vector of 3-tuples (like node coordinates) - signal for multi-line
            return "COORDINATES_MULTILINE"
        else
            # Complex vector with multiple items
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
    format_matrix_multiline(matrix::AbstractMatrix, indent::Int=4)::String

Format a matrix with each row on a separate line for readability.
All rows are indented consistently with opening bracket on first row.
Subsequent rows aligned to start column with a single space indent.
Empty strings are displayed as "" for clarity.
"""
function format_matrix_multiline(matrix::AbstractMatrix, indent::Int=4)::String
    if isempty(matrix)
        return "[]"
    end
    
    rows = size(matrix, 1)
    cols = size(matrix, 2)
    
    # Calculate field width for alignment
    max_width = 0
    formatted_vals = Matrix{String}(undef, rows, cols)
    for i in 1:rows
        for j in 1:cols
            val = matrix[i, j]
            # Display empty strings as "" for clarity
            if val isa String && isempty(val)
                formatted_vals[i,j] = "\"\""
            else
                formatted_vals[i,j] = string(val)
            end
            max_width = max(max_width, length(formatted_vals[i, j]))
        end
    end
    
    # Build formatted rows with alignment
    prefix = " "^indent
    lines = String[]
    
    for i in 1:rows
        row_parts = String[]
        for j in 1:cols
            val_str = formatted_vals[i, j]
            # Right-align values
            padded = lpad(val_str, max_width)
            push!(row_parts, padded)
        end
        
        # Construct row with proper brackets
        if i == 1
            # First row: opening bracket + values with full indent
            row_str = prefix * "[" * join(row_parts, " ")
        else
            # Subsequent rows: values aligned to bracket column (single space)
            row_str = prefix * " " * join(row_parts, " ")
        end
        
        # Add row separator or closing bracket
        if i < rows
            row_str *= ";"
        else
            row_str *= "]"
        end
        
        push!(lines, row_str)
    end
    
    return join(lines, "\n")
end

"""
    format_coordinates_multiline(coords::Vector{Tuple}, indent::Int=4)::String

Format a vector of coordinate tuples with each on a separate line in matrix notation.
First row has opening bracket, subsequent rows aligned with single space indent.
"""
function format_coordinates_multiline(coords::Vector{<:Tuple}, indent::Int=4)::String
    if isempty(coords)
        return "[]"
    end
    
    prefix = " "^indent
    lines = String[]
    
    for (i, coord) in enumerate(coords)
        # Format coordinate values with space separation
        coord_vals = join(coord, " ")
        
        if i == 1
            # First row: start bracket with full indent
            push!(lines, prefix * "[" * coord_vals * ";")
        elseif i < length(coords)
            # Middle rows: single space indent for alignment
            push!(lines, prefix * " " * coord_vals * ";")
        else
            # Last row: single space indent with closing bracket
            push!(lines, prefix * " " * coord_vals * "]")
        end
    end
    
    return join(lines, "\n")
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
Handles multi-line formatting for matrices and coordinate vectors.
"""
function print_dict_short(dict::AbstractDict, section_name::String, indent::Int=2)
    isempty(dict) && return
    
    prefix = " "^indent
    
    for (key, value) in dict
        # Skip certain internal keys
        key in ("workspace",) && continue
        
        # Handle nested dicts separately for readability
        if value isa AbstractDict && !isempty(value)
            println("$prefix$key:")
            print_dict_short(value, "", indent + 2)
        elseif value isa Matrix
            # Format matrix with rows on separate lines
            println("$prefix$key:")
            matrix_str = format_matrix_multiline(value, indent + 2)
            println(matrix_str)
        elseif value isa Vector && all(v -> v isa Tuple && length(v) == 3, value) && !isempty(value)
            # Vector of 3-tuples (coordinates) - multi-line
            println("$prefix$key:")
            coords_str = format_coordinates_multiline(value, indent + 2)
            println(coords_str)
        else
            # Format single-line value
            val_str = format_value_short(value)
            
            # Skip MULTILINE signals that weren't caught above
            if val_str in ("MATRIX_MULTILINE", "COORDINATES_MULTILINE")
                val_str = string(value)
            end
            
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

    # Build Settings object from loaded settings dict
    # The Settings constructors handle all default values internally
    s = Settings(settings)
    set_task_settings(s)  # Store settings in task-local storage and set verbosity

    vinfo("Settings loaded and initialized from: $settings_path", level=1)
    flush(stdout)
    return s
end