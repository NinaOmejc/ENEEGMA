###################
## PARAMETER BOUNDS POLICY LAYER
## 
## Modular, composable bounds policy system with backward compatibility.
## Separates two concepts:
## 1. Bound policy: where bounds come from (scaled_defaults, table_by_type, named_table_level, unbounded)
## 2. Bound level: how permissive the bounds are (conservative, recommended, exploratory)
##
## Supported bound policies:
## - "scaled_defaults": Use model defaults and scale around them
## - "table_by_type": Load bounds from CSV with explicit lower/upper columns
## - "named_table_level": Load bounds from CSV with named level columns (e.g., conservative_lower/upper)
## - "unbounded": Keep bounds at ±Inf (for reparameterization-based optimization)
##
## Supported bound levels:
## - "conservative", "recommended", "exploratory"
## - Legacy aliases: "low" → "conservative", "medium" → "recommended", 
##                   "high" → "exploratory", "ultra" → "exploratory" (with warning)
###################

"""
    BoundPolicySpec

Specification for a parameter bounding policy.

# Fields
- `policy::String`: Policy type ("scaled_defaults", "named_table_level", "unbounded")
- `level::String`: Permissiveness level ("conservative", "recommended", "exploratory")
- `table_path::Union{String, Nothing}`: Path to bounds table (for named_table_level policy)

Supported policies:
- `"scaled_defaults"`: Bounds = model defaults * multipliers based on level
- `"named_table_level"`: Bounds from CSV with level-named columns ({level}_lower, {level}_upper)
- `"unbounded"`: All bounds set to ±Inf for reparameterization-only optimization
"""
mutable struct BoundPolicySpec
    policy::String
    level::String
    table_path::Union{String, Nothing}

    function BoundPolicySpec(;
        policy::String = "scaled_defaults",
        level::String = "recommended",
        table_path::Union{String, Nothing} = nothing
    )
        new(policy, level, table_path)
    end
end

"""
    normalize_bound_level(level::Union{String, Symbol})::String

Normalize bound level to canonical form with support for legacy aliases.

Mapping:
- "low" → "conservative"
- "medium" → "recommended"
- "high" → "exploratory"
- "ultra" → "exploratory" (emits deprecation warning)
- Other values returned as-is (assumed canonical)

# Arguments
- `level::Union{String, Symbol}`: Input level string or symbol

# Returns
- `String`: Canonical level name
"""
function normalize_bound_level(level::Union{String, Symbol})::String
    lev_str = lowercase(strip(String(level)))
    
    # Mapping of aliases to canonical names
    mapping = Dict(
        "low" => "conservative",
        "medium" => "recommended",
        "high" => "exploratory",
    )
    
    if haskey(mapping, lev_str)
        return mapping[lev_str]
    end
    
    # Handle deprecated "ultra" → "exploratory"
    if lev_str == "ultra"
        vwarn("bound_level='ultra' is deprecated; mapping to 'exploratory'"; level=1)
        return "exploratory"
    end
    
    # Assume canonical (conservative, recommended, exploratory, etc.)
    return lev_str
end

"""
    normalize_bound_policy(policy::Union{String, Symbol})::String

Validate and normalize bound policy name.

Valid policies:
- `"scaled_defaults"`: Multiplier-based bounds around model defaults
- `"named_table_level"`: Bounds from CSV with level-named columns
- `"unbounded"`: All bounds set to ±Inf

NOTE: "table_by_type" is deprecated; use "named_table_level" instead.

# Arguments
- `policy::Union{String, Symbol}`: Policy name

# Returns
- `String`: Normalized policy name

# Raises
- `ArgumentError` if policy is not recognized
"""
function normalize_bound_policy(policy::Union{String, Symbol})::String
    valid_policies = Set(["scaled_defaults", "named_table_level", "unbounded"])
    pol_str = lowercase(strip(String(policy)))
    
    # Handle deprecated "table_by_type" → "named_table_level"
    if pol_str == "table_by_type"
        vwarn(
            "bound_policy='table_by_type' is deprecated. " *
            "Use 'named_table_level' instead (CSV with {level}_lower and {level}_upper columns).",
            level=1
        )
        return "named_table_level"
    end
    
    if pol_str ∉ valid_policies
        throw(ArgumentError(
            "Invalid bound_policy='$pol_str'. Valid options: $(join(valid_policies, ", "))"
        ))
    end
    return pol_str
end

"""
    resolve_bound_policy(settings::Settings)::BoundPolicySpec

Resolve bound policy from settings with backward compatibility.

Handles migration from old naming (param_bound_scaling_level, empirical_bounds_table_path, etc.)
to new naming (bound_policy, bound_level, bounds_table_path, etc.).

Supported policies (3 total):
- `"scaled_defaults"`: Multiplier-based bounds around model defaults
- `"named_table_level"`: Bounds from CSV with level-named columns (recommended)
- `"unbounded"`: All bounds set to ±Inf

NOTE: "table_by_type" is deprecated and maps to "named_table_level".

Priority:
1. If new fields exist and bound_policy is set, use them.
2. Otherwise, migrate from param_bound_scaling_level and related old fields.
3. Default to conservative scaled_defaults if nothing is set.

# Arguments
- `settings::Settings`: Settings object with optimization_settings

# Returns
- `BoundPolicySpec`: Resolved policy specification

# Examples
```julia
policy = resolve_bound_policy(settings)
apply_bound_policy!(net.params, settings)
```
"""
function resolve_bound_policy(settings::Settings)::BoundPolicySpec
    os = settings.optimization_settings
    if os === nothing
        return BoundPolicySpec()
    end
    
    # Initialize variables with defaults
    policy = "scaled_defaults"
    level = "recommended"
    table_path = nothing
    
    # Check if bounds_task is present in old configs (deprecated)
    if hasproperty(os, :bounds_task) && os.bounds_task !== nothing && os.bounds_task != "" && os.bounds_task != "global"
        vwarn(
            "bounds_task='$(os.bounds_task)' is deprecated and ignored. " *
            "Parameter bounds are now always global by type only (task-independent).",
            level=1
        )
    end
    
    # Check if new fields are set (prefer new naming if explicitly set and non-empty)
    has_new_policy = hasproperty(os, :bound_policy) && 
                     os.bound_policy !== nothing && 
                     os.bound_policy != ""
    
    if has_new_policy
        # Use new naming
        policy = os.bound_policy
        level = hasproperty(os, :bound_level) ? os.bound_level : "recommended"
        table_path = hasproperty(os, :bounds_table_path) ? os.bounds_table_path : nothing
    else
        # Migrate from old naming (param_bound_scaling_level)
        old_level = hasproperty(os, :param_bound_scaling_level) ? 
                    lowercase(os.param_bound_scaling_level) : "medium"
        
        # Map old levels to new policy+level
        if old_level in ["low", "medium", "high", "ultra"]
            # Multiplier-based scaling
            policy = "scaled_defaults"
            level = normalize_bound_level(old_level)
        elseif old_level in ["conservative", "recommended", "exploratory"]
            # Named table level
            policy = "named_table_level"
            level = old_level
            table_path = hasproperty(os, :empirical_bounds_table_path) ? 
                         os.empirical_bounds_table_path : nothing
        elseif old_level == "empirical"
            # Old empirical → named_table_level with default table
            policy = "named_table_level"
            level = "recommended"
            table_path = hasproperty(os, :empirical_bounds_table_path) ? 
                         os.empirical_bounds_table_path : 
                         joinpath("grammars", "recommended_neural_mass_parameter_bounds_three_levels.csv")
        elseif old_level == "unbounded"
            policy = "unbounded"
            level = "recommended"  # Ignored
        else
            # Fallback
            vwarn("Unknown param_bound_scaling_level='$old_level', using default"; level=1)
            policy = "scaled_defaults"
            level = "recommended"
        end
    end
    
    # Validate
    policy = normalize_bound_policy(policy)
    level = normalize_bound_level(level)
    
    return BoundPolicySpec(;
        policy = policy,
        level = level,
        table_path = table_path
    )
end

"""
    load_bounds_table(path::String)::DataFrame

Load and validate a bounds table from CSV.

Expected columns:
- For table_by_type: "type", lower_column, upper_column
- For named_table_level: "type", "conservative_lower", "conservative_upper", 
                         "recommended_lower", "recommended_upper",
                         "exploratory_lower", "exploratory_upper"

NOTE: If a "task" column is detected in the table, a deprecation warning is emitted
(task-dependent bounds are no longer supported).

# Arguments
- `path::String`: Path to CSV file (can be relative to package root)

# Returns
- `DataFrame`: Loaded table

# Raises
- `ArgumentError` if file not found or table is invalid
"""
function load_bounds_table(path::String)::DataFrame
    # Resolve package-relative paths
    resolved_path = if isfile(path)
        path
    else
        pkg_path = joinpath(pkgdir(ENEEGMA), path)
        if isfile(pkg_path)
            pkg_path
        else
            error("Bounds table not found: '$path' (also checked: '$pkg_path')")
        end
    end
    
    try
        table = CSV.read(resolved_path, DataFrame)
        
        # Basic validation
        if nrow(table) == 0
            error("Bounds table is empty: '$resolved_path'")
        end
        
        # Check for deprecated task column
        if "task" in names(table)
            vwarn(
                "Bounds table contains deprecated 'task' column. " *
                "Task-dependent bounds are no longer supported; task column will be ignored.",
                level=1
            )
        end
        
        vinfo("Loaded bounds table: $resolved_path ($(nrow(table)) rows)"; level=2)
        return table
    catch e
        throw(ArgumentError("Failed to load bounds table from '$resolved_path': $e"))
    end
end

"""
    apply_scaled_default_bounds!(paramset::ParamSet, policy::BoundPolicySpec)::ParamSet

Apply multiplier-based bounds around parameter defaults.

Uses PARAM_RANGE_MULTIPLIERS and ZERO_DEFAULT_SPANS constants based on policy.level.

Mapping:
- "conservative" → (0.5, 2.0)   multipliers, 1.0 zero_span
- "recommended"  → (0.25, 4.0)  multipliers, 5.0 zero_span
- "exploratory"  → (0.125, 8.0) multipliers, 10.0 zero_span

# Arguments
- `paramset::ParamSet`: Parameters to update
- `policy::BoundPolicySpec`: Policy specification (level field used)

# Returns
- Updated ParamSet
"""
function apply_scaled_default_bounds!(paramset::ParamSet, policy::BoundPolicySpec)::ParamSet
    level = policy.level
    
    haskey(PARAM_RANGE_MULTIPLIERS, level) || error(
        "Unknown bound_level='$level' for scaled_defaults policy. Valid: $(keys(PARAM_RANGE_MULTIPLIERS))"
    )
    
    lower_mult, upper_mult = PARAM_RANGE_MULTIPLIERS[level]
    zero_span = ZERO_DEFAULT_SPANS[level]
    
    for param in paramset.params
        if is_verbose(2)
            vinfo("Current param: $(param.name) | default: $(param.default) | min: $(param.min) | max: $(param.max)"; level=2)
        end
        _apply_multiplier_range!(param, lower_mult, upper_mult, zero_span)
        if is_verbose(2)
            vinfo("Updated param: $(param.name) | type: $(param.type) | min: $(param.min) | max: $(param.max)"; level=2)
        end
    end
    
    return paramset
end

"""
    apply_table_bounds_by_type!(paramset::ParamSet, table::DataFrame, 
                                 policy::BoundPolicySpec)::ParamSet

DEPRECATED: Apply bounds from a CSV table using explicit lower/upper columns.

NOTE: This function is deprecated. Use `apply_named_table_level_bounds!()` instead with
a table containing level-named columns (e.g., conservative_lower, conservative_upper).

The table should have columns:
- "type": Parameter type (e.g., "frequency", "damping", "gain")
- policy.lower_column: Lower bound values
- policy.upper_column: Upper bound values

Global bounds by type only: If multiple rows exist for the same parameter type,
the bounds are collapsed globally:
- Lower bounds use the minimum across all rows for that type
- Upper bounds use the maximum across all rows for that type

Missing types default to keeping existing bounds with a warning.

# Arguments
- `paramset::ParamSet`: Parameters to update
- `table::DataFrame`: Bounds table
- `policy::BoundPolicySpec`: Policy (lower_column, upper_column fields used)

# Returns
- Updated ParamSet
"""
function apply_table_bounds_by_type!(paramset::ParamSet, table::DataFrame, 
                                     policy::BoundPolicySpec)::ParamSet
    vwarn(
        "apply_table_bounds_by_type!() is deprecated. " *
        "Use apply_named_table_level_bounds!() with a table containing level-named columns instead.",
        level=1
    )
    
    # This function is kept for backward compatibility but should not be called
    # in normal operation. If called, it will use hardcoded column names.
    lb_col = Symbol("lower_bound")
    ub_col = Symbol("upper_bound")
    
    # Validate columns exist
    "type" in names(table) || error("Bounds table missing 'type' column")
    string(lb_col) in names(table) || error("Bounds table missing lower bound column '$(lb_col)'")
    string(ub_col) in names(table) || error("Bounds table missing upper bound column '$(ub_col)'")
    
    # Build lookup dict with global bounds (min/max across all rows for each type)
    bounds_dict = Dict{Symbol, Tuple{Float64, Float64}}()
    duplicate_types = Set{Symbol}()
    
    for row in eachrow(table)
        type_str = String(row["type"])
        ty = _normalize_param_type(type_str)
        try
            lb = Float64(row[lb_col])
            ub = Float64(row[ub_col])
            
            if haskey(bounds_dict, ty)
                # Duplicate type: collapse to global bounds (min/max)
                push!(duplicate_types, ty)
                existing_lb, existing_ub = bounds_dict[ty]
                bounds_dict[ty] = (min(existing_lb, lb), max(existing_ub, ub))
            else
                bounds_dict[ty] = (lb, ub)
            end
        catch e
            vwarn("Skipping row with invalid bounds for type '$type_str': $e"; level=2)
        end
    end
    
    # Warn about duplicate types that were collapsed
    if !isempty(duplicate_types)
        vwarn(
            "Bounds table contains duplicate type rows for: $(join(duplicate_types, ", ")). " *
            "Collapsed to global bounds (min of lowers, max of uppers).",
            level=2
        )
    end
    
    # Apply to parameters
    for param in paramset.params
        ty = _normalize_param_type(param.type)
        
        if !haskey(bounds_dict, ty)
            # Try :unknown as fallback
            if ty != :unknown && haskey(bounds_dict, :unknown)
                ty = :unknown
            else
                vwarn("No bounds found for parameter type '$(param.type)' (param: $(param.name))"; level=2)
                continue
            end
        end
        
        lb, ub = bounds_dict[ty]
        lb, ub = _sanitize_bounds(param, lb, ub)
        param.min = lb
        param.max = ub
    end
    
    return paramset
end

"""
    apply_named_table_level_bounds!(paramset::ParamSet, table::DataFrame, 
                                     policy::BoundPolicySpec)::ParamSet

Apply bounds from a CSV table using named level columns.

Table should have columns like:
- "type"
- "conservative_lower", "conservative_upper"
- "recommended_lower", "recommended_upper"
- "exploratory_lower", "exploratory_upper"

Global bounds by type only: If multiple rows exist for the same parameter type,
the bounds are collapsed globally:
- Lower bounds use the minimum across all rows for that type
- Upper bounds use the maximum across all rows for that type

The level (policy.level) determines which pair of columns to use.

# Arguments
- `paramset::ParamSet`: Parameters to update
- `table::DataFrame`: Bounds table with named level columns
- `policy::BoundPolicySpec`: Policy (level field used)

# Returns
- Updated ParamSet
"""
function apply_named_table_level_bounds!(paramset::ParamSet, table::DataFrame,
                                         policy::BoundPolicySpec)::ParamSet
    level = policy.level
    lb_col = Symbol("$(level)_lower")
    ub_col = Symbol("$(level)_upper")
    
    # Validate columns exist
    "type" in names(table) || error("Bounds table missing 'type' column")
    string(lb_col) in names(table) || error("Bounds table missing level column '$(lb_col)' for level='$level'")
    string(ub_col) in names(table) || error("Bounds table missing level column '$(ub_col)' for level='$level'")
    
    # Build lookup dict with global bounds (min/max across all rows for each type)
    bounds_dict = Dict{Symbol, Tuple{Float64, Float64}}()
    duplicate_types = Set{Symbol}()
    
    for row in eachrow(table)
        type_str = String(row["type"])
        ty = _normalize_param_type(type_str)
        try
            lb = Float64(row[lb_col])
            ub = Float64(row[ub_col])
            
            if haskey(bounds_dict, ty)
                # Duplicate type: collapse to global bounds (min/max)
                push!(duplicate_types, ty)
                existing_lb, existing_ub = bounds_dict[ty]
                bounds_dict[ty] = (min(existing_lb, lb), max(existing_ub, ub))
            else
                bounds_dict[ty] = (lb, ub)
            end
        catch e
            vwarn("Skipping row with invalid bounds for type '$type_str': $e"; level=2)
        end
    end
    
    # Warn about duplicate types that were collapsed
    if !isempty(duplicate_types)
        vwarn(
            "Bounds table contains duplicate type rows for: $(join(duplicate_types, ", ")). " *
            "Collapsed to global bounds (min of lowers, max of uppers).",
            level=2
        )
    end
    
    # Apply to parameters
    for param in paramset.params
        ty = _normalize_param_type(param.type)
        
        if !haskey(bounds_dict, ty)
            # Try :unknown as fallback
            if ty != :unknown && haskey(bounds_dict, :unknown)
                ty = :unknown
            else
                vwarn("No bounds found for parameter type '$(param.type)' (param: $(param.name))"; level=2)
                continue
            end
        end
        
        lb, ub = bounds_dict[ty]
        lb, ub = _sanitize_bounds(param, lb, ub)
        param.min = lb
        param.max = ub
    end
    
    return paramset
end

"""
    apply_unbounded_bounds!(paramset::ParamSet, policy::BoundPolicySpec)::ParamSet

Set all bounds to ±Inf (unbounded).

Useful when relying entirely on reparameterization transforms for constrained optimization.

# Arguments
- `paramset::ParamSet`: Parameters to update
- `policy::BoundPolicySpec`: Policy (unused, included for consistency)

# Returns
- Updated ParamSet
"""
function apply_unbounded_bounds!(paramset::ParamSet, policy::BoundPolicySpec)::ParamSet
    for param in paramset.params
        if is_verbose(2)
            vinfo("Current param: $(param.name) | default: $(param.default) | min: $(param.min) | max: $(param.max)"; level=2)
        end
        param.min = -Inf
        param.max = Inf
        if is_verbose(2)
            vinfo("Updated param: $(param.name) | type: $(param.type) | unbounded"; level=2)
        end
    end
    return paramset
end

"""
    _sanitize_bounds(param::Param, lb::Float64, ub::Float64)::Tuple{Float64, Float64}

Sanitize parameter bounds based on parameter type.

Rules:
- probability: intersect with [0, 1]
- positive-only types (frequency, time_constant, rate, damping, gain, noise_std, tscale):
  enforce lower_bound >= 1e-9
- if ub <= lb: error (should not happen with well-formed table)
- missing type: warn and keep as-is

# Arguments
- `param::Param`: Parameter (for type information)
- `lb::Float64`: Lower bound
- `ub::Float64`: Upper bound

# Returns
- `(lb, ub)::Tuple{Float64, Float64}`: Sanitized bounds
"""
function _sanitize_bounds(param::Param, lb::Float64, ub::Float64)::Tuple{Float64, Float64}
    # Probability: intersect with [0, 1]
    if param.type == :probability
        lb = max(lb, 0.0)
        ub = min(ub, 1.0)
    end
    
    # Positive-only types: enforce minimum >= 1e-9
    if param.type in POSITIVE_ONLY_PARAM_TYPES
        lb = max(lb, 1e-9)
        ub = max(ub, lb + 1e-9)
    end
    
    # Validity check
    if ub <= lb
        error(
            "Invalid bounds for parameter '$(param.name)': " *
            "lower=$lb >= upper=$ub"
        )
    end
    
    return (lb, ub)
end

"""
    apply_bound_policy!(paramset::ParamSet, settings::Settings)::ParamSet

Apply parameter bounds policy based on settings.

Dispatches to specific policy function (scaled_defaults, named_table_level, unbounded).

This is the main entry point for applying bounds policies.

# Arguments
- `paramset::ParamSet`: Parameters to update
- `settings::Settings`: Settings with optimization_settings containing policy config

# Returns
- Updated ParamSet
"""
function apply_bound_policy!(paramset::ParamSet, settings::Settings)::ParamSet
    policy = resolve_bound_policy(settings)
    
    vinfo("Applying bound policy: policy=$(policy.policy), level=$(policy.level)"; level=1)
    
    if policy.policy == "scaled_defaults"
        return apply_scaled_default_bounds!(paramset, policy)
    
    elseif policy.policy == "named_table_level"
        table_path = policy.table_path
        table_path === nothing && error(
            "Policy 'named_table_level' requires bounds_table_path in settings"
        )
        table = load_bounds_table(table_path)
        return apply_named_table_level_bounds!(paramset, table, policy)
    
    elseif policy.policy == "unbounded"
        return apply_unbounded_bounds!(paramset, policy)
    
    else
        error("Unknown bound_policy: $(policy.policy)")
    end
end
