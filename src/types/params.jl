###################
## PARAMETERS
###################

const VALID_PARAM_TYPES = Set{Symbol}([
    :frequency,
    :time_constant,
    :rate,
    :damping,
    :gain,
    :node_coupling,
    :population_coupling,
    :sensory_coupling,
    :noise_std,
    :probability,
    :offset,    
    :potential,    
    :poly_coeff,
    :tscale,
    :unknown,
])


"Groups used to share transforms + bounds policies."
const PARAM_TYPE_GROUP = Dict{Symbol,Symbol}(
    # Group A: positive time-like / rate-like
    :frequency       => :pos_time,
    :time_constant   => :pos_time,
    :rate            => :pos_time,
    :damping         => :pos_time,
    :tscale          => :pos_time,

    # Group B: positive amplitudes
    :gain            => :pos_amp,
    :noise_std       => :pos_amp,

    # Group C: signed weights/coefficients
    :population_coupling => :signed_weight,
    :node_coupling       => :signed_weight,
    :sensory_coupling    => :signed_weight,
    :poly_coeff          => :signed_weight,

    # Group D: signed offsets/potentials
    :offset         => :signed_offset,
    :potential      => :signed_offset,

    # Group E: [0,1]
    :probability    => :unit_interval,

    # Fallback
    :unknown        => :signed_weight,
)

const DEFAULT_WIDEN_FRAC = 0.00

const TYPE_WIDEN_FRAC = Dict{Symbol,Float64}(
    :time_constant => 0.00,
    :frequency     => 0.00,
    :rate          => 0.00,
    :damping       => 0.00,

    :gain          => 0.00,
    :offset        => 0.00,
    :potential     => 0.00,

    :population_coupling => 0.00,
    :poly_coeff          => 0.00,

    :noise_std     => 0.00,
    :probability   => 0.00,
)


const MIN_PARAM_VAL = -Inf
const MAX_PARAM_VAL = Inf

const POSITIVE_ONLY_PARAM_TYPES = Set{Symbol}([
    :frequency,
    :time_constant,
    :damping,
    :noise_std,
    :tscale,
    :gain,
    :rate,
])

const TIME_RELATED_PARAM_TYPES = Set{Symbol}([
    :frequency,
    :time_constant,
    :rate,
    :damping,
])

const WEIGHT_PARAM_TYPES = Set{Symbol}([
    :node_coupling,
    :population_coupling,
    :sensory_coupling,
    :poly_coeff,
    :unknown
])

const PARAM_RANGE_MULTIPLIERS = Dict(
    "low" => (0.5, 2.0),
    "medium" => (0.25, 4.0),
    "high" => (0.125, 8.0),
    "ultra" => (0.0625, 16.0)
)

const ZERO_DEFAULT_SPANS = Dict(
    "low" => 1.0,
    "medium" => 5.0,
    "high" => 10.0,
    "ultra" => 25.0
)

function _normalize_param_type(param_type::Union{String, Symbol})::Symbol
    raw_sym = param_type isa Symbol ? param_type : Symbol(param_type)
    lowered = Symbol(lowercase(String(raw_sym)))

    # Backward-compatible aliases:
    normalized =
        lowered == :coupling ? :population_coupling :
        lowered == :node_coupling ? :population_coupling :
        lowered == :sensory_coupling ? :population_coupling :
        lowered == :scale ? :unknown :  # legacy; should be retagged upstream
        lowered

    return normalized in VALID_PARAM_TYPES ? normalized : :unknown
end

mutable struct Param
    name::String
    symbol::Num
    type::Symbol
    default::Float64
    min::Float64
    max::Float64
    isdelay::Bool
    istimevarying::Bool
    tunable::Bool
    unit::String
    description::String
    parent_pop::AbstractPopulation

    # Constructor that creates the symbol from the name
    function Param(name::String, type::Union{String, Symbol}, parent_pop::AbstractPopulation;
                   default::Float64=1., min::Float64=MIN_PARAM_VAL, max::Float64=MAX_PARAM_VAL,
                   isdelay::Bool=false, istimevarying::Bool=false, tunable::Bool=true,
                   unit::String="", description::String="")
        # Create the symbol
        symbol = string2num(name)
        normalized_type = _normalize_param_type(type)
        new(name, symbol, normalized_type, default, min, max, isdelay, istimevarying, tunable,
            unit, description, parent_pop)
    end
end

mutable struct ParamSet
    params::Vector{Param}

    function ParamSet(params::Vector{Param})
        new(params)
    end
    
    # Add default constructor for empty ParamSet
    function ParamSet()
        new(Vector{Param}())
    end
end


# Helper to add a Param to a ParamSet
function add_param!(paramset::ParamSet, param::Param)
    if param_in_paramset(paramset, param)
        vwarn("Parameter $(param.name) already exists in ParamSet. Skipping.")
        return paramset
    end
    push!(paramset.params, param)
    return paramset
end



# Join multiple ParamSet objects into one
function join_paramsets(param_sets::Vector{ParamSet})::ParamSet
    all_params = Vector{Param}()
    
    for ps in param_sets
        append!(all_params, ps.params)
    end
    
    return ParamSet(all_params)
end

function join_paramsets!(paramset::ParamSet, param_sets::Vector{ParamSet})::ParamSet
    for ps in param_sets
        append!(paramset.params, ps.params)
    end
    return paramset
end


function get_param_by_name(paramset::ParamSet, name::String)
    for param in paramset.params
        if param.name == name
            return param
        end
    end
    error("Parameter '$name' not found in ParamSet.")
end

function get_param_by_symbol(paramset::ParamSet, param_symbol::Union{Num, Symbol})
    # if param_symbol is Symbol, convert to Num
    if eltype(param_symbol) == Symbol || param_symbol isa Symbol
        param_symbol = string2num(string(param_symbol))
    end
    for param in paramset.params
        if isequal(param.symbol, param_symbol)
            return param
        end
    end
    error("Parameter with symbol '$param_symbol' not found in ParamSet.")
end

function get_params_by_nodeid(paramset::ParamSet, node_id::Int)::ParamSet
    filtered_params = [p for p in paramset.params if p.parent_pop.parent_node.id == node_id]
    return ParamSet(filtered_params)
end


function get_tunable_params(paramset::ParamSet)::ParamSet
    tunable_params = [p for p in paramset.params if p.tunable]
    return ParamSet(tunable_params)
end


function get_postfix_index(param::Param)
    m = match(r"\d+$", param.name)
    if m === nothing
        error("No postfix index found in parameter name: $(param.name)")
    end
    return parse(Int, m.match)
end


function get_highest_postfix_index(paramset::ParamSet; node_id::Int=0, pop_id::Int=0, param_idx_only::Bool=true)
    if node_id != 0
        paramset = get_params_by_nodeid(paramset, node_id)
    end

    indices = Int[]
    for param in paramset.params
        push!(indices, get_postfix_index(param))
    end

    if pop_id != 0
        indices = filter(i -> parse(Int, string(i)[1]) == pop_id, indices)
    end

    if param_idx_only
        indices = [parse(Int, string(i)[2:end]) for i in indices]
    end

    return isempty(indices) ? 0 : maximum(indices)
end


function get_delay_params(paramset::ParamSet; return_type::String="named_tuple", sort::Bool=true)
    
    # Filter delay parameters
    delay_params = [p for p in paramset.params if p.isdelay]
    if isempty(delay_params)
        return nothing
    else
        return ParamSet(delay_params)
    end 
    #= # Extract names and values
    names = Symbol[p.name for p in delay_params]
    values = [p.default for p in delay_params]

    # Sort symbols if requested
    if sort
        sorted_indices = sortperm(names, by=s -> sort_symbols([s])[1]) # Sort based on sort_symbols
        names = names[sorted_indices]
        values = values[sorted_indices]
    end
    
    # Return based on the specified return_type
    if return_type == "named_tuple"
        return (; zip(names, values)...)
    elseif return_type == "vector"
        return values
    else
        throw(ArgumentError("Invalid return_type: $return_type. Use \"named_tuple\" or \"vector\"."))
    end =#
end


function get_symbols(paramset::ParamSet; sort::Bool=true, node_id::Int=0)
    
    # If node_id is specified, filter parameters by node_id
    if node_id != 0
        paramset = get_params_by_nodeid(paramset, node_id)
    end
    
    if sort
        return sort_symbols([p.symbol for p in paramset.params])
    else
        return [p.symbol for p in paramset.params]
    end
end


function get_param_default_values(paramset::ParamSet; p_subset::Vector{<:Union{Num, Symbol}}=Num[], 
    return_type::String="named_tuple", sort::Bool=true)
    
    param_symbols = get_symbols(paramset, sort=true)

    if !isempty(p_subset)
        # p_subset = [p for p in param_symbols if p in p_subset]
        if eltype(p_subset) == Symbol
            p_subset = [string2num(string(s)) for s in p_subset]
        end
        p_subset_filtered = Num[]
        for p_sym in param_symbols
            for p_target in p_subset
                if isequal(p_sym, p_target)
                    push!(p_subset_filtered, p_sym)
                    break
                end
            end
        end
        p_subset = p_subset_filtered
    else
        p_subset = param_symbols
    end

    p_names = Symbol[]
    p_values = Float64[]

    for ip in p_subset
        p = get_param_by_symbol(paramset, ip)

        # check if parameter already in p_names
        if Symbol(p.name) in p_names
            vwarn("Parameter $(p.name) already exists in names. Skipping.")
            continue
        end

        push!(p_names, Symbol(p.name))
        push!(p_values, Float64(p.default))
    end
    
    if return_type == "named_tuple"
        return (; zip(p_names, p_values)...)
    elseif return_type == "vector"
        return p_values
    elseif return_type == "dict"
        return Dict(p_names[i] => p_values[i] for i in eachindex(p_names))
    else
        error("Unknown return_type: $return_type. Use \"named_tuple\", \"vector\", or \"dict\".")
    end
end

function get_param_minmax_values(paramset::ParamSet; 
                                  p_subset::Vector{<:Union{Num, Symbol}}=Num[], 
                                  return_type::String="vector")
    # Collect parameter names and bounds
    p_names = Symbol[]
    min_values = Float64[]
    max_values = Float64[]

    if isempty(p_subset)
        # Use all params in the set
        for p in paramset.params
            # Avoid duplicates by name
            if Symbol(p.name) in p_names
                vwarn("Parameter $(p.name) already exists in names. Skipping.")
                continue
            end
            push!(p_names, Symbol(p.name))
            push!(min_values, Float64(p.min))
            push!(max_values, Float64(p.max))
        end
    else
        # Use only the requested subset (by symbol)
        for ip in p_subset
            p = get_param_by_symbol(paramset, ip)
            if Symbol(p.name) in p_names
                vwarn("Parameter $(p.name) already exists in names. Skipping.")
                continue
            end
            push!(p_names, Symbol(p.name))
            push!(min_values, Float64(p.min))
            push!(max_values, Float64(p.max))
        end
    end

    if return_type == "named_tuple"
        # NamedTuple of name => (min,max)
        return (; (p_names[i] => (min_values[i], max_values[i]) for i in eachindex(p_names))...)
    elseif return_type == "vector"
        # Tuple of vectors (lower_bounds, upper_bounds)
        return (min_values, max_values)
    elseif return_type == "dict"
        # Dict{Symbol,Tuple{Float64,Float64}}
        return Dict(p_names[i] => (min_values[i], max_values[i]) for i in eachindex(p_names))
    else
        error("Unknown return_type: $return_type. Use \"named_tuple\", \"vector\", or \"dict\".")
    end
end


function update_param_values!(paramset::ParamSet, new_values::AbstractDict{<:Any, <:Union{Real, AbstractVector{<:Real}, Tuple{Vararg{Real}}}})
    for (raw_name, values) in new_values
        param_name = raw_name isa AbstractString ? String(raw_name) : string(raw_name)
        try
            param = get_param_by_name(paramset, param_name)
            if values isa Number
                # Single numeric value: update default
                param.default = Float64(values)
            elseif values isa Tuple || values isa AbstractVector
                if length(values) == 1
                    # Only one value provided: update default
                    param.default = Float64(values[1])
                elseif length(values) == 3
                    # Three values provided: update default, min, and max
                    param.default = Float64(values[1])
                    param.min = Float64(values[2])
                    param.max = Float64(values[3])
                else
                    vwarn("Tuple for parameter '$(param_name)' has an unexpected number of values ($(length(values))). Skipping update.")
                end
            else
                vwarn("Unsupported value type for parameter '$(param_name)': $(typeof(values)). Skipping update.")
            end
        catch
            @warn "Param '$(param_name)' not found in ParamSet. Skipping update."
            continue
        end
    end
    return paramset
end


function update_param_tunability!(paramset::ParamSet, new_tunability::Union{Dict{String, Bool}, 
                                                                    OrderedDict{String, Bool}})
    for (name, tunable) in new_tunability
        try
            param = get_param_by_name(paramset, name)
            param.tunable = tunable
        catch e
            vwarn("Parameter '$name' not found in ParamSet. Skipping update.")
            continue
        end
    end
    return paramset
end


"""
    set_all_params_tunable!(paramset)

Force every parameter in `paramset` to be tunable. Useful when scripts need to
override per-model defaults before running custom optimization routines.
"""
function set_all_params_tunable!(paramset::ParamSet)::ParamSet
    for param in paramset.params
        param.tunable = true
    end
    return paramset
end



function _apply_type_based_range!(param::Param)::Bool
    grp = get(PARAM_TYPE_GROUP, param.type, :signed_weight)

    if grp == :pos_time || grp == :pos_amp
        param.min = 0.0
        param.max = MAX_PARAM_VAL
        return true
    elseif grp == :unit_interval
        param.min = 0.0
        param.max = 1.0
        return true
    else
        # signed_weight, signed_offset, unknown
        param.min = MIN_PARAM_VAL
        param.max = MAX_PARAM_VAL
        return true
    end
end

function _apply_multiplier_range!(param::Param, lower_mult::Float64, upper_mult::Float64, zero_span::Float64)
    default = param.default
    if iszero(default)
        if param.type in POSITIVE_ONLY_PARAM_TYPES
            param.min = 0.0
            param.max = zero_span
        elseif param.type in WEIGHT_PARAM_TYPES
            param.min = -zero_span
            param.max = zero_span
        else
            param.min = -zero_span
            param.max = zero_span
        end
        return
    end

    if default > 0.0
        min_val = default * lower_mult
        max_val = default * upper_mult
    else
        min_val = default * upper_mult
        max_val = default * lower_mult
    end

    if param.type in POSITIVE_ONLY_PARAM_TYPES
        min_val = max(min_val, 0.0)
        max_val = max(max_val, min_val + eps())
    end

    param.min = min_val
    param.max = max_val
end

function _describe_type_based_transform(param::Param;
                                        type_scales::Union{Nothing, AbstractDict{Symbol, Float64}}=nothing,
                                        target_bounds::Tuple{<:Real, <:Real}=(-5.0, 5.0))
    transform = build_param_reparam_transform([param.min], [param.max];
                                              target_bounds=target_bounds,
                                              types=[param.type],
                                              strategy=:typed,
                                              type_scales=type_scales)
    tr = transform.transforms[1]
    transform_name = string(nameof(typeof(tr)))
    eff_min = to_phys(transform.target_min, tr, transform.target_min, transform.target_max)
    eff_max = to_phys(transform.target_max, tr, transform.target_min, transform.target_max)
    return transform_name, eff_min, eff_max
end
# ==================================================================================
#  PARAMETER CONFIGURATION - HIGH-LEVEL INTERFACE
# ==================================================================================

"""Configure all network parameters: defaults, tunability, and bounds.

This is the main entry point for parameter configuration during network building.
Also supports independent updates via update_param_defaults!, update_param_tunability!, set_param_bounds!.

# Arguments
- `net::Network`: Network with parameters to configure
- `settings::Settings`: Complete settings object with optimization, network, and sampling settings
- `node_build_setts::Union{Nothing, Vector}`: Optional node build settings for model-specific overrides

# Examples
```julia
# During network building (updates everything)
configure_network_parameters!(net, settings)

# Independent updates
update_param_defaults!(net.params, settings)
set_param_bounds!(net.params, settings)
```
"""
function configure_network_parameters!(net, settings;
                                      node_build_setts::Union{Nothing, Vector}=nothing)
    # 1. Set parameter default values (from model definitions or empirical table)
    if node_build_setts !== nothing
        for node_sett in node_build_setts
            haskey(node_sett, :new_param_values) && 
                update_param_values!(net.params, node_sett[:new_param_values])
        end
    end
    update_param_defaults!(net.params, settings)
    
    # 2. Set tunability flags
    if node_build_setts !== nothing
        for node_sett in node_build_setts
            haskey(node_sett, :new_param_tunability) && 
                update_param_tunability!(net.params, node_sett[:new_param_tunability])
        end
    end
    
    # 3. Set parameter bounds (after defaults are set, as bounds may depend on them)
    set_param_bounds!(net.params, settings)
    
    return net
end


"""Update parameter default values from empirical table or keep model defaults.

# Arguments
- `paramset::ParamSet`: Parameters to update
- `settings::Settings`: Settings with optimization_settings.empirical_param_table_path

# Behavior
- If param_range_level == "empirical" AND empirical table path exists, applies empirical medians
- Otherwise keeps existing default values from model definitions
"""
function update_param_defaults!(paramset::ParamSet, settings)
    os = settings.optimization_settings
    
    # Only use empirical defaults when param_range_level is "empirical"
    if os === nothing || !hasproperty(os, :param_range_level)
        return paramset  # Keep model defaults
    end
    
    level = lowercase(os.param_range_level)
    if level != "empirical"
        return paramset  # Keep model defaults for non-empirical strategies
    end
    
    # Load and apply empirical defaults
    empirical_table = nothing
    if hasproperty(os, :empirical_param_table_path) && 
       os.empirical_param_table_path !== nothing && os.empirical_param_table_path != ""
        try
            empirical_table = CSV.read(os.empirical_param_table_path, DataFrame)
            vprint("Loaded empirical parameter table for defaults: $(os.empirical_param_table_path)"; level=2)
        catch e
            @warn "Failed to load empirical parameter table from $(os.empirical_param_table_path): $e"
            return paramset
        end
    else
        @warn "param_range_level='empirical' requires empirical_param_table_path in settings"
        return paramset
    end
    
    if empirical_table !== nothing
        task = hasproperty(settings.data_settings, :task_type) ? Symbol(settings.data_settings.task_type) : :rest
        _apply_empirical_defaults!(paramset, empirical_table; task=task)
    end
    
    return paramset
end


"""Set parameter bounds based on specified strategy.

# Strategies (from settings.optimization_settings.param_range_level):
- **"empirical"**: Use percentiles from empirical table
- **"unbounded"**: Set all bounds to ±Inf (was called "type_based")
- **"low", "medium", "high", "ultra"**: Multiplier-based ranges around defaults

# Arguments
- `paramset::ParamSet`: Parameters to set bounds for
- `settings::Settings`: Complete settings object

# Examples
```julia
set_param_bounds!(net.params, settings)  # Use strategy from settings
```
"""
function set_param_bounds!(paramset::ParamSet, settings)
    os = settings.optimization_settings
    
    if os === nothing || !hasproperty(os, :param_range_level)
        @warn "No optimization settings found, skipping bounds update"
        return paramset
    end
    
    level = lowercase(os.param_range_level)
    type_scales = hasproperty(os, :reparam_type_scales) && !isempty(os.reparam_type_scales) ? 
                  os.reparam_type_scales : nothing
    
    if level == "empirical"
        # Load empirical table and apply bounds
        empirical_table = nothing
        if hasproperty(os, :empirical_param_table_path) && 
           os.empirical_param_table_path !== nothing && os.empirical_param_table_path != ""
            try
                empirical_table = CSV.read(os.empirical_param_table_path, DataFrame)
                vprint("Loaded empirical parameter table for bounds: $(os.empirical_param_table_path)"; level=2)
            catch e
                error("param_range_level='empirical' requires valid empirical_param_table_path: $e")
            end
        else
            error("param_range_level='empirical' requires empirical_param_table_path in settings")
        end
        
        task = hasproperty(settings.data_settings, :task_type) ? Symbol(settings.data_settings.task_type) : :rest
        bounds_task = hasproperty(os, :bounds_task) ? Symbol(os.bounds_task) : :global
        lb_col = hasproperty(os, :empirical_lb_col) ? Symbol(os.empirical_lb_col) : Symbol("5perc")
        ub_col = hasproperty(os, :empirical_ub_col) ? Symbol(os.empirical_ub_col) : Symbol("95perc")
        _apply_empirical_bounds!(paramset, empirical_table; task=task, bounds_task=bounds_task, lb_col=lb_col, ub_col=ub_col)
        return paramset
    end
    
    if level == "unbounded"
        # Set all bounds to ±Inf (useful when relying entirely on reparametrization transforms)
        for param in paramset.params
            if is_verbose(2)
                vprint(string("Current param: ", param.name, " | default: ", param.default,
                              " | min: ", param.min, " | max: ", param.max); level=2)
            end
            _apply_unbounded_range!(param)
            if is_verbose(2)
                transform_name, eff_min, eff_max = _describe_type_based_transform(param; type_scales=type_scales)
                vprint(string("Updated param: ", param.name, " | type: ", param.type,
                              " | transform: ", transform_name, " | default: ", param.default,
                              " | min: ", round(eff_min; digits=6),
                              " | max: ", round(eff_max; digits=6)); level=2)
            end
        end
        return paramset
    end

    # Multiplier-based bounds
    haskey(PARAM_RANGE_MULTIPLIERS, level) || error("Unknown parameter range level: $(os.param_range_level). Valid options: 'empirical', 'unbounded', 'low', 'medium', 'high', 'ultra'")
    lower_mult, upper_mult = PARAM_RANGE_MULTIPLIERS[level]
    zero_span = ZERO_DEFAULT_SPANS[level]

    for param in paramset.params
        if is_verbose(2)
            vprint(string("Current param: ", param.name, " | default: ", param.default,
                          " | min: ", param.min, " | max: ", param.max); level=2)
        end
        _apply_multiplier_range!(param, lower_mult, upper_mult, zero_span)
        if is_verbose(2)
            vprint(string("Updated param: ", param.name, " | type: ", param.type,
                          " | strategy: multiplier | default: ", param.default,
                          " | min: ", param.min, " | max: ", param.max); level=2)
        end
    end
    return paramset
end


# ==================================================================================
#  LEGACY COMPATIBILITY WRAPPER
# ==================================================================================

"""Legacy wrapper for update_param_minmax! -> set_param_bounds!.

**DEPRECATED**: Use `set_param_bounds!(paramset, settings)` or `configure_network_parameters!(net, settings)` instead.
"""
function update_param_minmax!(paramset::ParamSet, param_range_level::String="medium";
                              type_scales::Union{Nothing, AbstractDict{Symbol, Float64}}=nothing,
                              empirical_table::Union{Nothing, DataFrame}=nothing,
                              task::Symbol=:rest,
                              bounds_task::Symbol=:global)
    @warn "update_param_minmax! is deprecated. Use set_param_bounds!(paramset, settings) instead."
    
    # Create minimal settings-like structure for backwards compatibility
    # This is a hack to support old calling convention
    if param_range_level == "empirical" && empirical_table !== nothing
        _apply_empirical_bounds!(paramset, empirical_table; task=task, bounds_task=bounds_task)
    elseif param_range_level == "type_based" || param_range_level == "unbounded"
        for param in paramset.params
            _apply_unbounded_range!(param)
        end
    else
        haskey(PARAM_RANGE_MULTIPLIERS, lowercase(param_range_level)) || 
            error("Unknown parameter range level: $param_range_level")
        lower_mult, upper_mult = PARAM_RANGE_MULTIPLIERS[lowercase(param_range_level)]
        zero_span = ZERO_DEFAULT_SPANS[lowercase(param_range_level)]
        for param in paramset.params
            _apply_multiplier_range!(param, lower_mult, upper_mult, zero_span)
        end
    end
    return paramset
end


function param_in_paramset(paramset::ParamSet, param::Param)::Bool
    for p in paramset.params
        if p.name == param.name &&
           p.parent_pop.parent_node.id == param.parent_pop.parent_node.id &&
           p.parent_pop.id == param.parent_pop.id
            return true
        end
    end
    return false
end


"""
    needs_tscale(paramset::ParamSet)::Bool

Determine if a node/network requires a time-scale parameter `tscale` for optimization.
Returns `true` if the parameter set contains NO parameters with time-related types
(:frequency, :time_constant, :rate), indicating a \"unitless\" model (e.g., FHN, VDP, SL)
that needs explicit time scaling.
Exception: Models using voltage_gated_dynamics (e.g., Larter-Breakspear) have internal
time constants as part of their biophysics but still need tscale for frequency tuning.
This exception is handled in construct_network_dynamics! in build_network.jl."""
function needs_tscale(paramset::ParamSet)::Bool
    for param in paramset.params
        if param.type in TIME_RELATED_PARAM_TYPES
            return false  # Has time-related params, doesn't need tscale
        end
    end
    return true  # No time-related params found, needs tscale
end


function sample_param_values(paramset::ParamSet; 
                             p_subset::Vector{<:Union{Num, Symbol}}=Num[], 
                             return_type::String="named_tuple")
 
    param_symbols = get_symbols(paramset, sort=true)

    if !isempty(p_subset)
        if eltype(p_subset) == Symbol
            p_subset = [string2num(string(s)) for s in p_subset]
        end
        p_subset_filtered = Num[]
        for p_sym in param_symbols
            for p_target in p_subset
                if isequal(p_sym, p_target)
                    push!(p_subset_filtered, p_sym)
                    break
                end
            end
        end
        p_subset = p_subset_filtered
    else
        p_subset = param_symbols
    end

    p_names = Symbol[]
    p_values = Float64[]

    for ip in p_subset
        p = get_param_by_symbol(paramset, ip)
        if Symbol(p.name) in p_names
            vwarn("Parameter $(p.name) already exists in names. Skipping.")
            continue
        end
        push!(p_names, Symbol(p.name))
        
        # Handle infinite bounds with sensible defaults
        p_min = isinf(p.min) ? -5.0 : p.min
        p_max = isinf(p.max) ? 5.0 : p.max
        
        push!(p_values, rand() * (p_max - p_min) + p_min)
    end

    if return_type == "named_tuple"
        return (; zip(p_names, p_values)...)
    elseif return_type == "vector"
        return p_values
    else
        error("Unknown return_type: $return_type. Use \"named_tuple\" or \"vector\".")
    end
end

function get_param_tunability(paramset::ParamSet; 
                              p_subset::Vector{<:Union{Num, Symbol}}=Num[], 
                              return_type::String="named_tuple",
                              only_tunable::Bool=false)
    # Resolve subset to Num symbols (consistent with other utilities)
    param_symbols = get_symbols(paramset, sort=true)
    if !isempty(p_subset)
        if eltype(p_subset) == Symbol
            p_subset = [string2num(string(s)) for s in p_subset]
        end
        p_subset_filtered = Num[]
        for p_sym in param_symbols
            for p_target in p_subset
                if isequal(p_sym, p_target)
                    push!(p_subset_filtered, p_sym)
                    break
                end
            end
        end
        p_subset = p_subset_filtered
    else
        p_subset = param_symbols
    end

    # Collect names and tunability flags
    p_names = Symbol[]
    flags = Bool[]
    for ip in p_subset
        p = get_param_by_symbol(paramset, ip)
        # avoid duplicates
        if Symbol(p.name) in p_names
            vwarn("Parameter $(p.name) already exists in names. Skipping.")
            continue
        end
        if !only_tunable || p.tunable
            push!(p_names, Symbol(p.name))
            push!(flags, Bool(p.tunable))
        end
    end

    if return_type == "dict"
        return Dict(p_names[i] => flags[i] for i in eachindex(p_names))
    elseif return_type == "named_tuple"
        return (; (p_names[i] => flags[i] for i in eachindex(p_names))...)
    else
        error("Unknown return_type: $return_type. Use \"dict\", \"named_tuple\".")
    end
end

function get_param_type(paramset::ParamSet;
                        p_subset::Vector{<:Union{Num, Symbol}}=Num[],
                        return_type::String="named_tuple")
    param_symbols = get_symbols(paramset, sort=true)
    if !isempty(p_subset)
        if eltype(p_subset) == Symbol
            p_subset = [string2num(string(s)) for s in p_subset]
        end
        p_subset_filtered = Num[]
        for p_sym in param_symbols
            for p_target in p_subset
                if isequal(p_sym, p_target)
                    push!(p_subset_filtered, p_sym)
                    break
                end
            end
        end
        p_subset = p_subset_filtered
    else
        p_subset = param_symbols
    end

    p_names = Symbol[]
    p_types = Symbol[]
    for ip in p_subset
        p = get_param_by_symbol(paramset, ip)
        if Symbol(p.name) in p_names
            vwarn("Parameter $(p.name) already exists in names. Skipping.")
            continue
        end
        push!(p_names, Symbol(p.name))
        push!(p_types, Symbol(p.type))
    end

    if return_type == "dict"
        return Dict(p_names[i] => p_types[i] for i in eachindex(p_names))
    elseif return_type == "named_tuple"
        return (; (p_names[i] => p_types[i] for i in eachindex(p_names))...)
    else
        error("Unknown return_type: $return_type. Use \"dict\" or \"named_tuple\".")
    end
end


# ==================================================================================
#  EMPIRICAL PARAMETER HELPERS (SPLIT INTO DEFAULTS AND BOUNDS)
# ==================================================================================

"""Apply empirical default values from task-specific medians."""
function _apply_empirical_defaults!(paramset::ParamSet, joined_table::DataFrame;
                                   task::Symbol = :rest,
                                   task_col::Symbol = :task,
                                   type_col::Symbol = :type,
                                   median_col::Symbol = :median,
                                   retag_scale::Bool = true,
                                   clamp_to_positive::Bool = true)
    
    tbl = copy(joined_table)
    tbl[!, :_type_sym] = Symbol.(_normalize_param_type.(Symbol.(String.(tbl[!, type_col]))))
    tbl[!, :_task_sym] = Symbol.(lowercase.(String.(tbl[!, task_col])))
    task_sym = Symbol(lowercase(String(task)))
    
    df_task = tbl[tbl[!, :_task_sym] .== task_sym, :]
    defaults = Dict{Symbol,Float64}()
    for r in eachrow(df_task)
        defaults[r[:_type_sym]] = Float64(r[median_col])
    end
    
    infer_scale_type = p -> begin
        desc = (hasproperty(p, :description) && getfield(p, :description) !== nothing) ?
               lowercase(String(getfield(p, :description))) : ""
        occursin("conn", desc) || occursin("coupling", desc) ? :population_coupling :
        occursin("steep", desc) || occursin("slope", desc) ? :rate :
        occursin("maximum output", desc) || occursin("amplitude", desc) ? :gain :
        occursin("decay", desc) || occursin("leak", desc) ? :damping : :unknown
    end
    
    for p in paramset.params
        ty = _normalize_param_type(getfield(p, :type))
        if ty == :scale && retag_scale
            ty = infer_scale_type(p)
            setfield!(p, :type, ty)
        end
        
        haskey(defaults, ty) || continue
        d = defaults[ty]
        
        if ty in POSITIVE_ONLY_PARAM_TYPES && clamp_to_positive
            d = max(d, 1e-9)
        end
        
        hasproperty(p, :default) && setfield!(p, :default, d)
    end
    
    return paramset
end


"""Apply empirical bounds from percentiles or other statistics.

Default columns are '5perc'/'95perc', but can be configured via settings to use:
- Percentiles: '5perc'/'95perc', '10perc'/'90perc'
- Quartiles: 'q1'/'q3'
- Extremes: 'min'/'max'
- Or any other columns in the empirical parameter table
"""
function _apply_empirical_bounds!(paramset::ParamSet, joined_table::DataFrame;
                                 task::Symbol = :rest,
                                 bounds_task::Symbol = :global,
                                 task_col::Symbol = :task,
                                 type_col::Symbol = :type,
                                 lb_col::Symbol = Symbol("5perc"),
                                 ub_col::Symbol = Symbol("95perc"),
                                 widen_frac::Float64 = 0.10,
                                 retag_scale::Bool = true,
                                 clamp_to_positive::Bool = true)
    
    tbl = copy(joined_table)
    tbl[!, :_type_sym] = Symbol.(_normalize_param_type.(Symbol.(String.(tbl[!, type_col]))))
    tbl[!, :_task_sym] = Symbol.(lowercase.(String.(tbl[!, task_col])))
    task_sym = Symbol(lowercase(String(task)))
    
    widen_bounds = (lb, ub, α) -> begin
        (!isfinite(lb) || !isfinite(ub) || ub <= lb) && return (lb, ub)
        span = ub - lb
        pad = α * span
        return (lb - pad, ub + pad)
    end
    
    bounds = Dict{Symbol,Tuple{Float64,Float64}}()
    if bounds_task == :global
        for sub in groupby(tbl, :_type_sym)
            ty = sub[1, :_type_sym]
            lb = minimum(Float64.(sub[!, lb_col]))
            ub = maximum(Float64.(sub[!, ub_col]))
            α = get(TYPE_WIDEN_FRAC, ty, DEFAULT_WIDEN_FRAC)
            bounds[ty] = widen_bounds(lb, ub, α)
        end
    elseif bounds_task == :task
        df_task = tbl[tbl[!, :_task_sym] .== task_sym, :]
        for r in eachrow(df_task)
            ty = r[:_type_sym]
            lb = Float64(r[lb_col])
            ub = Float64(r[ub_col])
            α = get(TYPE_WIDEN_FRAC, ty, DEFAULT_WIDEN_FRAC)
            bounds[ty] = widen_bounds(lb, ub, α)
        end
    else
        error("bounds_task must be :global or :task, got $(bounds_task)")
    end
    
    infer_scale_type = p -> begin
        desc = (hasproperty(p, :description) && getfield(p, :description) !== nothing) ?
               lowercase(String(getfield(p, :description))) : ""
        occursin("conn", desc) || occursin("coupling", desc) ? :population_coupling :
        occursin("steep", desc) || occursin("slope", desc) ? :rate :
        occursin("maximum output", desc) || occursin("amplitude", desc) ? :gain :
        occursin("decay", desc) || occursin("leak", desc) ? :damping : :unknown
    end
    
    for p in paramset.params
        ty = _normalize_param_type(getfield(p, :type))
        if ty == :scale && retag_scale
            ty = infer_scale_type(p)
            setfield!(p, :type, ty)
        end
        
        haskey(bounds, ty) || continue
        lb, ub = bounds[ty]
        
        if ty == :probability
            lb = max(lb, 0.0)
            ub = min(ub, 1.0)
        end
        
        if ty in POSITIVE_ONLY_PARAM_TYPES && clamp_to_positive
            lb = max(lb, 1e-9)
            ub = max(ub, lb + 1e-9)
        end
        
        hasproperty(p, :min) && setfield!(p, :min, lb)
        hasproperty(p, :max) && setfield!(p, :max, ub)
    end
    
    return paramset
end


"""
    set_params_empirically!(x, joined_table; task=:rest, bounds_task=:global, kwargs...)

Set both defaults and bounds from empirical table.

DEPRECATED: Use `update_param_defaults!` and `set_param_bounds!` separately for cleaner separation.

- Bounds come from GLOBAL per-type p5/p95 across all tasks (default), or from the chosen task only.
- Defaults come from task-specific medians (rest vs ssvep).
- Table types are normalized via `_normalize_param_type` so legacy rows like \"scale\" don't leak through.
- Enforces positivity for types in `POSITIVE_ONLY_PARAM_TYPES`.
"""
function set_params_empirically!(x, joined_table::DataFrame;
                                        task::Symbol = :rest,
                                        bounds_task::Symbol = :global,
                                        kwargs...)
    @warn "set_params_empirically! is deprecated. Use update_param_defaults! and set_param_bounds! separately."
    
    ps = x isa ParamSet ? x : getfield(x, :params)
    _apply_empirical_defaults!(ps, joined_table; task=task, kwargs...)
    _apply_empirical_bounds!(ps, joined_table; task=task, bounds_task=bounds_task, kwargs...)
    
    return x
end



# ── Compact summaries and pretty printing for parameters ─────────

function Base.summary(io::IO, p::Param)
    print(io, "Param($(p.name))")
end

function Base.show(io::IO, p::Param)
    node = p.parent_pop.parent_node
    pop  = p.parent_pop
    print(io,
        "Param($(p.name); type=$(p.type), default=$(p.default), ",
        "node=$(node.name)#$(node.id), pop=$(pop.name)#$(pop.id))"
    )
end

function Base.show(io::IO, ::MIME"text/plain", p::Param)
    node = p.parent_pop.parent_node
    pop  = p.parent_pop
    println(io, "Param")
    println(io, "  name: ", p.name, "   type: ", p.type)
    println(io, "  symbol: ", p.symbol)
    println(io, "  default: ", p.default, "   range: [", p.min, ", ", p.max, "]",
        "   tunable: ", p.tunable)
    print(io,
        "  parent: node=", node.name, "(", node.id, ")",
        ", pop=", pop.name, "(", pop.id, ")"
    )
end

function Base.show(io::IO, ps::ParamSet)
    print(io, "ParamSet($(length(ps.params)) params)")
end

function Base.show(io::IO, ::MIME"text/plain", ps::ParamSet)
    names = getfield.(ps.params, :name)
    preview = length(names) <= 10 ? names : vcat(names[1:9], "…", names[end])
    println(io, "ParamSet with $(length(names)) params")
    print(io, "  names: ", join(preview, ", "))
end


