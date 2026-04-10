#############################
# variables.jl  (refactored)
#############################

using SymbolicUtils

# You likely already have these utilities/types in your codebase:
# - string2num, string2symbolicfun, sort_symbols, vinfo
# - AbstractPopulation with fields: name, id, parent_node (which has name, id)

# -------------------------
# Common supertype & shared base
# -------------------------

abstract type Var end

Base.@kwdef mutable struct VarBase
    name::String
    symbol::Union{Num, SymbolicUtils.BasicSymbolic}
    eq_idx::Int64
    unit::String
    description::String
    parent_pop::AbstractPopulation
end

# Convenience accessors
name(v::Var)        = v.base.name
symbol(v::Var)      = v.base.symbol
eq_idx(v::Var)      = v.base.eq_idx
unit(v::Var)        = v.base.unit
description(v::Var) = v.base.description
parent_pop(v::Var)  = v.base.parent_pop

# Forward missing fields on Var to VarBase for compatibility with v.parent_pop, v.name, etc.
function Base.getproperty(v::Var, s::Symbol)
    if s === :base
        return getfield(v, :base)
    end
    T = typeof(v)
    if hasfield(T, s)
        return getfield(v, s)
    elseif hasfield(VarBase, s)
        return getfield(getfield(v, :base), s)
    else
        return getfield(v, s) # will throw a meaningful error if truly missing
    end
end

# -------------------------
# 1) State variables
# -------------------------
# State variables live inside the system (ODE/PDE/etc.), have init ranges,
# and flags about sending/receiving different kinds of inputs/outputs.

mutable struct StateVar <: Var
    base::VarBase
    gets_sensory_input::Bool
    gets_internode_input::Bool
    gets_interpop_input::Bool
    gets_additive_noise::Bool
    sends_internode_output::Bool
    sends_interpop_output::Bool
    sends_intrapop_output::Bool
    init_min::Float64
    init_max::Float64
end

# Ergonomic constructor mirroring your previous pattern
function StateVar(; name::String, eq_idx::Int64,
                  parent_pop::AbstractPopulation,
                  gets_sensory_input::Bool=false,
                  gets_internode_input::Bool=false,
                  gets_interpop_input::Bool=false,
                  gets_additive_noise::Bool=false,
                  sends_internode_output::Bool=false,
                  sends_interpop_output::Bool=false,
                  sends_intrapop_output::Bool=false,
                  init_min::Float64=-10., init_max::Float64=10.,
                  unit::String="", description::String="")
    # States use a standard symbolic variable
    sym = string2num(name)
    base = VarBase(; name, symbol=sym, eq_idx, unit, description, parent_pop)
    return StateVar(base, gets_sensory_input, gets_internode_input, gets_interpop_input,
                    gets_additive_noise, sends_internode_output, sends_interpop_output,
                    sends_intrapop_output, init_min, init_max)
end

# -------------------------
# 2) Driver variables (history, delayed links, sensory inputs, noise)
# -------------------------
# "Drivers" are exogenous or link variables that feed the system:
# - delayed history terms
# - sensory inputs (symbolic functions of time)
# - additive noise
mutable struct ExtraVar <: Var
    base::VarBase
    type::String
    is_history_var::Bool
    delay_param_symbol::Num
    is_sensory_input::Bool
    is_additive_noise::Bool
end

function ExtraVar(; name::String, type::String, eq_idx::Int64,
                   parent_pop::AbstractPopulation,
                   is_history_var::Bool=false,
                   is_sensory_input::Bool=false,
                   is_additive_noise::Bool=false,
                   unit::String="", description::String="")
    # Sensory inputs are typically symbolic functions; others are plain symbols
    sym = is_sensory_input ? string2symbolicfun(name) : string2num(name)

    # If this is a history/lagged driver, synthesize the associated delay parameter symbol
    delay_sym = if is_history_var
        name_parts = split(name, "₊")
        string2num("$(parent_pop.parent_node.name)₊τ_$(name_parts[end])")
    else
        Num("")
    end

    base = VarBase(; name, symbol=sym, eq_idx, unit, description, parent_pop)
    return ExtraVar(base, type, is_history_var, delay_sym,
                     is_sensory_input, is_additive_noise)
end

# -------------------------
# VarSet container (unchanged API)
# -------------------------

mutable struct VarSet
    vars::Vector{Var}
    function VarSet(vars::AbstractVector{<:Var})
        new(Vector{Var}(vars))
    end
    function VarSet()
        new(Vector{Var}())
    end
end

# -------------------------
# Utility: shallow copy (keeps categories)
# -------------------------

function copy_var(v::Var)::Var
    b = v.base
    if v isa StateVar
        return StateVar(; name=b.name, eq_idx=b.eq_idx, parent_pop=b.parent_pop,
                        gets_sensory_input=v.gets_sensory_input,
                        gets_internode_input=v.gets_internode_input,
                        gets_interpop_input=v.gets_interpop_input,
                        gets_additive_noise=v.gets_additive_noise,
                        sends_internode_output=v.sends_internode_output,
                        sends_interpop_output=v.sends_interpop_output,
                        sends_intrapop_output=v.sends_intrapop_output,
                        init_min=v.init_min, init_max=v.init_max,
                        unit=b.unit, description=b.description)
    elseif v isa ExtraVar
        return ExtraVar(; name=b.name, type=v.type, eq_idx=b.eq_idx, parent_pop=b.parent_pop,
                         is_history_var=v.is_history_var,
                         is_sensory_input=v.is_sensory_input,
                         is_additive_noise=v.is_additive_noise,
                         unit=b.unit, description=b.description)
    else
        error("Unknown Var subtype.")
    end
end

# -------------------------
# Set operations
# -------------------------

function add_var!(varset::VarSet, var::Var)::VarSet
    if var_in_varset(var, varset)
        return varset
    else
        push!(varset.vars, var)
        return varset
    end
end

function join_varsets(var_sets::Vector{VarSet})::VarSet
    all_vars = Vector{Var}()
    for vs in var_sets
        append!(all_vars, vs.vars)
    end
    return VarSet(all_vars)
end

function join_varsets!(varset::VarSet, var_sets::Vector{VarSet})::VarSet
    for vs in var_sets
        append!(varset.vars, vs.vars)
    end
    return varset
end

# -------------------------
# Lookups & filters
# -------------------------

function get_var_by_name(varset::VarSet, name::String)::Var
    for v in varset.vars
        if v.base.name == name
            return v
        end
    end
    error("Variable '$name' not found in VarSet.")
end

function get_vars_by_name(varset::VarSet, name::String)::VarSet
    filtered = [v for v in varset.vars if v.base.name == name]
    if isempty(filtered)
        vinfo("No variables found matching names: $(name)"; level=2)
        return VarSet()
    end
    return VarSet(filtered)
end

function get_var_by_symbol(varset::VarSet, sym::Union{Symbol, Num})
    target = sym isa Symbol ? Num(sym) : sym
    for v in varset.vars
        # Num == Num works; for BasicSymbolic, isequal is safer
        if isequal(symbol(v), target)
            return v
        end
    end
    error("Variable with symbol '$sym' not found in VarSet.")
end

function get_vars_by_type(varset::VarSet, type::String)::VarSet
    filtered = [v for v in varset.vars if (v isa ExtraVar) && (v.type == type)]
    if isempty(filtered)
        vinfo("No variables found of type: $type"; level=2)
        return VarSet()
    end
    return VarSet(filtered)
end

# Filters specific to StateVars
function get_vars_sending_internode_output(varset::VarSet)::VarSet
    filtered = [v for v in varset.vars if v isa StateVar && v.sends_internode_output]
    if isempty(filtered)
        vinfo("No variables found that send internode output in VarSet."; level=2)
        return VarSet()
    end
    return VarSet(filtered)
end

function get_vars_getting_internode_input(varset::VarSet)::VarSet
    filtered = [v for v in varset.vars if v isa StateVar && v.gets_internode_input]
    if isempty(filtered)
        vinfo("No variables found that get internode input in VarSet."; level=2)
        return VarSet()
    end
    return VarSet(filtered)
end

function get_vars_getting_interpop_input(varset::VarSet)::VarSet
    filtered = [v for v in varset.vars if v isa StateVar && v.gets_interpop_input]
    if isempty(filtered)
        vinfo("No variables found that get interpop input in VarSet."; level=2)
        return VarSet()
    end
    return VarSet(filtered)
end


function get_vars_sending_interpop_output(varset::VarSet)::VarSet
    filtered = [v for v in varset.vars if v isa StateVar && v.sends_interpop_output]
    if isempty(filtered)
        vinfo("No variables found that send interpopulation output in VarSet.";level=2)
        return VarSet()
    end
    return VarSet(filtered)
end

function get_vars_sending_intrapop_output(varset::VarSet)::VarSet
    filtered = [v for v in varset.vars if v isa StateVar && v.sends_intrapop_output]
    if isempty(filtered)
        vinfo("No variables found that send intrapopulation output in VarSet."; level=2)
        return VarSet()
    end
    return VarSet(filtered)
end

# Drivers: sensory input & history/delays
function get_sensory_input_var(varset::VarSet)::Union{Var, Nothing}
    for v in varset.vars
        if v isa ExtraVar && v.is_sensory_input
            return v
        end
    end
    vinfo("No sensory input variables found in VarSet."; level=2)
    return nothing
end

function get_history_vars(varset::VarSet)::VarSet
    filtered = [v for v in varset.vars if v isa ExtraVar && v.is_history_var]
    if isempty(filtered)
        vinfo("No history variables found in VarSet."; level=2)
        return VarSet()
    end
    return VarSet(filtered)
end

function get_state_vars(varset::VarSet)::VarSet
    filtered = [v for v in varset.vars if v isa StateVar]
    if isempty(filtered)
        vinfo("No state variables found in VarSet."; level=2)
        return VarSet()
    end
    return VarSet(filtered)
end

function get_vars_by_nodeid(varset::VarSet, node_id::Int)::VarSet
    filtered = [v for v in varset.vars if parent_pop(v).parent_node.id == node_id]
    if isempty(filtered)
        vinfo("No variables found for node_id: $node_id"; level=2)
        return VarSet()
    end
    return VarSet(filtered)
end

function get_vars_by_eq_idx(varset::VarSet, idx::Int)::VarSet
    filtered = [v for v in varset.vars if eq_idx(v) == idx]
    if isempty(filtered)
        vinfo("No variables found with system index: $idx"; level=2)
        return VarSet()
    end
    return VarSet(filtered)
end

# -------------------------
# Symbols & initial conditions
# -------------------------

function get_symbols(varset::VarSet; sort::Bool=true)
    syms = [symbol(v) for v in varset.vars]
    return sort ? sort_symbols(syms) : syms
end

"""
    sample_inits(network_vars::VarSet; subset::Vector{Num}=Num[], 
                 return_type::String="vector", sort::Bool=true,
                 seed::Union{Int, Nothing}=nothing)

Efficiently sample initial conditions for the given network variables.

# Arguments
- `network_vars::VarSet`: Variable set to sample from
- `subset::Vector{Num}`: Optional vector of variable symbols to restrict sampling (only state vars)
- `return_type::String`: "vector" or "named_tuple"
- `sort::Bool`: If true, sorts variables for reproducibility
- `seed::Union{Int, Nothing}`: Random seed for reproducibility (if nothing, uses global RNG)

Returns a vector or named tuple of initial state values, respecting init_min/init_max bounds.
"""
function sample_inits(network_vars::VarSet; subset::Vector{Num}=Num[], 
                     return_type::String="vector", sort::Bool=true,
                     seed::Union{Int, Nothing}=nothing)
    # Use provided seed or global RNG
    rng = seed === nothing ? Random.GLOBAL_RNG : MersenneTwister(seed)
    
    # target only state variables
    state_syms = get_symbols(get_state_vars(network_vars); sort=true)
    if isempty(subset)
        subset = [s for s in state_syms]
    end

    names = Symbol[]
    values = Float64[]

    for var_sym in subset
        v = get_var_by_symbol(network_vars, var_sym)
        if !(v isa StateVar)
            # ignore non-state entries in subset, if any
            continue
        end
        
        # Sample respecting bounds strictly
        if isfinite(v.init_min) && isfinite(v.init_max)
            # Both bounds finite: uniform in [init_min, init_max]
            init = v.init_min + (v.init_max - v.init_min) * rand(rng)
        elseif isfinite(v.init_min) && !isfinite(v.init_max)
            # Lower bound only: sample in [init_min, init_min + 10]
            init = v.init_min + 10.0 * rand(rng)
        elseif !isfinite(v.init_min) && isfinite(v.init_max)
            # Upper bound only: sample in [init_max - 10, init_max]
            init = v.init_max - 10.0 * rand(rng)
        else
            # Unbounded: standard normal
            init = randn(rng)
        end
        
        push!(names, Symbol(name(v)))
        push!(values, Float64(init))
    end

    if return_type == "named_tuple"
        return (; zip(names, values)...)
    elseif return_type == "vector"
        return values
    else
        error("Unknown return_type: $return_type. Use \"named_tuple\" or \"vector\".")
    end
end

function update_var_inits!(varset::VarSet, new_values::AbstractDict)::VarSet
    for (nm, (lo, hi)) in new_values
        try
            v = get_var_by_name(varset, nm)
            if v isa StateVar
                v.init_min = lo
                v.init_max = hi
            else
                vwarn("Var '$nm' is not a StateVar. Skipping init update."; level=2)
            end
        catch e
            vwarn("Var '$nm' not found in VarSet. Skipping update."; level=2)
            continue
        end
    end
    return varset
end


"""
    update_var_inits!(varset::VarSet, var_name::String, init_min::Real, init_max::Real)

Update initialization bounds for a single state variable.

# Arguments
- `varset::VarSet`: Variable set to update
- `var_name::String`: Name of the state variable to update
- `init_min::Real`: New minimum initialization value
- `init_max::Real`: New maximum initialization value

# Examples
```julia
update_var_inits!(net.vars, "E", 0.1, 0.5)
```
"""
function update_var_inits!(varset::VarSet, var_name::String, init_min::Real, init_max::Real)::VarSet
    try
        v = get_var_by_name(varset, var_name)
        if v isa StateVar
            v.init_min = Float64(init_min)
            v.init_max = Float64(init_max)
        else
            vwarn("Var '$var_name' is not a StateVar. Skipping init update."; level=2)
        end
    catch e
        vwarn("Var '$var_name' not found in VarSet. Skipping update."; level=2)
    end
    return varset
end

"""
    get_var_mean_inits(varset::VarSet; return_type::String="named_tuple")

Return mean initial values for state variables in `varset` as a NamedTuple.
Mean is (init_min + init_max)/2 when both bounds are finite; otherwise 0.0.
If duplicate variable names exist, the first occurrence is used.
"""
function get_var_mean_inits(varset::VarSet; return_type::String="named_tuple")
    vals = Dict{Symbol, Float64}()
    for v in varset.vars
        if v isa StateVar
            nm = Symbol(name(v))
            if !haskey(vals, nm)
                mean_init = (isfinite(v.init_min) && isfinite(v.init_max)) ? (v.init_min + v.init_max) / 2 : 0.0
                vals[nm] = Float64(mean_init)
            end
        end
    end
    if return_type == "named_tuple"
        return (; vals...)
    elseif return_type == "vector"
        return collect(values(vals))
    else
        error("Unknown return_type: $return_type. Use \"named_tuple\" or \"vector\".")
    end
end

"""
    get_var_minmax_values(varset::VarSet; return_type::String="named_tuple", include_infinite::Bool=false)

Collect initial (min,max) bounds for state variables in `varset`.

- return_type = "named_tuple" → NamedTuple(name => (min,max))
- return_type = "dict"        → Dict{Symbol,Tuple{Float64,Float64}}
- return_type = "vector"      → (Vector{Float64}, Vector{Float64}) of (mins, maxs) aligned with collected names

By default, variables with non-finite bounds are skipped. Set `include_infinite=true` to include them.
"""
function get_var_minmax_values(varset::VarSet; return_type::String="named_tuple", include_infinite::Bool=false)
    names = Symbol[]
    mins  = Float64[]
    maxs  = Float64[]

    for v in varset.vars
        if v isa StateVar
            lo, hi = v.init_min, v.init_max
            if include_infinite || (isfinite(lo) && isfinite(hi))
                push!(names, Symbol(name(v)))
                push!(mins, Float64(lo))
                push!(maxs, Float64(hi))
            end
        end
    end

    if return_type == "named_tuple"
        return (; (names[i] => (mins[i], maxs[i]) for i in eachindex(names))...)
    elseif return_type == "dict"
        return Dict(names[i] => (mins[i], maxs[i]) for i in eachindex(names))
    elseif return_type == "vector"
        return (mins, maxs)
    else
        error("Unknown return_type: $return_type. Use \"named_tuple\", \"dict\", or \"vector\".")
    end
end

# -------------------------
# Equality/containment & helpers
# -------------------------

function get_postfix_index(v::Var)
    m = match(r"\d+$", name(v))
    return m === nothing ? 0 : parse(Int, m.match)
end

function var_in_varset(v::Var, varset::VarSet)::Bool
    for w in varset.vars
        if name(w) == name(v) &&
           parent_pop(w).parent_node.id == parent_pop(v).parent_node.id &&
           parent_pop(w).id == parent_pop(v).id
            return true
        end
    end
    return false
end

function get_highest_postfix_index(varset::VarSet; node_id::Int=0, pop_id::Int=0, var_idx_only::Bool=true)
    vs = (node_id != 0) ? get_vars_by_nodeid(varset, node_id) : varset

    indices = Int[]
    for v in vs.vars
        push!(indices, get_postfix_index(v))
    end

    if pop_id != 0
        indices = filter(i -> !isempty(string(i)) && parse(Int, string(i)[1]) == pop_id, indices)
    end

    if var_idx_only
        indices = [length(string(i)) > 1 ? parse(Int, string(i)[2:end]) : 0 for i in indices]
    end

    return isempty(indices) ? 0 : maximum(indices)
end

# -------------------------
# Pretty printing
# -------------------------

function Base.summary(io::IO, v::Var)
    print(io, nameof(typeof(v)), "(", name(v), ")")
end

function Base.show(io::IO, v::Var)
    b = v.base
    if v isa StateVar
        print(io, "StateVar(", b.name, "; type=state, eq_idx=", b.eq_idx, ", ",
              "node=", b.parent_pop.parent_node.name, "#", b.parent_pop.parent_node.id, ", ",
              "pop=", b.parent_pop.name, "#", b.parent_pop.id, ")")
    elseif v isa ExtraVar
        print(io, "ExtraVar(", b.name, "; type=", v.type, ", eq_idx=", b.eq_idx, ", ",
              "node=", b.parent_pop.parent_node.name, "#", b.parent_pop.parent_node.id, ", ",
              "pop=", b.parent_pop.name, "#", b.parent_pop.id, ")")
    else
        print(io, "Var(", b.name, ")")
    end
end

function Base.show(io::IO, ::MIME"text/plain", v::Var)
    b = v.base
    println(io, nameof(typeof(v)))
    println(io, "  name: ", b.name)
    if v isa StateVar
        println(io, "  eq_idx: ", b.eq_idx)
        println(io,
            "  flags: gets_sensory=", v.gets_sensory_input,
            " gets_internode=", v.gets_internode_input,
            " gets_interpop=", v.gets_interpop_input,
            " sends_interpop=", v.sends_interpop_output,
            " noise=", v.gets_additive_noise
        )
        println(io, "  init: [", v.init_min, ", ", v.init_max, "]")
    elseif v isa ExtraVar
        println(io, "  type: ", v.type, "   eq_idx: ", b.eq_idx)
        println(io,
            "  flags: is_history_var=", v.is_history_var,
            " is_sensory_input=", v.is_sensory_input,
            " is_additive_noise=", v.is_additive_noise
        )
        println(io, "  delay_param_symbol: ", v.delay_param_symbol)
    end
    print(io,
        "  parent: node=", b.parent_pop.parent_node.name, "(", b.parent_pop.parent_node.id, ")",
        ", pop=", b.parent_pop.name, "(", b.parent_pop.id, ")"
    )
end

function Base.show(io::IO, vs::VarSet)
    print(io, "VarSet($(length(vs.vars)) vars)")
end

function Base.length(vs::VarSet)
    return length(vs.vars)
end

function Base.show(io::IO, ::MIME"text/plain", vs::VarSet)
    names = getfield.(getfield.(vs.vars, :base), :name)
    preview = length(names) <= 6 ? names : vcat(names[1:5], "…", names[end])
    println(io, "VarSet with $(length(names)) vars")
    print(io, "  names: ", join(preview, ", "))
end

# -------------------------
# Pretty printing summaries
# -------------------------

"""
    print_vars_summary(varset::VarSet; format_type::String="short")

Pretty-print variable initial values in compact format.

# Arguments
- `varset::VarSet`: Variable set to display
- `format_type::String`: Output format (default "short")
  - "short": Compact `name: [min, max]` format, one per line
  - "long": Extended format with equation index, input/output flags, and parent details

# Example
```julia
print_vars_summary(net.vars)
print_vars_summary(net.vars; format_type="long")
```
"""
function print_vars_summary(varset::VarSet; format_type::String="short")
    # Filter to only StateVar since those have init values
    state_vars = get_state_vars(varset)
    
    if isempty(state_vars.vars)
        println("[Variables (Init Ranges)]")
        println("  (no state variables)")
        return
    end
    
    println("[Variables (Init Ranges)]")
    
    if format_type == "short"
        for v in state_vars.vars
            if v isa StateVar
                println("  $(v.base.name): [$(v.init_min), $(v.init_max)]")
            end
        end
    else  # long format
        for v in state_vars.vars
            if v isa StateVar
                println("  $(v.base.name): [$(v.init_min), $(v.init_max)]")
                println("    eq_idx: $(v.base.eq_idx), gets_sensory: $(v.gets_sensory_input), sends_internode: $(v.sends_internode_output)")
            end
        end
    end
    println()  # Add extra newline for separation
end
