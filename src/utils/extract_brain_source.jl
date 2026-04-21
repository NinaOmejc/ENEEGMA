#= 
"""
    extract_brain_sources(dfs::Vector{DataFrame}, net::Network)::Vector{DataFrame}

Extract brain source (EEG) data from simulation results for all nodes.

For each DataFrame in the input vector, creates output DataFrame with columns
for each node's brain source signal. Returns vector of DataFrames.

# Arguments
- `dfs::Vector{DataFrame}`: Vector of DataFrames containing simulation results
- `net::Network`: Network object with node information and brain_source definitions

# Returns
- `Vector{DataFrame}`: Vector of DataFrames with extracted brain source signals
"""
function extract_brain_sources(dfs::Vector{DataFrame}, net::Network)::Vector{DataFrame}
    gs = net.settings.general_settings
    ss = net.settings.simulation_settings
    dfs_sources = Vector{DataFrame}(undef, length(dfs))
    
    for (i, df) in enumerate(dfs)
        dfs_sources[i] = extract_brain_source(df, net, gs, ss)
    end
    
    return dfs_sources
end


"""
    extract_brain_source(df::DataFrame, net::Network, gs::GeneralSettings, ss::SimulationSettings)::DataFrame

Extract brain source signals for all nodes from a single DataFrame.

For every node in the network, creates a column `<node_name>₊source` containing either:
- The raw column referenced by `node.brain_source` if it exists in df
- An evaluated linear combination (e.g., "x21 - x31") if expression given

The returned DataFrame always has a `time` column copied from input.

# Arguments
- `df::DataFrame`: Input DataFrame with time series data
- `net::Network`: Network with node brain_source definitions
- `gs::GeneralSettings`: General settings
- `ss::SimulationSettings`: Simulation settings

# Returns
- `DataFrame`: DataFrame with time column and `<node_name>₊source` columns for each node
"""
function extract_brain_source(df::DataFrame,
                               net::Network,
                               gs::GeneralSettings,
                               ss::SimulationSettings)::DataFrame
    df_sources = DataFrame(time = df.time)

    for node in net.nodes
        key = "$(node.name)₊source"

        try
            bsrc = node.brain_source

            if bsrc in names(df)
                # Fast path: simple column reference
                df_sources[!, key] = df[!, bsrc]
            else
                # Parse and evaluate as combination expression
                terms = _parse_brain_source_terms(bsrc)
                df_sources[!, key] = _evaluate_brain_source_combination(df, terms)
            end

        catch e
            vwarn("Warning: cannot resolve brain source for $(node.name), $(bsrc). Skipping.")
            showerror(stderr, e)
            continue
        end
    end

    return df_sources
end
 =#

"""
    extract_brain_sources(settings::Settings, net::Network, df::DataFrame;
                          node_names::Union{Nothing, Vector{String}}=nothing,
                          return_source_expressions::Bool=false)

Extract brain source signals for plotting, with optional node filtering and eeg_output overrides.

Returns a DataFrame with:
- `time` column (copied from input)
- `<node_name>₊source` columns for each target node

Uses `settings.network_settings.eeg_output` as override if set, falls back to `node.brain_source`.

# Arguments
- `settings::Settings`: Settings with optional eeg_output overrides
- `net::Network`: Network with node brain_source defaults
- `df::DataFrame`: DataFrame with simulation data
- `node_names::Union{Nothing, Vector{String}}=nothing`: Specific node names to include (all if nothing)
- `return_source_expressions::Bool=false`: If `true`, also return a dictionary
  mapping source column names to the resolved expressions used

# Returns
- `DataFrame`: Time + brain source signals for target nodes
- `Tuple{DataFrame, Dict{String, String}}`: Returned only when
  `return_source_expressions=true`

# Example
```julia
df_sources = extract_brain_sources(settings, net, df)
df_sources = extract_brain_sources(settings, net, df; node_names=["N1"])
df_sources, expr_map = extract_brain_sources(settings, net, df; return_source_expressions=true)
# Returns: DataFrame with columns [:time, :N1₊source] or [:time, :N1₊source, :N2₊source, ...]
```
"""
function extract_brain_sources(settings::Settings, net::Network, df::DataFrame;
    node_names::Union{Nothing, Vector{String}}=nothing,
    return_source_expressions::Bool=false)
    
    df_sources = DataFrame(time = df.time)
    
    # Filter nodes based on node_names parameter
    target_nodes = [String(node.name) for node in net.nodes 
                    if node_names === nothing || String(node.name) in node_names]
    
    source_expressions = Dict{String, String}()                
    for target_node in target_nodes
        key = "$(target_node)₊source"
        
        try
            # Determine expression: eeg_output override or brain_source default
            expr = get(settings.network_settings.eeg_output, target_node, "")
            if isempty(expr)
                idx = findfirst(n -> String(n.name) == target_node, net.nodes)
                expr = idx !== nothing ? String(net.nodes[idx].brain_source) : ""
            end
            
            if isempty(expr)
                vwarn("No brain source expression for $target_node"; level=2)
                continue
            end
            
            # Resolve expression
            if expr in names(df)
                # Simple column reference
                df_sources[!, key] = df[!, expr]
            else
                # Parse and evaluate combination
                terms = _parse_brain_source_terms(expr)
                df_sources[!, key] = _evaluate_brain_source_combination(df, terms)
            end
            source_expressions[key] = expr
            
        catch e
            source_expressions[key] = ""
            vwarn("Could not resolve brain source for $target_node: '$expr'"; level=2)
            showerror(stderr, e)
            continue
        end
    end
    
    if return_source_expressions
        return df_sources, source_expressions
    end

    return df_sources
end


"""
    _parse_brain_source_terms(expr::String)::Vector{Tuple{Float64, String, Float64}}

Parse a brain source expression into (sign, colname, scale) tuples.

Supports tokens like:
- "x21"         → (+1) * df[!, "x21"]
- "-x31"        → (−1) * df[!, "x31"]
- "+0.5*x41"    → (+0.5) * df[!, "x41"]
- "x21 - x31"   → Multiple terms with signs and scales
"""
function _parse_brain_source_terms(expr::String)::Vector{Tuple{Float64, String, Float64}}
    terms = []
    # remove spaces, then split on + and - while keeping the sign
    clean = replace(expr, " " => "")
    matches = eachmatch(r"([+\-]?)(\d*\.?\d*)\*?([^+\-]+)", clean)
    for m in matches
        sign_str, scale_str, name = m.captures
        sign  = sign_str == "-" ? -1.0 : 1.0
        scale = isempty(scale_str) ? 1.0 : parse(Float64, scale_str)
        push!(terms, (sign, name, scale))
    end
    return terms
end


"""
    _evaluate_brain_source_combination(df::DataFrame, terms::Vector{Tuple{Float64, String, Float64}})::Vector{Float64}

Evaluate a parsed brain source combination expression against DataFrame columns.

Computes: sum over terms of (sign * scale * df[!, column])
"""
function _evaluate_brain_source_combination(df::DataFrame, terms::Vector{Tuple{Float64, String, Float64}})::Vector{Float64}
    n = nrow(df)
    out = zeros(n)
    for (sgn, col, scl) in terms
        if col in names(df)
            @. out += sgn * scl * df[!, col]
        else
            throw(KeyError(col))
        end
    end
    return out
end
