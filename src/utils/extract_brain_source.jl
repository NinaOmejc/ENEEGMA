function _brain_source_target_nodes(net::Network,
                                    node_names::Union{Nothing, Vector{String}}=nothing)::Vector{String}
    return [String(node.name) for node in net.nodes
            if node_names === nothing || String(node.name) in node_names]
end

function _brain_source_output_key(target_node::String, output_column_style::Symbol)::String
    if output_column_style == :node_name
        return target_node
    elseif output_column_style == :source
        return string(target_node, "₊source")
    end

    error("Unknown output_column_style: $output_column_style. Supported styles: :source, :node_name")
end

function _resolve_brain_source_expression(settings::Settings,
                                          net::Network,
                                          target_node::String)::String
    expr = get(settings.network_settings.eeg_output, target_node, "")
    if isempty(expr)
        idx = findfirst(n -> String(n.name) == target_node, net.nodes)
        expr = idx !== nothing ? String(net.nodes[idx].brain_source) : ""
    end
    return expr
end

function _state_symbol_index_map(net::Network)::Dict{String, Int}
    state_syms = string.(get_symbols(get_state_vars(net.vars); sort=true))
    return Dict(sym => idx for (idx, sym) in enumerate(state_syms))
end

"""
    extract_brain_sources(settings::Settings, net::Network, df::DataFrame;
                          node_names::Union{Nothing, Vector{String}}=nothing,
                          return_source_expressions::Bool=false,
                          output_column_style::Symbol=:source)

Extract per-node brain source signals from a simulation dataframe.
"""
function extract_brain_sources(settings::Settings, net::Network, df::DataFrame;
    node_names::Union{Nothing, Vector{String}}=nothing,
    return_source_expressions::Bool=false,
    output_column_style::Symbol=:source)

    df_sources = DataFrame(time=df.time)
    target_nodes = _brain_source_target_nodes(net, node_names)
    source_expressions = Dict{String, String}()

    for target_node in target_nodes
        key = _brain_source_output_key(target_node, output_column_style)

        try
            expr = _resolve_brain_source_expression(settings, net, target_node)
            if isempty(expr)
                vwarn("No brain source expression for $target_node"; level=2)
                continue
            end

            if expr in names(df)
                df_sources[!, key] = df[!, expr]
            else
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
    extract_brain_sources(settings::Settings, net::Network, sol::SciMLBase.AbstractTimeseriesSolution;
                          node_names::Union{Nothing, Vector{String}}=nothing,
                          keep_idx=nothing,
                          return_source_expressions::Bool=false,
                          output_column_style::Symbol=:source,
                          return_type::Symbol=:dataframe)

Extract per-node brain source signals directly from a solution object without
materializing an intermediate dataframe. Use `return_type=:dict` for the hot
optimization path.
"""
function extract_brain_sources(settings::Settings,
                               net::Network,
                               sol::SciMLBase.AbstractTimeseriesSolution;
                               node_names::Union{Nothing, Vector{String}}=nothing,
                               keep_idx=nothing,
                               return_source_expressions::Bool=false,
                               output_column_style::Symbol=:source,
                               return_type::Symbol=:dataframe)
    sample_idx = keep_idx === nothing ? (1:length(sol.t)) : keep_idx
    target_nodes = _brain_source_target_nodes(net, node_names)
    source_expressions = Dict{String, String}()
    state_idx_map = _state_symbol_index_map(net)

    if return_type == :dataframe
        output = DataFrame(time=Float64.(sol.t[sample_idx]))
    elseif return_type == :dict
        output = Dict{String, Vector{Float64}}()
    else
        error("Unknown return_type: $return_type. Supported values: :dataframe, :dict")
    end

    for target_node in target_nodes
        key = _brain_source_output_key(target_node, output_column_style)

        try
            expr = _resolve_brain_source_expression(settings, net, target_node)
            if isempty(expr)
                vwarn("No brain source expression for $target_node"; level=2)
                continue
            end

            signal = if haskey(state_idx_map, expr)
                Vector{Float64}(sol[state_idx_map[expr], sample_idx])
            else
                terms = _parse_brain_source_terms(expr)
                _evaluate_brain_source_combination(sol, state_idx_map, sample_idx, terms)
            end

            if output isa DataFrame
                output[!, key] = signal
            else
                output[key] = signal
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
        return output, source_expressions
    end

    return output
end

"""
    _parse_brain_source_terms(expr::String)::Vector{Tuple{Float64, String, Float64}}

Parse a brain source expression into `(sign, colname, scale)` tuples.
"""
function _parse_brain_source_terms(expr::String)::Vector{Tuple{Float64, String, Float64}}
    terms = Tuple{Float64, String, Float64}[]
    clean = replace(expr, " " => "")
    matches = eachmatch(r"([+\-]?)(\d*\.?\d*)\*?([^+\-]+)", clean)
    for m in matches
        sign_str, scale_str, name = m.captures
        sign = sign_str == "-" ? -1.0 : 1.0
        scale = isempty(scale_str) ? 1.0 : parse(Float64, scale_str)
        push!(terms, (sign, name, scale))
    end
    return terms
end

"""
    _evaluate_brain_source_combination(df::DataFrame, terms)::Vector{Float64}

Evaluate a parsed brain source combination expression against dataframe columns.
"""
function _evaluate_brain_source_combination(df::DataFrame,
                                            terms::Vector{Tuple{Float64, String, Float64}})::Vector{Float64}
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

"""
    _evaluate_brain_source_combination(sol, state_idx_map, sample_idx, terms)::Vector{Float64}

Evaluate a parsed brain source combination expression directly against a
solution's state trajectories.
"""
function _evaluate_brain_source_combination(sol::SciMLBase.AbstractTimeseriesSolution,
                                            state_idx_map::Dict{String, Int},
                                            sample_idx,
                                            terms::Vector{Tuple{Float64, String, Float64}})::Vector{Float64}
    out = zeros(length(sample_idx))
    for (sgn, col, scl) in terms
        if haskey(state_idx_map, col)
            values = Vector{Float64}(sol[state_idx_map[col], sample_idx])
            @. out += sgn * scl * values
        else
            throw(KeyError(col))
        end
    end
    return out
end
