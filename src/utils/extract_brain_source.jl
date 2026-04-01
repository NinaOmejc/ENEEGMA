function extract_brain_sources(dfs::Vector{DataFrame}, net::Network)::Vector{DataFrame}
    """
    Extract EEG data from the simulation results.
    
    Args:
        dfs: Vector of DataFrames containing simulation results.
        net: The network object containing node information.
        gs: General settings for the simulation.
        ss: Simulation settings.
    
    Returns:
        A vector of DataFrames with extracted EEG data.
    """
    gs = net.settings.general_settings
    ss = net.settings.simulation_settings
    dfs_sources = Vector{DataFrame}(undef, length(dfs))
    
    for (i, df) in enumerate(dfs)
        dfs_sources[i] = extract_brain_source(df, net, gs, ss)
    end
    
    return dfs_sources
    
end


function extract_brain_source(df::DataFrame,
                               net::Network,
                               gs::GeneralSettings,
                               ss::SimulationSettings)::DataFrame
    """
    For every node in `net`, create a column `<node_name>₊source`
    containing either the raw column referenced by `node.brain_source`
    or an evaluated linear combination of columns.

    The returned DataFrame always has a `time` column copied
    verbatim from `df`.
    """
    df_sources = DataFrame(time = df.time)

    for node in net.nodes
        key = "$(node.name)₊source"

        try
            bsrc = node.brain_source

            if bsrc in names(df)                # fast path: simple column
                df_sources[!, key] = df[!, bsrc]

            else                               # need to evaluate a formula
                terms = _parse_terms(bsrc)
                df_sources[!, key] = _evaluate_combination(df, terms)
            end

        catch e
            vwarn("Warning: cannot resolve brain source for $(node.name), $(bsrc).  Skipping.")
            showerror(stderr, e)
            continue
        end
    end

    return df_sources
end

function _parse_terms(expr::String)
    """
    Return a vector of (sign::Float64, colname::String, scale::Float64).

    Supports tokens like
        "x21"         → (+1) * df[!, "x21"]
        "-x31"        → (−1) * df[!, "x31"]
        "+0.5*x41"    → (+0.5) * df[!, "x41"]
    """
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

function _evaluate_combination(df::DataFrame, terms)::Vector{Float64}
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