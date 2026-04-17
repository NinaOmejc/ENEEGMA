
function string2num(name::String)::Num
    # Convert to a proper symbol
    sym = Symbol(name)
    
    # Create a symbolic variable using @variables
    ex = :(Symbolics.@variables $sym)
    try
        # Evaluate in the current module
        Core.eval(@__MODULE__, ex)
        # Return the variable
        return Core.eval(@__MODULE__, sym)
    catch e
        # If variable already exists, just get it
        return Symbolics.variable(sym)
    end
end


"""
    string2symbolicfun(name::String) -> Symbolics.Term

Creates a callable symbolic function of time (t).
Returns a symbolic function that can be used in differential equations.

# Arguments
- `name::String`: The name of the callable symbol to create

# Returns
- `Symbolics.Term`: A callable symbolic function
"""
function string2symbolicfun(name::String)
    # Convert to a proper symbol
    sym = Symbol(name)
    
    # Create a callable symbolic variable using @syms
    ex = :(Symbolics.@syms $(sym)(t::Real))
    
    try
        # Evaluate in the current module
        Core.eval(@__MODULE__, ex)
        # Return the callable function
        return Core.eval(@__MODULE__, :($sym))
    catch e
        # If function already exists, just get it
        # This is a bit trickier for callable symbols
        # We'll use the registry of symbolic variables
        if haskey(Symbolics.symtype_registry, sym)
            # The function was already defined
            return Core.eval(@__MODULE__, :($sym))
        else
            # Rethrow the error if it's not just a redefinition issue
            rethrow(e)
        end
    end
end


#= 
function soft_wrap(eq::LaTeXString; width::Int = 110)::LaTeXString
    raw = String(eq)                   # work with a plain String
    buf = IOBuffer()
    col = 0

    for tok in split(raw, ' ')
        toklen = ncodeunits(tok) + 1   # token width + trailing space
        if col + toklen > width && occursin(r"^[\+\-]", tok)
            print(buf, "\\\\\n\\;")    # manual break + small indent
            col = 0
        end
        print(buf, tok, ' ')
        col += toklen
    end
    return LaTeXString(String(take!(buf)))
end



function transform2latex(eqs::Vector{Equation};
                         show_text::Bool = false,
                         show_plot::Bool = true,
                         path_tex::String = "")::Nothing

    # -------------------------------
    # (a) preview in REPL / notebook
    # -------------------------------
    if show_text
        println(latexify(eqs))
    end
    if show_plot
        render(latexify(eqs))
    end

    # -------------------------------
    # (b) write a .tex file if asked
    # -------------------------------
    if !isempty(path_tex)
        # 1. latexify each equation -> LaTeXString
        raw_eqs = latexify.(eqs)      # one eq per env

        # 2. wrap long lines
        wrapped_eqs = soft_wrap.(raw_eqs)

        # 3. convert every LaTeXString -> String and join with new-lines
        eq_block = join(String.(wrapped_eqs), "\n\n")

        # 4. embed in a minimal LaTeX document
        latex_code = """
        \\documentclass{article}
        \\usepackage[a4paper,landscape,margin=1cm]{geometry}
        \\usepackage{amsmath}                % you keep amsmath
        \\begin{document}
        $eq_block
        \\end{document}
        """

        open(path_tex, "w") do f
            write(f, latex_code)
        end
    end
    return nothing
end =#


# 1. no manual soft_wrap neededfunction transform2latex(eqs::Vector{Equation};
function transform2latex(eqs::Vector{Equation};
                         show_text::Bool=false,
                         show_plot::Bool=true,
                         path_tex::String="")
    # 1) Get only the raw math, no \begin{} from latexify
    tex_raw = latexify.(eqs; env = :raw)   # guaranteed supported

    # 2) Wrap each raw math in breqn’s dmath
    breqn_wrapped = ["\\begin{dmath}\n" * String(x) * "\n\\end{dmath}"
                     for x in tex_raw]

    # 3) Join all equations into one block
    tex_block = join(breqn_wrapped, "\n\n")

    # Optional preview
    if show_text
        println(tex_block)
    end
    if show_plot
        render(LaTeXString(tex_block))
    end

    # 4) Write a standalone .tex file if requested
    if !isempty(path_tex)
        latex_doc = """
        \\documentclass[fleqn]{article}
        \\usepackage[a4paper,landscape,margin=1cm]{geometry}
        \\usepackage{breqn}   % dmath for automatic line breaking
        \\setlength{\\mathindent}{0pt}
        \\begin{document}
        $tex_block
        \\end{document}
        """
        open(path_tex, "w") do io
            write(io, latex_doc)
        end
    end
    return nothing
end

#= function sort_symbols(symbols::Union{Vector{Symbol}, Vector{Num}})::Vector
    function parse_symbol_parts(sym::Union{Symbol, Num})
        s = string(sym)
        # Match node_name, separator, var, and trailing digits
        m = match(r"^([^\+_]+)[₊_](\D+?)(\d*)$", s)
        if m !== nothing
            node = m.captures[1]
            var = m.captures[2]
            idx = isempty(m.captures[3]) ? 0 : parse(Int, m.captures[3])
            return (node, var, idx)
        else
            # fallback: try to split and parse what we can
            parts = split(s, r"[₊_]")
            if length(parts) == 2
                var, idx = match(r"(\D+?)(\d+)$", parts[2]).captures
                return (parts[1], var, parse(Int, idx))
            else
                return (s, "", 0)
            end
        end
    end
    sorted_syms = sort(symbols, by=parse_symbol_parts)
    return sorted_syms
end
 =#

function sort_symbols(symbols::Union{Vector{Symbol}, Vector{Num},  Vector{Any}})::Vector
    function parse_symbol_parts(sym::Union{Symbol, Num})
        s = string(sym)
        
        # Regex to capture:
        # 1. Node name (anything before ₊ or _)
        # 2. Variable prefix (letters after ₊ or _)
        # 3. Full numeric postfix (all digits after variable prefix, can be empty)
        m = match(r"^([^\+_]+)[₊_]([a-zA-Z]+)(\d*)$", s)

        if m !== nothing
            node_str = m.captures[1]
            var_prefix_str = m.captures[2] # e.g., "c", "x"
            full_numeric_postfix_str = m.captures[3]

            pop_id_int = 0
            param_idx_int = 0

            if !isempty(full_numeric_postfix_str)
                # First digit of the full_numeric_postfix_str is the pop_id
                pop_id_str = string(full_numeric_postfix_str[1])
                pop_id_int = parse(Int, pop_id_str)

                # The rest of the full_numeric_postfix_str is the param_idx
                param_idx_suffix_str = SubString(full_numeric_postfix_str, 2) # Can be empty
                if !isempty(param_idx_suffix_str)
                    param_idx_int = parse(Int, param_idx_suffix_str)
                else
                    # If full_numeric_postfix_str was only one digit (e.g., "c1"),
                    # pop_id is that digit, and param_idx can be considered 0 or 1.
                    # For consistency with "c11" (pop=1, idx=1), let's make single digit imply idx=0
                    # or if your convention is that "c1" means pop_id=1, param_idx=1, set param_idx_int = 1 here.
                    # Based on N1₊c11 -> (N1,1,1,"c"), this implies if only pop_id is present, param_idx is 0.
                    param_idx_int = 0 
                end
            end
            # Sorting tuple: (Node, PopID, Param/VarIdx, VarPrefix for tie-breaking)
            return (node_str, pop_id_int, param_idx_int, var_prefix_str)
        else
            # Fallback for symbols that don't match the primary pattern.
            # Try to extract a numeric suffix for basic numeric sorting.
            m_fallback = match(r"^(.*?)(\d+)$", s)
            if m_fallback !== nothing
                prefix_str = m_fallback.captures[1]
                numeric_suffix_int = parse(Int, m_fallback.captures[2])
                # Sort these after well-formed symbols by using a large pop_id_int value.
                return (prefix_str, typemax(Int) - 1, numeric_suffix_int, "") 
            else
                # If no numeric suffix, sort by the full string, placing them last.
                return (s, typemax(Int), 0, "")
            end
        end
    end

    # Sort the symbols using the parsed parts.
    # The `lt` keyword argument can be used for more complex sorting if needed,
    # but tuple comparison should work directly here.
    sorted_syms = sort(symbols, by=parse_symbol_parts)
    return sorted_syms
end


"""
    rebuild_network_problem!(net::Network, new_params::Union{Dict, NamedTuple})::Nothing

Rebuild the network problem with updated parameters.

Updates net.problem with new parameter values. This is useful after modifying parameters
to ensure the problem reflects the current parameter state. The function handles both 
Dict and NamedTuple parameter specifications, automatically converting Dicts to NamedTuples
and merging with existing parameters to preserve unchanged values.

# Arguments
- `net::Network`: Network whose problem should be rebuilt
- `new_params::Union{Dict, NamedTuple}`: Parameters to update as Dict{String/Symbol, Real} or NamedTuple

# Returns
- `nothing`

# Example
```julia
net = build_network(settings)
# Modify parameter defaults
update_param_defaults!(net, Dict("N1₊tscale1" => 2.0, "N1₊c18" => 5.0))
# Problem is automatically rebuilt - now simulate with updated params
df = simulate_network(net)
```
"""
function rebuild_network_problem!(net::Network, new_params::Union{Dict, NamedTuple})::Nothing
    # Convert Dict to NamedTuple if needed
    params_for_remake = new_params
    
    if new_params isa Dict
        vinfo("Converting parameter Dict to NamedTuple"; level=2)
        # Handle both String and Symbol keys by converting to Symbols
        sym_dict = Dict(Symbol(k) => v for (k, v) in pairs(new_params))
        new_params_nt = NamedTuple(sym_dict)
        # Merge with existing parameters to preserve unchanged ones
        params_for_remake = merge(net.problem.p, new_params_nt)
    elseif new_params isa NamedTuple
        # Merge NamedTuple with existing parameters to preserve unchanged ones
        params_for_remake = merge(net.problem.p, new_params)
    end
    
    # Rebuild problem with new parameters
    net.problem = DifferentialEquations.remake(net.problem; p=params_for_remake)
    vinfo("Network problem rebuilt with updated parameters"; level=2)
    
    return nothing
end


"""
    update_param_defaults!(net::Network, dict::AbstractDict)::Network

Update parameter defaults and rebuild the network problem.

Provides a cleaner API for updating parameters when working with a Network object.
Updates parameter default values in net.params and automatically rebuilds net.problem
so it reflects the new parameter state. This eliminates the need to separately pass
parameters to simulate_network().

# Arguments
- `net::Network`: Network whose parameters should be updated
- `dict::AbstractDict`: Dictionary mapping parameter names to new default values

# Returns
- `net::Network`: The updated network (for method chaining)

# Example
```julia
net = build_network(settings)

# Update parameters and rebuild problem (single call)
update_param_defaults!(net, Dict("N1₊tscale1" => 1.777, "N1₊c18" => 7.9))

# Simulate with updated parameters (no need to pass new_params again)
df = simulate_network(net)
```

# See Also
- [`update_param_defaults!(::ParamSet, ::AbstractDict)`](@ref): Updates only the ParamSet
- [`rebuild_network_problem!`](@ref): Rebuilds network problem after parameter changes
"""
function update_param_defaults!(net::Network, dict::AbstractDict)::Network
    # Update parameter defaults in ParamSet
    update_param_defaults!(net.params, dict)
    
    # Rebuild network problem with updated parameters
    rebuild_network_problem!(net, dict)
    
    vinfo("Network $(net.name) parameters updated and problem rebuilt"; level=1)
    return net
end