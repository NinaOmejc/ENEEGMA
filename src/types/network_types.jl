###########################################################
## POPULATION STRUCTURE
############################################################
"""
This module defines the `Population` type, which represents a collection of neurons that have
a common dynamics within one node. They have a common way to receive input (synapto-dendritic dynamics), 
process it internally (somatic dynamics), and produce/cary output (spatial-gradient dynamics). Each node
can have multiple populations.
"""

struct InputDynamics
    dynamics::Vector{Equation}
    vars::VarSet
    params::ParamSet
end

struct OutputDynamics
    dynamics::Vector{Equation}
    vars::VarSet
    params::ParamSet
end

mutable struct PopBuildSetts
    sensory_conn_func::Vector{String}
    internode_conn_func::Vector{String}
    input2output_conn_func::String
    gets_sensory_input::Bool
    gets_internode_input::Bool
    sends_internode_output::Bool
    input_dynamics_spec::Vector{String}
    output_dynamics_spec::String
    noise_dynamics_spec::Vector{String}
    highest_var_idx::Int
    highest_param_idx::Int
end

mutable struct Population <: AbstractPopulation
    id::Int
    name::String
    parent_node::AbstractNode
    dynamics::Vector{Equation}
    vars::VarSet
    params::ParamSet
    n_state_vars::Int
    input_dynamics::Vector{InputDynamics}
    output_dynamics::Vector{OutputDynamics}
    noise_dynamics::String
    additive_noise_func::Union{Interpolations.Extrapolation, Nothing}
    build_setts::PopBuildSetts

    function Population(pop_id::Int, pop_name::String, parent_node::AbstractNode;
                        sends_internode_output::Bool=false,
                        sensory_conn_func::Union{String, Vector{String}}="none",
                        internode_conn_func::Union{String, Vector{String}}="none",
                        input_dynamics_spec::Union{String, Vector{String}}="none",
                        output_dynamics_spec::String="none",
                        input2output_conn_func::String="linear",
                        noise_dynamics_spec::Union{String, Vector{String}}="none",
                        noise_dynamics::String="")
        # Initialize
        dynamics = Vector{Equation}()
        vars = VarSet()
        params = ParamSet()
        n_state_vars = -1
        input_dynamics = Vector{InputDynamics}()
        output_dynamics = Vector{OutputDynamics}()
        additive_noise_func = nothing

        input_types_vec = isa(input_dynamics_spec, String) ? [input_dynamics_spec] : input_dynamics_spec

        # Normalize specs to vectors (short and consistent)
        sensory_vec   = normalize_spec(sensory_conn_func,   length(input_types_vec))
        internode_vec = normalize_spec(internode_conn_func, length(input_types_vec))
        noise_types_vec = normalize_spec(noise_dynamics_spec, length(input_types_vec))

        # Booleans derived from normalized vectors (any non-"none")
        gets_sensory_input   = any(s -> lowercase(s) != "none", sensory_vec)
        gets_internode_input = any(s -> lowercase(s) != "none", internode_vec)

        build_setts = PopBuildSetts(
            sensory_vec,
            internode_vec,
            input2output_conn_func,
            gets_sensory_input,
            gets_internode_input,
            sends_internode_output,
            input_types_vec,
            output_dynamics_spec,
            noise_types_vec,
            0,  # highest_var_idx
            0   # highest_param_idx
        )

        return new(pop_id, pop_name, parent_node, dynamics, vars, params, n_state_vars,
                   input_dynamics, output_dynamics, noise_dynamics, additive_noise_func, build_setts)
    end
end

###########################################################
## NODE STRUCTURE
############################################################

mutable struct NodeBuildSetts
    model::String
    n_pops::Int
    pop_models::Vector{String}
    pop_conn::Vector{String}
    pop_conn_motif::String
    new_state_var_inits::OrderedDict{String, Tuple{Float64, Float64}}
    new_param_values::OrderedDict{String, Tuple{Vararg{Float64}}}
    new_param_tunability::OrderedDict{String, Bool}
end

mutable struct Node <: AbstractNode
    id::Int
    name::String
    coords::Tuple{Float64, Float64, Float64}
    n_pops::Int
    dynamics::Vector{Equation}
    vars::VarSet
    params::ParamSet
    populations::Vector{Population}
    brain_source::String
    build_setts::NodeBuildSetts

    function Node(id::Int, name::String, model::String;
                  node_coordinates::Tuple{Float64, Float64, Float64} = (0.0, 0.0, 0.0),
                  pop_conn::Vector{String}=["none"])::Node
        dynamics = Vector{Equation}()
        vars = VarSet()
        params = ParamSet()
        populations = Vector{Population}()
        n_pops = 0
        brain_source = ""
        build_setts = NodeBuildSetts(
            model,
            0,
            String[],
            pop_conn,
            "",
            OrderedDict{String, Tuple{Float64, Float64}}(),
            OrderedDict{String, Tuple{Vararg{Float64}}}(),
            OrderedDict{String, Bool}()
        )
        return new(id, name, node_coordinates, n_pops, dynamics, vars, params, populations, brain_source, build_setts)
    end
end

function get_pop_by_id(node::Node, pop_id::Int)::Population
    for pop in node.populations
        if pop.id == pop_id
            return pop
        end
    end
    throw(KeyError("Population with ID $pop_id not found in node $(node.name)"))
end

###########################################################
## NETWORK STRUCTURE
############################################################

mutable struct Network
    name::String
    dynamics::Vector{Equation}
    diffusion_dynamics::Vector{Equation}
    vars::VarSet
    params::ParamSet
    problem::SciMLBase.AbstractDEProblem
    nodes::Vector{Node}
    conn::Matrix{Float64}
    sensory_input_conn::Vector{Bool}
    sensory_input_str::String
    sensory_input_func::Union{Interpolations.Extrapolation, Nothing}
    sensory_randomness::Bool
    internode_conn_eqs::Vector{Equation}
    internode_conn_params::ParamSet
    signature::String
    hash::String
    settings::Settings

    function Network(settings::Settings;)
        network_settings = settings.network_settings
        network_name = network_settings.network_name
        network_conn = network_settings.network_conn
        sensory_input_conn = network_settings.sensory_input_conn
        sensory_input_func_string = network_settings.sensory_input_func

        sensory_input = nothing
        internode_conn_eqs = Vector{Equation}()
        internode_conn_params = ParamSet()
        nodes = Vector{Node}()
        dynamics = Vector{Equation}()
        diffusion_dynamics = Vector{Equation}()
        vars = VarSet()
        params = ParamSet()
        problem = ODEProblem((du,u,p,t)->du .= 0, [0.0], (0.0,1.0)) # dummy ODEProblem
        sensory_input_randomness = occursin(r"rand\s*\(.*\)", sensory_input_func_string) || 
                                   occursin(r"randn\s*\(.*\)", sensory_input_func_string)       

        new(network_name, dynamics, diffusion_dynamics, vars, params, problem, 
            nodes, network_conn, sensory_input_conn, sensory_input_func_string, sensory_input, sensory_input_randomness,
            internode_conn_eqs, internode_conn_params,
            "", "", settings)
    end
end

function get_sensory_input(net::Network, time::Vector{Float64})::DataFrame
    if net.sensory_input_func === nothing
        return DataFrame()
    else
        sensory_values = [net.sensory_input_func(t) for t in time]
        df_input = DataFrame(time=time, sensory_input=sensory_values)

        return df_input
    end
end

function get_node_by_nodeid(net::Network, id::Int)::Node
    for node in net.nodes
        if node.id == id
            return node
        end
    end
    throw(KeyError("Node with ID $id not found in network $(net.name)"))
end


"""
Return true if any population defines additive random noise via its noise_dynamics string.
"""
function has_random_additive_noise(net::Network)::Bool
    for node in net.nodes
        for pop in node.populations
            for nds in pop.build_setts.noise_dynamics_spec
                if !isempty(nds) && lowercase(nds) == "additive" && !isempty(pop.noise_dynamics)
                    nd = pop.noise_dynamics
                    if occursin(r"\brandn\s*\(", nd) || occursin(r"\brand\s*\(", nd)
                        return true
                    end
                end
            end
        end
    end
    return false
end


###########################################################
# ── Compact summaries and pretty printing for node/pop/dynamics ──

function Base.summary(io::IO, n::Node)
    print(io, "Node($(n.id), $(n.name))")
end

function Base.show(io::IO, n::Node)
    print(io, "Node($(n.id), $(n.name); pops=$(length(n.populations)))")
end

function Base.show(io::IO, ::MIME"text/plain", n::Node)
    println(io, "Node")
    println(io, "  id: ", n.id, "  name: ", n.name)
    println(io, "  coords: ", n.coords)
    print(io,   "  n_pops: ", length(n.populations))
end

function Base.summary(io::IO, p::Population)
    print(io, "Population($(p.id), $(p.name))")
end

function Base.show(io::IO, p::Population)
    print(io,
        "Population($(p.id), $(p.name); node=$(p.parent_node.name)#$(p.parent_node.id), ",
        "states=$(p.n_state_vars))"
    )
end

function Base.show(io::IO, ::MIME"text/plain", p::Population)
    println(io, "Population")
    println(io, "  id: ", p.id, "  name: ", p.name)
    println(io, "  parent node: ", p.parent_node.name, "(", p.parent_node.id, ")")
    println(io, "  n_state_vars: ", p.n_state_vars)
    println(io, "  input dynamics: ", length(p.input_dynamics))
    print(io,   "  output dynamics: ", length(p.output_dynamics))
end


function Base.show(io::IO, idyn::InputDynamics)
    print(io,
        "InputDynamics(",
        "vars=$(length(idyn.vars.vars)), params=$(length(idyn.params.params)))"
    )
end

function Base.show(io::IO, ::MIME"text/plain", idyn::InputDynamics)
    println(io, "InputDynamics")
    println(io, "  vars: ", length(idyn.vars.vars))
    print(io,   "  params: ", length(idyn.params.params))
end

# Utility: normalize a String or Vector{String} to a Vector{String}.
# If given "none", return a vector filled with "none" of length n.
# Case-insensitive match for "none".
normalize_spec(spec::Union{String, Vector{String}}, n::Int)::Vector{String} =
    spec isa String ?
        (lowercase(spec) == "none" ? fill("none", n) : vcat([spec], fill("none", n - 1))) :
        String.(spec)