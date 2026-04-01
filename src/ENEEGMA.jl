"""
    ENEEGMA

A Julia package for grammar-based sampling, construction, and optimization of brain network models coupled to EEG data.

## Key Features
- **Grammar-based model sampling**: Generate diverse model structures from formal grammars
- **Network construction**: Build multi-node networks with flexible coupling
- **Simulation framework**: Solve coupled ODEs efficiently
- **Parameter optimization**: Fit model parameters to real EEG data

## Quick Start

```julia
using ENEEGMA

# Create settings with defaults
settings_dict = create_default_settings()

# Sample a model from grammar
grammar = load_grammar()
model_name = sample_from_grammar(grammar)

# Build and simulate network
result = simulate_network(settings_dict)

# Optimize parameters
opt_result = optimize_network(settings_dict, target_data)
```

See `examples/` directory for working examples.
"""
module ENEEGMA

# --- Standard Packages ---
using DataFrames, CSV, JSON, DataStructures
using Dates, Random, LinearAlgebra
using Reexport
using Printf

# --- Core Modeling Packages ---
using Symbolics, SymbolicIndexingInterface
using DifferentialEquations
using OrdinaryDiffEq

# --- Optimization Packages ---
using Optimization, OptimizationOptimJL

# --- Signal Processing ---
using Statistics, Distributions, FFTW

# --- Plotting (optional) ---
using Plots

# --- Logging ---
using Logging
const QUIET_SOLVER_LOGGER = Logging.SimpleLogger(stderr, Logging.Error)

# --- Differential Equation Context ---
@variables t
D = Differential(t)

# ============================================================================
# TYPE DEFINITIONS
# ============================================================================

include("types/abstract_types.jl")
include("types/settings.jl")
include("types/params.jl")
include("types/variables.jl")
include("types/network_types.jl")

export Settings, GeneralSettings, NetworkSettings, SimulationSettings, OptimizationSettings
export OptimizerSettings, SamplingSettings, DataSettings, LossSettings
export Population, InputDynamics, SensoryInput, InternodeInput, InterpopulationInput

# Type management
export update_param_minmax!, update_param_values!, update_param_tunability!
export get_param_minmax_values, get_param_default_values, get_param_tunability, get_param_type
export sample_param_values, needs_tscale, MIN_PARAM_VAL, MAX_PARAM_VAL

# ============================================================================
# UTILITIES
# ============================================================================

include("utils/utils.jl")
include("utils/settings_manager.jl")
include("utils/defaults.jl")

export set_verbose, vwarn, vinfo, vprint, center
export make_rng, haspropnn
export create_default_settings, load_settings_from_file, save_settings_to_json
export settings_to_dict, print_settings_summary

# ============================================================================
# GRAMMAR-BASED SAMPLING
# ============================================================================

include("grammar/grammar.jl")
include("grammar/grammar_utils.jl")

export Grammar, GrammarRule
export sample_from_grammar, export_grammar, save_grammar, load_grammar
export sample_rule, sample_pop_connectivity, terminals2rules

# ============================================================================
# MODEL BUILDING
# ============================================================================

include("build/build_node.jl")
include("build/build_population.jl")
include("build/build_network.jl")
include("build/build_utils.jl")
include("build/connectivity_functions.jl")
include("build/connectivity_motifs.jl")
include("build/input_dynamics.jl")
include("build/output_dynamics.jl")

export build_population, build_populations
export build_network, build_node
export create_input_dynamics, create_interneuron_input
export apply_connectivity_motif

# ============================================================================
# SIMULATION
# ============================================================================

include("simulate/simulate_node.jl")
include("simulate/simulate_network.jl")

export simulate_network, simulate_node
export prepare_ode_problem, solve_ode_system

# ============================================================================
# OPTIMIZATION
# ============================================================================

include("optimize/losses.jl")
include("optimize/optimization_utils.jl")
include("optimize/optimize_network.jl")

export compute_loss, compute_loss_from_simulation
export optimize_network, setup_optimization_problem
export ObjectiveFunction

end  # module ENEEGMA
