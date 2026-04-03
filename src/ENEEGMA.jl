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
using Latexify
using DifferentialEquations
using OrdinaryDiffEq

# --- Optimization Packages ---
using Optimization, OptimizationOptimJL
using Evolutionary

# --- Signal Processing ---
using Statistics, Distributions, FFTW
using Interpolations

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
export print_params_summary

# Variable management
export StateVar, ExtraVar, Var, VarSet
export copy_var, add_var!, join_varsets, join_varsets!
export get_var_by_name, get_vars_by_name, get_var_by_symbol, get_vars_by_type
export get_vars_sending_internode_output, get_vars_getting_internode_input
export get_vars_getting_interpop_input, get_vars_sending_interpop_output, get_vars_sending_intrapop_output
export get_sensory_input_var, get_history_vars, get_state_vars
export get_vars_by_nodeid, get_vars_by_eq_idx, get_symbols
export sample_inits, update_var_inits!, get_var_mean_inits, get_var_minmax_values
export get_postfix_index, var_in_varset, get_highest_postfix_index
export print_vars_summary

# ============================================================================
# UTILITIES
# ============================================================================

include("utils/utils.jl")
include("utils/io.jl")
include("utils/extract_brain_source.jl")
include("utils/separate_psd_comps.jl")
include("utils/spectral_transforms.jl")
include("utils/smooth_norm.jl")

export set_verbose, vwarn, vinfo, vprint, center
export make_rng, haspropnn, set_task_settings, is_verbose
export create_default_settings, load_settings_from_file, save_settings, load_settings_file
export settings_to_dict, print_settings_summary, load_data

# Signal processing utilities
export extract_brain_sources, extract_brain_source
export separate_psd_components
export WelchWorkspace, SpectrumWorkspace
export calculate_spectra, calculate_spectrum
export compute_cwt, compute_stft, compute_welch_pow_spectrum, compute_welch_pow_spectra
export parse_psd_preproc_pipeline, psd_preproc_flags_from_spec
export smooth_vector, smooth_df
export normalize_timeseries, normalize_timeseries_df
export detrend_vector, detrend_df
export normalize_spectrum, normalize_spectra

# ============================================================================
# GRAMMAR-BASED SAMPLING
# ============================================================================

include("grammar/grammar.jl")
include("grammar/grammar_utils.jl")

export Grammar, GrammarRule
export sample_from_grammar, export_grammar, save_grammar, load_grammar
export sample_rule, sample_pop_connectivity, terminals2rules
export list_rules, normalize_rule_groups!, ensure_rule_ids!, assign_rule_ids!
export rule_id_map

# ============================================================================
# MODEL BUILDING
# ============================================================================

include("build/known_node_models.jl")
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
export list_known_node_models, list_known_node_models_codes, get_known_node_model_info!
export update_network_parameters!, construct_network_problem!
export set_network_signature!, export_network
export transform2latex, string2num, string2symbolicfun, soft_wrap, sort_symbols
export get_pop_by_id, get_sensory_input, get_node_by_nodeid, has_random_additive_noise
export get_conn_funcs, get_input_dynamics, get_output_dynamics
export add_conn_motif, add_conn_motif_builder, add_random_conn_motif

# ============================================================================
# SIMULATION
# ============================================================================

include("simulate/simulate_network.jl")

export simulate_network
export prepare_ode_problem, solve_ode_system
export simulate_problem, safe_solve, get_solver, get_solver_kwargs
export solver_needs_dt, sol2df, save_params_and_inits, save_ts_data

# ============================================================================
# OPTIMIZATION
# ============================================================================

include("optimize/target_preparation.jl")
include("optimize/losses.jl")
include("optimize/optimization_utils.jl")
include("optimize/reparametrization.jl")
include("optimize/optimize_network.jl")

export compute_loss, compute_loss_from_simulation
export optimize_network, setup_optimization_problem
export ObjectiveFunction
export TargetPSD, prepare_target!, ReparamSpec
export get_metric_function, get_loss_function
export apply_subject_specific_peak_range!
export detect_peak_windows, build_broad_peak_metadata
export estimate_sigma_init, estimate_sigma_floor
export maybe_initialize_std_measured_noise!
export findpeaks, compute_peak_score, compute_background_score

end  # module ENEEGMA
