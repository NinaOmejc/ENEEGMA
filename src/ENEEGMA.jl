"""
    ENEEGMA

A Julia package for grammar-based sampling, construction, and optimization of brain network models coupled to EEG data.

## Key Features
- **Grammar-based model sampling**: Generate diverse model structures from formal grammars
- **Network construction**: Build multi-node networks with flexible coupling
- **Simulation framework**: Solve coupled ODEs efficiently
- **Parameter optimization**: Fit model parameters to real EEG data

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
using Optimization, OptimizationEvolutionary
    
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
# GRAMMAR-BASED SAMPLING (included before types so RuleTree is available)
# ============================================================================

include("grammar/grammar.jl")
include("grammar/grammar_utils.jl")

export Grammar, GrammarRule
export sample_from_grammar, export_grammar, save_grammar, load_grammar
export sample_rule, sample_pop_connectivity, terminals2rules, save_parse_trees
export list_rules, normalize_rule_groups!, ensure_rule_ids!, assign_rule_ids!
export rule_id_map

# ============================================================================
# TYPE DEFINITIONS
# ============================================================================

include("types/abstract_types.jl")
include("types/settings.jl")
include("types/variables.jl")
include("types/params.jl")
include("types/network_types.jl")
include("types/data.jl")
include("types/optimization_types.jl")

export Settings, GeneralSettings, NetworkSettings, SimulationSettings, OptimizationSettings
export OptimizerSettings, HyperparameterSweepSettings, HyperparameterAxis, SamplingSettings, DataSettings, LossSettings
export Population, InputDynamics, SensoryInput, InternodeInput, InterpopulationInput
export Data, NodeData, OptLogEntry, ReparamSpec
export ParamTransform, ParamReparamTransform, Affine01, ExpPos, SoftplusPos, SigmoidBound, TanhBound, Identity

# Type management
export update_param_minmax!, update_param_values!, update_param_tunability!
export update_param_defaults!, update_param_bounds!
export get_param_minmax_values, get_param_default_values, get_param_tunability, get_param_type
export sample_param_values, MIN_PARAM_VAL, MAX_PARAM_VAL
export print_params_summary, set_all_params_tunable!
export configure_network_parameters!

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
include("utils/spectral_transforms.jl")
include("utils/visualization.jl")

export set_verbose, vwarn, vinfo, verror, center
export make_rng, haspropnn, get_eeg_signal, set_task_settings, is_verbose
export create_default_settings, load_settings_from_file, save_settings, load_settings, check_settings
export settings_to_dict, print_settings_summary, load_data, normalize_parameter_name

# Signal processing utilities
export extract_brain_sources, extract_brain_source
export WelchWorkspace, SpectrumWorkspace
export compute_cwt, compute_stft
export compute_welch_psd, compute_preprocessed_welch_psd, compute_noisy_preprocessed_welch_psd
export parse_psd_preproc_pipeline, psd_preproc_flags_from_spec
export normalize_spectrum, normalize_spectra

# Visualization utilities
export plot_psd_single, plot_psd_comparison
export plot_timeseries_windows, plot_simulation_results
export find_next_numbered_folder, construct_output_dir, write_compact_json

# ============================================================================
# MODEL BUILDING
# ============================================================================

include("build/canonical_node_models.jl")
include("build/build_node.jl")
include("build/build_population.jl")
include("build/build_network.jl")
include("build/build_utils.jl")
include("build/connectivity_functions.jl")
include("build/connectivity_motifs.jl")
include("build/input_dynamics.jl")
include("build/output_dynamics.jl")

export rebuild_network_problem!

export build_population, build_populations
export build_network, build_node
export create_input_dynamics, create_interneuron_input
export apply_connectivity_motif
export list_canonical_node_models, list_canonical_node_models_codes, get_canonical_node_model_info!
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
include("simulate/save_simulation_results.jl")

export simulate_network
export prepare_ode_problem, solve_ode_system
export simulate_network, safe_solve, get_solver, get_solver_kwargs
export solver_needs_dt, sol2df, save_params_and_inits, save_ts_data
export save_simulation_results

# ============================================================================
# OPTIMIZATION
# ============================================================================

include("optimize/data_preparation.jl")
include("optimize/losses.jl")
include("optimize/optimization_utils.jl")
include("optimize/reparametrization.jl")
include("optimize/optimize_network.jl")
include("optimize/save_optimization_results.jl")
include("optimize/evaluation.jl")
include("optimize/hyperparameter_sweep.jl")

export compute_loss, compute_loss_from_simulation
export optimize_network, setup_optimization_problem
export ObjectiveFunction
export prepare_data!
export get_metric_function, get_loss_function
export apply_subject_specific_peak_range!
export detect_peak_windows, build_broad_peak_metadata
export estimate_sigma_init, estimate_sigma_floor
export maybe_initialize_std_measured_noise!
export save_optimization_results
export run_hyperparameter_sweep, show_hyperparameter_combos, add_hyperparameter_axis!

end  # module ENEEGMA
