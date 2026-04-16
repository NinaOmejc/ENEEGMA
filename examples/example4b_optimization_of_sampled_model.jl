# Example 4b: Parameter Optimization of Sampled Model
# ============================================================================
# This example demonstrates how to:
# 1. Load target data and prepare for optimization
# 2. Load candidate models from grammar sampling results
# 3. Select and build a network from one candidate
# 4. Optimize network parameters to fit target data

using Revise
using ENEEGMA
using CSV
using DataFrames
using JSON

# ============================================================================
# Step 1: Create or Load Settings
# ============================================================================
settings = create_default_settings();
# print_settings_summary(settings; section="general_settings")

# ============================================================================
# Step 2: Configure Data Settings
# ============================================================================
print_settings_summary(settings; section="data_settings")
data = prepare_data!(settings);

# ============================================================================
# Step 3: Load candidate models from grammar sampling CSV
# ============================================================================
# Load the parse trees from grammar sampling results
models_file = joinpath("examples", "example_grammar_parse_trees.csv")
candidate_models = CSV.read(models_file, DataFrame)

println("\nLoaded $(nrow(candidate_models)) candidate models")

# ============================================================================
# Step 4: Select first model and apply to settings
# ============================================================================
# Get the first candidate (G1)
first_model_idx = 1
model_name = candidate_models.model_name[first_model_idx]
parse_tree = candidate_models.parse_tree[first_model_idx]

# Set the model in network settings (parse_tree can be used directly as String)
settings.network_settings.node_models = [parse_tree]
settings.network_settings.name = model_name

# ============================================================================
# Step 5: Build Network and Configure Optimization
# ============================================================================

net = build_network(settings);

# Make all parameters tunable
set_all_params_tunable!(net.params)

# Update parameter tunability (selective)
# new_tunability = Dict("N1₊c11" => true, "N1₊c12" => false)
# update_param_tunability!(net.params, new_tunability)

# Update default values
# new_param_defaults = Dict("N1₊c11" => 0.5, "N1₊c12" => 0.3)
# update_param_defaults!(net.params, new_param_defaults)

# Update parameter bounds
# new_param_bounds = Dict("N1₊c11" => (0.0, 1.0), "N1₊c12" => (0.0, 1.0))
# update_param_bounds!(net.params, new_param_bounds)

# Update initial condition bounds for state variables
# new_inits_bounds = Dict("N1₊x11" => (0.1, 0.5), "N1₊x12" => (0.2, 0.6))
# update_var_inits!(net.vars, new_inits_bounds)

# ============================================================================
# Step 6: Run Optimization (commented - uncomment when ready)
# ============================================================================
print_settings_summary(settings; section="optimization_settings")
settings.optimization_settings.time_limit_minutes = 1
settings.optimization_settings.n_restarts = 2
optsol, optlogger, setter, blocks = optimize_network(net, data, settings)

