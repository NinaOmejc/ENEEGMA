# Example 4: Parameter Optimization with Grammar-Sampled Models
# ===============================================================
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

# ============================================================================
# Step 2: Configure Data Settings
# ============================================================================
# Data file defaults to examples/example_data_rest.csv with IC3 channel
# The example data uses fs=256 and task_type="rest"
# You can override these defaults, e.g.
# settings.data_settings.data_file = "path/to/your/data.csv"
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

println("Building network with candidate: $(model_name)")

# Set the model in network settings (parse_tree can be used directly as String)
settings.network_settings.node_models = [parse_tree]
settings.general_settings.candidate_name = model_name

# ============================================================================
# Step 5: Build Network
# ============================================================================
net = build_network(settings);

# Make all parameters tunable for optimization
set_all_params_tunable!(net.params);

# Alternatively, you can specify which parameters to tune, e.g.
# set_params_tunable!(net.params, ["J", "tau_e", "tau_i"])

# Additionaly, you can modify the parameter default values or bounds here if desired, e.g.
# update_param_default_value!(net.params, "J", 1.0)
# update_param_minmax!(net.params, "J", 0.5, 2.0)

# Additionally, you can modify initial values or bounds for state variables if desired, e.g.
# update_var_inits!(net.vars, "E", 0.1, 0.5)

# ============================================================================
# Step 7: Run Optimization (commented - uncomment when ready)
# ============================================================================
print_settings_summary(settings; section="optimization_settings")
settings.optimization_settings.time_limit_minutes = 1 
optsol, optlogger, setter, blocks = optimize_network(net, data, settings)

