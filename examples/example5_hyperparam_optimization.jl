# Example 5: Hyperparameter Sweep with Grammar-Sampled Models
# ===========================================================
# This example demonstrates how to:
# 1. Load target data and prepare for optimization
# 2. Load candidate models from grammar sampling results
# 3. Select a specific model for optimization
# 4. Run a systematic hyperparameter sweep to find optimal configuration
using ENEEGMA
using CSV
using DataFrames
using JSON
using Revise 

# ============================================================================
# Step 1: Create Settings
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
# using Plots
# plot(data.times, data.signal, title="Target EEG Signal", xlabel="Time (s)", ylabel="Amplitude")
# plot(data.freqs, data.powers, title="Target PSD", xlabel="Frequency (Hz)", ylabel="Power")

# ============================================================================
# Step 3: Load candidate models from grammar sampling CSV
# ============================================================================
# Load the parse trees from grammar sampling results
models_file = joinpath("examples", "example_grammar_parse_trees.csv")
candidate_models = CSV.read(models_file, DataFrame)

println("\nLoaded $(nrow(candidate_models)) candidate models")

# ============================================================================
# Step 4: Select model and build network
# ============================================================================
# Select a model (change index to test different models)
model_idx = 1
model_name = candidate_models.model_name[model_idx]
parse_tree = candidate_models.parse_tree[model_idx]

println("Building network with candidate: $(model_name)")

# Set the model in network settings (parse_tree can be used directly as String)
settings.network_settings.node_models = [parse_tree]
settings.general_settings.candidate_name = model_name

# Build Network
net = build_network(settings)
println("Network built successfully")
println("System size: $(length(net.dynamics)) equations")

# Make all parameters tunable for optimization
set_all_params_tunable!(net.params)

# ============================================================================
# Step 5: Configure Hyperparameter Sweep
# ============================================================================
# Hyperparameter sweep settings are in optimization_settings.hyperparameter_sweep
# 
# Configuration options:
#   - population_grid: Vector of population sizes to test [e.g., [20, 40, 80]]
#   - sigma_grid: Vector of initial step sizes [e.g., [0.5, 1.0, 2.0]]
print_settings_summary(settings; section="optimization_settings")

# ============================================================================
# Step 6: Run Hyperparameter Sweep
# ============================================================================
# The sweep will run optimization for each hyperparameter combination
run_hyperparameter_sweep(settings, data)
