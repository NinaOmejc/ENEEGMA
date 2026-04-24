# Example 5: Hyperparameter Optimization
# ============================================================================
# This example demonstrates systematic hyperparameter optimization:
# 1. Load target data and configure neural network
# 2. Define hyperparameter sweep axes
# 3. Review all sweep combinations
# 4. Run optimization across the parameter space

using ENEEGMA
using CSV
using DataFrames
using JSON

# ============================================================================
# Step 1: Load Settings and Data and build network as in previous examples
# ============================================================================
println("Step 1: Loading settings and data...")
settings = create_default_settings()

# Load and prepare data
print_settings_summary(settings; section="data_settings")
data = prepare_data!(settings)

# Build and configure network
net = build_network(settings)
set_all_params_tunable!(net.params)

# ============================================================================
# Step 2: Configure Hyperparameter Sweep
# ============================================================================

# DEFAULT SWEEP AXES (automatically included):
#   • param_bound_scaling_level: ["medium", "high"]
#   • sigma0: [-1.0, 2.0, 8.0]  (where -1.0 means use CMAES default)
#   • population_size: [-1, 100, 150]  (where -1 means use CMAES default)
#
# These provide 2 × 3 × 3 = 18 combinations

# Reset to defaults only (uncomment to clear custom axes):
# empty!(settings.optimization_settings.hyperparameter_sweep.hyperparameter_axes)

# ADD CUSTOM AXES using add_hyperparameter_axis!()
add_hyperparameter_axis!(settings, 
    "optimization_settings.loss_settings.bg_weight", 
    [0.5, 0.75, 1.0]
)

# ============================================================================
# Step 3: Review All Sweep Combinations
# ============================================================================
show_hyperparameter_combos(settings)

# Show specific combination (e.g., combination #3):
combo_idx = 16
show_hyperparameter_combos(settings; combo_idx=combo_idx)

# ============================================================================
# Step 4: Run Hyperparameter Sweep
# ============================================================================
settings.optimization_settings.maxiters = 10 # set low for testing; increase for better optimization

# OPTION A: Run full sweep (all combinations)
# run_hyperparameter_sweep(settings, data)

# OPTION B: Run specific combination (useful for debugging/testing)
run_hyperparameter_sweep(settings, data; combo_idx=combo_idx)

println("Example complete! Results saved to: $(settings.general_settings.path_out)")


