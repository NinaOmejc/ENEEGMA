# Example 4c: Two-Node Network Optimization with Synthetic Data
# ============================================================================
# This example demonstrates how to:
# 1. Load 2-node network settings from JSON
# 2. Simulate the network with known parameters to generate synthetic data
# 3. Add measurement noise to the simulated source time series
# 4. Save the noisy data as target data for optimization
# 5. Run parameter optimization with random initialization
# 6. Compare optimized parameters with ground truth

using ENEEGMA
using CSV
using DataFrames
using JSON
using Random
using Statistics

println("\nExample 4c: Two-Node Network Optimization with Synthetic Data\n")

# ============================================================================
# Step 1: Load 2-Node Settings from JSON
# ============================================================================
println("\n[Step 1] Loading 2-node settings from JSON...")
settings_path = joinpath("examples", "example_settings_2nodes.json")
settings = load_settings(settings_path)

# ============================================================================
# Step 2: Build and Simulate the Network
# ============================================================================
println("\n[Step 2] Building network...")
net = build_network(settings)
print_params_summary(net.params)

# Resample initial conditions
println("\n[Step 2b] Sampling initial conditions...")
new_inits = sample_inits(net.vars)

println("\n[Step 2c] Simulating network with ground truth parameters...")
df_true = simulate_network(net; new_inits=new_inits)
simulation_path = save_simulation_results(df_true, net, settings)

# ============================================================================
# Step 3: Prepare Data
# ============================================================================
settings.network_settings.node_names = ["N1", "N2"]
settings.data_settings.data_file = "$(simulation_path)\\$(settings.general_settings.exp_name)_$(net.name)_simulated.csv"
settings.data_settings.target_channel = Dict(
    "N1" => "$(settings.network_settings.node_names[1])₊source", 
    "N2" => "$(settings.network_settings.node_names[2])₊source"
    )
settings.data_settings.spectral_roi_manual = [(9., 14.)]
check_settings(settings)
data = prepare_data!(settings)

# ============================================================================
# Step 4: Optimization
# ============================================================================
set_all_params_tunable!(net.params)
settings.optimization_settings.maxiters = 10 # set low for testing; increase for better optimization
optsol, optlogger, setter, blocks = optimize_network(net, data, settings)

# ============================================================================
# Step 5: Simulate with new parameters and save results
# ============================================================================
net_optim = deepcopy(net)
settings.network_settings.name *= "_optimized"
net_optim.name *= "_optimized"

update_param_defaults!(net_optim, optsol);
print_params_summary(net_optim.params)
df4 = simulate_network(net_optim; settings=settings);
save_simulation_results(df4, net_optim, settings)

# ============================================================================
# Step 6: Compare with Ground Truth (Optional)
# ============================================================================

println("\nGround Truth Parameters (from simulation):")
print_params_summary(net.params)

println("\nOptimized Parameters:")
print_params_summary(net_optim.params)

println("\nExample 4c completed successfully!\n")
