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

# Set seed for reproducibility and resample initial conditions
println("\n[Step 2b] Sampling initial conditions...")
Random.seed!(12345)
new_inits = sample_inits(net.vars; seed=12345)

println("\n[Step 2c] Simulating network with ground truth parameters...")
df_true = simulate_network(net; new_inits=new_inits)
simulation_path = save_simulation_results(df_true, net, settings)

# ============================================================================
# Step 3: Prepare Data
# ============================================================================
settings.data_settings.data_file = "$(simulation_path)\\$(settings.general_settings.exp_name)_$(net.name)_simulated.csv"
settings.network_settings.node_names = ["N1", "N2"]
settings.data_settings.target_channel = Dict(
    "N1" => "$(settings.network_settings.node_names[1])₊source", 
    "N2" => "$(settings.network_settings.node_names[2])₊source"
    )
check_settings(settings)

data = prepare_data!(settings)

# ============================================================================
# Step 4: Optimization
# ============================================================================
set_all_params_tunable!(net.params)
optsol, optlogger, setter, blocks = optimize_network(net, data, settings)

# ============================================================================
# Step 5: Compare with Ground Truth (Optional)
# ============================================================================

println("\nGround Truth Parameters (from simulation):")
for param in net.params.params[1:min(10, length(net.params.params))]
    println("  $(param.name): $(round(param.default, digits=6))")
end
if length(net.params.params) > 10
    println("  ... and $(length(net.params.params) - 10) more parameters")
end

if optsol !== nothing && length(optsol.u) >= length(net.params.params)
    println("\nOptimized Parameters (first 10):")
    opt_params = optsol.u[1:length(net_opt.params.params)]
    for (i, param) in enumerate(net_opt.params.params[1:min(10, length(opt_params))])
        opt_val = opt_params[i]
        gt_val = param.default
        error_pct = 100 * abs(opt_val - gt_val) / (abs(gt_val) + 1e-6)
        println("  $(param.name):")
        println("    Ground truth: $(round(gt_val, digits=6))")
        println("    Optimized:    $(round(opt_val, digits=6))")
        println("    Error:        $(round(error_pct, digits=2))%")
    end
    if length(opt_params) > 10
        println("  ... and $(length(opt_params) - 10) more parameters")
    end
end


println("\nExample 4c completed successfully!\n")
