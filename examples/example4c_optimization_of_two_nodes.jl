# Example 4c: Two-Node Network Optimization with Synthetic Data
# ============================================================================
# This example demonstrates how to:
# 1. Load 2-node network settings from JSON
# 2. Simulate the network with known parameters to generate synthetic data
# 3. Add measurement noise to the simulated data
# 4. Save the noisy data as target data for optimization
# 5. Run parameter optimization with random initialization
# 6. Compare optimized parameters with ground truth

using ENEEGMA
using CSV
using DataFrames
using JSON
using Random
using Statistics

println("\n" * "="^80)
println("Example 4c: Two-Node Network Optimization with Synthetic Data")
println("="^80)

# ============================================================================
# Step 1: Load 2-Node Settings from JSON
# ============================================================================
println("\n[Step 1] Loading 2-node settings from JSON...")
settings_path = joinpath("examples", "example_settings_2nodes.json")
settings = load_settings(settings_path)

# Update general settings
settings.general_settings.exp_name = "exp_2nodes_optimization"
settings.general_settings.verbosity_level = 1

# Update simulation settings to get good quality data
settings.simulation_settings.tspan = (0.0, 60.0)  # 60 seconds of data
settings.simulation_settings.saveat = 0.002  # 500 Hz sampling rate

println("✓ Settings loaded:")
println("  - Network: $(settings.general_settings.exp_name)")
println("  - Nodes: $(settings.network_settings.node_names)")
println("  - Simulation time: $(settings.simulation_settings.tspan[2])s")
println("  - Sampling rate: $(1.0 / settings.simulation_settings.saveat) Hz")

# ============================================================================
# Step 2: Set Network Models (Both nodes use same canonical model)
# ============================================================================
println("\n[Step 2] Configuring network models...")
# Use Wilson-Cowan canonical model for both nodes
canonical_model = "WC"  # Wilson-Cowan model
settings.network_settings.node_models = [canonical_model, canonical_model]

println("✓ Network model configured:")
println("  - Model: $canonical_model")
println("  - Nodes: $(length(settings.network_settings.node_models)) nodes")

# ============================================================================
# Step 3: Build and Simulate the Network
# ============================================================================
println("\n[Step 3] Building network...")
net = build_network(settings)

println("✓ Network built successfully")
println("  - Parameters: $(length(net.params.params))")
println("  - Variables: $(length(net.vars.vars))")

# Set seed for reproducibility and resample initial conditions
println("\n[Step 3b] Sampling initial conditions...")
Random.seed!(12345)
new_inits = sample_inits(net.vars; seed=12345)
println("✓ Initial conditions sampled")

println("\n[Step 3c] Simulating network with ground truth parameters...")
df_true = simulate_network(net; new_inits=new_inits)
println("✓ Simulation completed")
println("  - Simulated time points: $(nrow(df_true))")

# ============================================================================
# Step 4: Extract Signals and Create Target Data
# ============================================================================
println("\n[Step 4] Extracting signals from simulation...")

# For 2-node network, we need separate signals for each node
# Extract brain source variables for each node
signal_1 = df_true[!, Symbol("N1₊x11")]  # First node's excitatory state
signal_2 = df_true[!, Symbol("N2₊x11")]  # Second node's excitatory state

times = df_true.time
fs = 1.0 / settings.simulation_settings.saveat

println("✓ Signals extracted:")
println("  - Node N1 signal length: $(length(signal_1))")
println("  - Node N2 signal length: $(length(signal_2))")
println("  - Sampling frequency: $fs Hz")

# ============================================================================
# Step 5: Compute Reference PSDs (Before Adding Noise)
# ============================================================================
println("\n[Step 5] Computing reference PSDs...")

# Compute PSDs for both signals
freqs_1, powers_1 = ENEEGMA.compute_preprocessed_welch_psd(
    signal_1, fs; data_settings=settings.data_settings
)
freqs_2, powers_2 = ENEEGMA.compute_preprocessed_welch_psd(
    signal_2, fs; data_settings=settings.data_settings
)

println("✓ Reference PSDs computed:")
println("  - Frequency resolution: $(round(freqs_1[2] - freqs_1[1], digits=3)) Hz")
println("  - Frequency range: [$(round(freqs_1[1], digits=1)), $(round(freqs_1[end], digits=1))] Hz")

# ============================================================================
# Step 6: Add Measurement Noise to Signals
# ============================================================================
println("\n[Step 6] Adding measurement noise to signals...")

# Define noise levels (as standard deviation)
noise_std_1 = 0.5  # Noise for Node 1
noise_std_2 = 0.3  # Noise for Node 2

# Add Gaussian noise
Random.seed!(54321)
noisy_signal_1 = signal_1 .+ noise_std_1 .* randn(length(signal_1))
noisy_signal_2 = signal_2 .+ noise_std_2 .* randn(length(signal_2))

# Compute PSDs for noisy signals
freqs_1_noisy, powers_1_noisy = ENEEGMA.compute_preprocessed_welch_psd(
    noisy_signal_1, fs; data_settings=settings.data_settings
)
freqs_2_noisy, powers_2_noisy = ENEEGMA.compute_preprocessed_welch_psd(
    noisy_signal_2, fs; data_settings=settings.data_settings
)

println("✓ Noise added:")
println("  - Node 1 noise std: $noise_std_1")
println("  - Node 2 noise std: $noise_std_2")
println("  - Signal-to-noise ratio N1: $(round(std(signal_1) / noise_std_1, digits=2))")
println("  - Signal-to-noise ratio N2: $(round(std(signal_2) / noise_std_2, digits=2))")

# ============================================================================
# Step 7: Create and Save Target Data CSV
# ============================================================================
println("\n[Step 7] Creating target data file...")

# Create results directory
results_dir = joinpath("results", "exp_2nodes_optimization", "target_data")
mkpath(results_dir)

# Save noisy signals to CSV
target_file = joinpath(results_dir, "target_data.csv")
target_df = DataFrame(
    time=times,
    signal_N1=noisy_signal_1,
    signal_N2=noisy_signal_2
)
CSV.write(target_file, target_df)

println("✓ Target data saved to: $target_file")
println("  - Rows: $(nrow(target_df))")
println("  - Columns: $(names(target_df))")

# ============================================================================
# Step 8: Configure Optimization Settings
# ============================================================================
println("\n[Step 8] Configuring optimization settings...")

# Create data structure for optimization (from noisy signals)
# Configure data settings to point to our simulated data
target_data_file = joinpath(results_dir, "target_data.csv")
settings.data_settings.data_file = target_data_file
settings.data_settings.target_channel = Dict("N1" => "signal_N1", "N2" => "signal_N2")

# Update node names to match
settings.network_settings.node_names = ["N1", "N2"]

# Validation
println("\nValidating settings...")
try
    check_settings(settings)
    println("✓ Settings validation passed")
catch e
    println("✗ Settings validation failed: $e")
    rethrow(e)
end

# ============================================================================
# Step 9: Prepare Data for Optimization
# ============================================================================
println("\n[Step 9] Preparing target data for optimization...")
data = prepare_data!(settings)

println("✓ Data prepared:")
println("  - Nodes in data: $(length(data.node_data))")
println("  - Node names: $(join(keys(data.node_data), ", "))")
for (node_name, node_data) in data.node_data
    println("  - $node_name: $(length(node_data.signal)) time samples, $(length(node_data.powers)) frequency bins")
end

# ============================================================================
# Step 10: Configure Network for Optimization
# ============================================================================
println("\n[Step 10] Configuring network for optimization...")

# Rebuild network for optimization
net_opt = build_network(settings)

# Make all parameters tunable for optimization
Random.seed!(99999)
set_all_params_tunable!(net_opt.params)

# Optionally: The network now has random initial conditions from build_network
# They will be used as the starting point for optimization

println("✓ Network configured for optimization:")
println("  - Tunable parameters: $(sum(1 for p in net_opt.params.params if p.tunable))")
println("  - Ready for optimization with per-node loss computation")

# ============================================================================
# Step 11: Update Optimization Settings
# ============================================================================
println("\n[Step 11] Setting optimization parameters...")

settings.optimization_settings.time_limit_minutes = 5  # Short run for demo
settings.optimization_settings.n_restarts = 2  # Multiple restarts for better convergence
settings.optimization_settings.maxiters = 1000

println("✓ Optimization settings updated:")
println("  - Time limit: $(settings.optimization_settings.time_limit_minutes) minutes")
println("  - Restarts: $(settings.optimization_settings.n_restarts)")
println("  - Max iterations/restart: $(settings.optimization_settings.maxiters)")

# ============================================================================
# Step 12: Run Optimization
# ============================================================================
println("\n[Step 12] Running optimization...")
println("-"^80)

optsol, optlogger, setter, blocks = optimize_network(net_opt, data, settings)

println("-"^80)
println("✓ Optimization completed")

# ============================================================================
# Step 13: Display Results
# ============================================================================
println("\n[Step 13] Optimization Results")
println("="^80)

if optsol !== nothing
    println("\n✓ Optimal solution found!")
    println("  - Final loss: $(round(optsol.objective, digits=6))")
    println("  - Termination reason: $(optsol.retcode)")
else
    println("\n⚠ Optimization did not produce a solution")
end

# ============================================================================
# Step 14: Compare with Ground Truth (Optional)
# ============================================================================
println("\n[Step 14] Comparison with Ground Truth Parameters")
println("="^80)

println("\nGround Truth Parameters (from simulation):")
for (i, param) in enumerate(net.params.params[1:min(10, length(net.params.params))])
    println("  $(param.name): $(round(param.default, digits=6))")
end
if length(net.params.params) > 10
    println("  ... and $(length(net.params.params) - 10) more parameters")
end

if optsol !== nothing && length(optsol.u) >= length(net_opt.params.params)
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

println("\n" * "="^80)
println("Example 4c completed successfully!")
println("="^80 * "\n")
