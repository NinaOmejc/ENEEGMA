# Example 3: Parameter Optimization Workflow
# ============================================
# This example demonstrates how to set up and run parameter optimization
# to fit a network model to target EEG data.

using ENEEGMA
using JSON

println("="^70)
println("ENEEGMA Example 3: Parameter Optimization")
println("="^70)

# ============================================================================
# Step 1: Create and Configure Settings for Optimization
# ============================================================================
println("\n[Step 1] Creating settings configured for optimization...")

settings_dict = create_default_settings(
    exp_name="OptimizationExample_2Node",
    n_nodes=2,
    tspan=(0.0, 500.0),
    dt=1.0  # 1 kHz sampling for optimization
)

# Set unified output directory
output_dir = "./eneegma_example_outputs"
settings_dict["general_settings"]["path_out"] = output_dir

# Key optimization-specific configuration
os = settings_dict["optimization_settings"]
os["method"] = "CMAES"  # Only CMAES is supported
ls = os["loss_settings"]

# Configure loss function for power spectral density fitting
ls["loss_fn"] = "psd_iae"                   # Integrated Absolute Error on PSD
ls["freq_bands"] = [1.0, 50.0]              # Focus on 1-50 Hz band
ls["psd_window_size"] = 4                   # Welsh window size
ls["psd_poly_order"] = 2                    # Detrending polynomial order
ls["psd_smooth_sigma"] = 0.5                # Smoothing strength

# Configure optimizer (CMAES only)
optset = os["optimizer_settings"]
optset["population_size"] = 50              # Population size for CMAES
optset["sigma0"] = -1.0                     # Automatic sigma initialization

# Data configuration (would point to real EEG data)
settings_dict["data_settings"]["data_path"] = ""  # User should provide path
settings_dict["data_settings"]["target_channel"] = "IC3"  # Which component to fit

println("✓ Optimization settings configured:")
println("  - Loss: $(ls["loss_fn"]) ($(join(skipmissing([ls["freq_bands"]...]), "-")) Hz)")
println("  - Method: $(os["method"])")
println("  - Population size: $(optset["population_size"])")

# ============================================================================
# Step 2: Save Configuration
# ============================================================================
println("\n[Step 2] Saving optimization configuration...")

isdir(output_dir) || mkpath(output_dir)

settings_path = joinpath(output_dir, "example3_settings.json")
save_settings(settings_dict, settings_path)
println("✓ Configuration saved to: $settings_path")

# ============================================================================
# Step 3: Load and Prepare for Optimization
# ============================================================================
println("\n[Step 3] Loading settings and preparing optimization environment...")

settings = load_settings_file(settings_path)

# Show what would be optimized
println("\n  Network to be optimized:")
println("    Name: $(settings.general_settings.exp_name)")
println("    Nodes: $(join(settings.network_settings.node_names, ", "))")
println("    Models: $(join(settings.network_settings.node_models, ", "))")

println("\n  Optimization will:")
println("    • Vary node parameters to minimize fit error")
println("    • Use $(settings.optimization_settings.method) with population size $(settings.optimization_settings.optimizer_settings.population_size)")
println("    • Track loss: $(settings.optimization_settings.loss)")

# ============================================================================
# Step 4: Document Optimization Protocol
# ============================================================================
println("\n[Step 4] Documenting optimization protocol...")

protocol_doc = Dict(
    "optimization_protocol" => Dict(
        "objective" => "Fit network model parameters to EEG spectral data",
        "loss_function" => "Power spectral density IAE (Integrated Absolute Error)",
        "frequency_range_hz" => [settings.optimization_settings.loss_settings.fmin, settings.optimization_settings.loss_settings.fmax],
        "target_data" => "User-provided EEG data (source-localized component)",
        "optimization_algorithm" => "CMAES (Covariance Matrix Adaptation Evolution Strategy)",
        "hyperparameters" => Dict(
            "population_size" => settings.optimization_settings.optimizer_settings.population_size,
            "sigma0" => settings.optimization_settings.optimizer_settings.sigma0
        ),
        "tunable_parameters" => "All node model parameters (specific to model structure)",
        "output_includes" => [
            "Optimized parameter values",
            "Loss trajectory over iterations",
            "Before/after spectral comparison",
            "Complete settings.json for reproducibility"
        ]
    ),
    "settings" => settings_dict
)

protocol_path = joinpath(output_dir, "example3_optimization_protocol.json")
open(protocol_path, "w") do f
    JSON.print(f, protocol_doc, 2)
end

println("✓ Protocol documented in: $protocol_path")

# ============================================================================
# Step 5: Show What a Real Optimization Produces
# ============================================================================
println("\n[Step 5] Expected optimization outputs...")

expected_outputs = Dict(
    "optimization_results" => Dict(
        "converged" => "boolean",
        "final_loss" => "float (lower is better)",
        "n_iterations" => "int",
        "optimization_time_seconds" => "float",
        "final_parameters" => "Dict of all optimized values",
        "parameter_changes" => "Dict showing param changes from init to final",
        "loss_trajectory" => "Vector of loss values per iteration"
    ),
    "diagnostics" => Dict(
        "initial_loss" => "float (starting fit error)",
        "final_loss" => "float (ending fit error)",
        "improvement_percent" => "float",
        "loss_reduction_per_iteration" => "float"
    )
)

results_path = joinpath(output_dir, "example3_expected_outputs.json")
open(results_path, "w") do f
    JSON.print(f, expected_outputs, 2)
end

println("✓ Expected output format saved to: $results_path")

# ========================================================================
# Step 6: Tips for Real Optimization
# ========================================================================
println("\n" * "="^70)
println("Tips for Running Actual Optimization")
println("="^70)
println("""
To run real parameter optimization:

1. Prepare your data:
   - Have source-localized EEG component (or simulated target PSD)
   - Format as CSV compatible with ENEEGMA data loaders
   - Path should be in data_settings['data_path']

2. Set model constraints:
   - Specify which parameters are tunable vs fixed
   - Set bounds on parameter ranges
   - Choose initialization strategy

3. Run optimization:
   settings = load_settings_file("$settings_path")
   target_data = load_target_data(settings)
   network = build_network(settings)
   result = optimize_network(network, target_data, settings)

4. Analyze results:
   - Check loss_trajectory for convergence
   - Compare before/after spectral fits
   - Examine optimized parameter values
   - Verify biological plausibility

5. Save reproducibly:
   - Results include settings.json
   - All hyperparameters, data paths documented
   - Can be rerun with identical configuration
""")

# ============================================================================
# Summary
# ============================================================================
println("\n" * "="^70)
println("Example 3 Complete!")
println("="^70)
println("""
You have learned how to:
  ✓ Configure optimization settings (loss function, optimizer, hyperparams)
  ✓ Document optimization protocol for reproducibility
  ✓ Understand the expected outputs and format
  ✓ Know what steps to take to run real optimization

Configuration files created:
  1. $settings_path
  2. $protocol_path
  3. $results_path

These files demonstrate best practices for:
  • Reproducible research
  • Complete documentation
  • Computational tracking
  • Sharing configurations

Next: Read the examples together to understand the full ENEEGMA workflow!
""")
println("="^70)
