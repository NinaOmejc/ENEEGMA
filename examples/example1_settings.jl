# Example 1: Settings Configuration Walkthrough
# ===============================================
# This example demonstrates how to create, customize, and save ENEEGMA settings
# All settings have sensible defaults that you can override as needed.

using ENEEGMA
using JSON

println("="^70)
println("ENEEGMA Example 1: Settings Configuration")
println("="^70)

# ============================================================================
# Step 1: Create Settings with Defaults
# ============================================================================
println("\n[Step 1] Creating default settings for a 2-node network...")

settings_dict = create_default_settings(
    network_name="Example1_TwoNode",
    n_nodes=2,
    tspan=(0.0, 1000.0),
    dt=0.1
)

# Print a summary of the created settings
print_settings_summary(settings_dict; verbosity=1)

# ============================================================================
# Step 2: Customize Network Settings
# ============================================================================
println("\n[Step 2] Customizing network configuration...")

# Access and modify specific settings
settings_dict["general_settings"]["verbose"] = true
settings_dict["general_settings"]["verbosity_level"] = 1

# Modify network connectivity
ns = settings_dict["network_settings"]
ns["node_names"] = ["Region_A", "Region_B"]
ns["node_models"] = ["JansenRit", "JansenRit"]  # Use known models, or "Unknown" for grammar-sampled

# Set up connectivity: unidirectional A -> B
ns["network_conn"][1, 2] = 0.15  # A to B
ns["network_conn"][2, 1] = 0.0   # B to A isolated
ns["network_delay"][1, 2] = 0.01  # 10ms delay
ns["network_delay"][2, 1] = 0.0

# Add sensory input to Region_A
ns["sensory_input_conn"] = [1, 0]  # Only Region_A receives input
ns["sensory_input_func"] = "sine_wave"

println("✓ Network customized:")
println("  - Nodes: $(join(ns["node_names"], ", "))")
println("  - Models: $(join(ns["node_models"], ", "))")
println("  - Input: $(ns["sensory_input_func"]) → $(ns["node_names"][1])")

# ============================================================================
# Step 3: Customize Simulation Settings
# ============================================================================
println("\n[Step 3] Customizing simulation parameters...")

ss = settings_dict["simulation_settings"]
ss["n_runs"] = 3           # Run 3 different random initializations
ss["tspan"] = [0.0, 2000.0]  # 2 seconds total
ss["dt"] = 0.1             # 100 Hz sampling

solver_kw = ss["solver_kwargs"]
solver_kw["abstol"] = 1e-7
solver_kw["reltol"] = 1e-6

println("✓ Simulation customized:")
println("  - Runs: $(ss["n_runs"])")
println("  - Duration: $(ss["tspan"][1]) to $(ss["tspan"][2]) ms")
println("  - Sampling: $(ss["dt"]) ms ($(1000/ss["dt"]) Hz)")
println("  - Solver: $(ss["solver"]) with tight tolerances")

# ============================================================================
# Step 4: Customize Optimization Settings (even if not optimizing yet)
# ============================================================================
println("\n[Step 4] Configuring optimization settings...")

os = settings_dict["optimization_settings"]
ls = os["loss_settings"]

ls["loss_fn"] = "psd_iae"       # Power spectral density integrated absolute error
ls["freq_bands"] = [1.0, 50.0]  # Focus on 1-50 Hz band
ls["psd_smooth_sigma"] = 0.5    # Moderate smoothing

optset = os["optimizer_settings"]
optset["solver_type"] = "Adam"
optset["learning_rate"] = 0.01
optset["maxiters"] = 50

println("✓ Optimization configured:")
println("  - Loss function: $(ls["loss_fn"])")
println("  - Frequency range: $(ls["freq_bands"]) Hz")
println("  - Optimizer: $(optset["solver_type"])")
println("  - Max iterations: $(optset["maxiters"])")

# ============================================================================
# Step 5: Save Settings to JSON
# ============================================================================
println("\n[Step 5] Saving settings to file...")

output_dir = "./eneegma_example_outputs"
isdir(output_dir) || mkpath(output_dir)

settings_path = joinpath(output_dir, "example1_settings.json")
save_settings_to_json(settings_dict, settings_path)

println("✓ Settings saved to: $settings_path")

# ============================================================================
# Step 6: Inspect Saved Settings
# ============================================================================
println("\n[Step 6] Inspecting the saved settings file...")

# Reload to show what was saved
saved_settings = load_settings_from_file(settings_path)

println("\nTop-level settings keys:")
for key in sort(collect(keys(saved_settings)))
    sub_dict = saved_settings[key]
    num_items = length(sub_dict)
    println("  - $key ($num_items items)")
end

# Show network-specific settings detail
println("\nNetwork configuration as saved:")
saved_ns = saved_settings["network_settings"]
println("  Network: $(saved_ns["network_name"])")
println("  Nodes: $(join(saved_ns["node_names"], ", "))")
println("  Models: $(join(saved_ns["node_models"], ", "))")
println("  Connectivity matrix:")
conn = saved_ns["network_conn"]
for i = 1:size(conn, 1)
    row = [round(conn[i,j], digits=3) for j in 1:size(conn, 2)]
    println("    $(saved_ns["node_names"][i]): $row")
end

# ============================================================================
# Summary
# ============================================================================
println("\n" * "="^70)
println("Example 1 Complete!")
println("="^70)
println("""
You have successfully:
  ✓ Created default settings for a 2-node network
  ✓ Customized general, network, simulation, and optimization settings
  ✓ Saved settings to a JSON file with full reproducibility
  ✓ Verified the saved configuration

Next step: Try Example 2 to learn how to sample models from a grammar
and run simulations!

The settings file '$(settings_path)' can now be:
  - Loaded as: settings = load_settings_from_file("path/to/settings.json")
  - Used by other ENEEGMA functions
  - Shared with collaborators for reproducibility
  - Modified and rerun with different parameters
""")
println("="^70)
