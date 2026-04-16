# Example 1: Settings Configuration Walkthrough
# ============================================================================
# Demonstrates creating, inspecting, and saving ENEEGMA settings.
# Settings are stored as Settings objects with sensible defaults.
using ENEEGMA
using JSON

println("\nENEEGMA Settings Configuration Example\n")

# ============================================================================
# Step 1: Create Default Settings
# ============================================================================
settings = create_default_settings()

# ============================================================================
# Step 2: Inspect Settings by Section
# Each section shows argument names, types, and short explanations. Sections are
# "general_settings", "network_settings", "sampling_settings", "simulation_settings", and
# "data_settings", "optimization_settings".
# ============================================================================
print_settings_summary(settings; section="general_settings")
# print_settings_summary(settings) # prints all sections

# ============================================================================
# Step 3: Save Settings to JSON
# ============================================================================
save_settings(settings)

# ============================================================================
# Step 4: Load Settings from JSON File
# ============================================================================
example_settings_path = joinpath(pkgdir(ENEEGMA), "examples", "example_settings_1node.json")
loaded_settings = load_settings(example_settings_path)

# ============================================================================
# Step 5: Modify Settings Programmatically
# All settings fields are mutable and can be modified directly in code.
# Here we change the simulation time span to 20 seconds (default is 10 seconds).
# ============================================================================
loaded_settings.simulation_settings.tspan = (0.0, 20.0) # simulation time span in s

