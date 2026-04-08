# Example 1: Settings Configuration Walkthrough
# ===============================================
# Demonstrates creating, inspecting, and saving ENEEGMA settings.
# Settings are stored as Settings objects with sensible defaults.
using ENEEGMA
using JSON

# ============================================================================
# Step 1: Create Default Settings
# ============================================================================

settings = create_default_settings()

# ============================================================================
# Step 2: Inspect Settings by Section
# ============================================================================

# Each section shows argument names, types, and short explanations
print_settings_summary(settings; section="general_settings")
print_settings_summary(settings; section="network_settings")
print_settings_summary(settings; section="optimization_settings")
print_settings_summary(settings; section="data_settings")
print_settings_summary(settings; section="simulation_settings")
print_settings_summary(settings) # prints all sections

# ============================================================================
# Step 3: Save Settings to JSON
# ============================================================================

save_settings(settings)

# ============================================================================
# Step 4: Load Settings from File
# ============================================================================

example_settings_path = joinpath(pkgdir(ENEEGMA), "examples", "example_settings_1node.json")
loaded_settings = load_settings_file(example_settings_path)

print_settings_summary(loaded_settings);

# ============================================================================
# Step 5: Modify Settings Programmatically
# ============================================================================
# All settings fields are mutable and can be modified directly in code.
# Here we change the model from Unkown to a model (WilsonCowan) from the library
loaded_settings.network_settings.node_models = ["WilsonCowan"]
