using ENEEGMA

s = create_default_settings()
println("=== SHORT FORMAT ===")
print_settings_summary(s; section="sampling_settings")

println("\n=== LONG FORMAT ===")
print_settings_summary(s; section="sampling_settings", format_type="long")
