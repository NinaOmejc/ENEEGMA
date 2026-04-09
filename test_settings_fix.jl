using ENEEGMA

println("Testing settings creation...")
settings = create_default_settings()
println("PASS: Settings created successfully")
println("LossSettings fields: $(length(fieldnames(LossSettings)))")
