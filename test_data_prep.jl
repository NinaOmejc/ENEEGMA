using ENEEGMA

println("Testing data preparation with new refactored loss system...")
settings = create_default_settings()
data = prepare_data!(settings)
println("PASS: Data prepared successfully")
println("Data struct fields: channel=$(data.channel), measurement_noise=$(data.measurement_noise_std), has_metadata=$(data.freq_peak_metadata !== nothing)")
if data.freq_peak_metadata !== nothing
    pm = data.freq_peak_metadata
    println("Metadata: roi_weight=$(pm.roi_weight), bg_weight=$(pm.bg_weight)")
    println("ROI mask: $(sum(pm.roi_mask)) regions marked")
    println("BG mask: $(sum(pm.bg_mask)) regions marked")
end
