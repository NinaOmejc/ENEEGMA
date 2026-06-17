#!/usr/bin/env julia
"""
Test loss function refactoring: explicit spectrum loss functions
"""

using ENEEGMA

sep = repeat("=", 80)
println(sep)
println("EXPLICIT SPECTRAL LOSS FUNCTION VALIDATION")
println(sep)

# Test 1: LossSettings has loss_function field
println("\nTest 1: LossSettings has loss_function field")
ls = LossSettings(Dict{String, Any}())
if !hasfield(LossSettings, :loss_function)
    println("✗ FAILED: loss_function field not in LossSettings")
    exit(1)
else
    println("✓ LossSettings has loss_function field")
    println("  Default: $(ls.loss_function)")
end

# Test 2: Canonical names work
println("\nTest 2: Canonical loss function names")
canonical_names = [
    "spectrum_mae",
    "region_weighted_spectrum_mae",
    "spectrum_iae",
    "region_weighted_spectrum_iae"
]

for name in canonical_names
    ls = LossSettings(Dict{String, Any}("loss_function" => name))
    if ls.loss_function != name
        println("✗ FAILED: loss_function should be '$name', got '$(ls.loss_function)'")
        exit(1)
    end
    println("✓ Canonical: $name")
end

# Test 3: Backward-compatible aliases work
println("\nTest 3: Backward-compatible aliases")
aliases = [
    ("mae", "spectrum_mae"),
    ("fsmae", "spectrum_mae"),
    ("weighted_mae", "region_weighted_spectrum_mae"),
    ("weighted_fsmae", "region_weighted_spectrum_mae"),
    ("iae", "spectrum_iae"),
    ("weighted_iae", "region_weighted_spectrum_iae"),
]

for (alias, expected) in aliases
    ls = LossSettings(Dict{String, Any}("loss_function" => alias))
    if ls.loss_function != expected
        println("✗ FAILED: alias '$alias' should map to '$expected', got '$(ls.loss_function)'")
        exit(1)
    end
    println("✓ Alias: '$alias' → '$expected'")
end

# Test 4: _canonical_metric_name works
println("\nTest 4: _canonical_metric_name function")
test_cases = [
    ("spectrum_mae", "spectrum_mae"),
    ("spectrum_iae", "spectrum_iae"),
    ("region_weighted_spectrum_mae", "region_weighted_spectrum_mae"),
    ("region_weighted_spectrum_iae", "region_weighted_spectrum_iae"),
    ("mae", "spectrum_mae"),
    ("fsmae", "spectrum_mae"),
    ("weighted_mae", "region_weighted_spectrum_mae"),
    ("weighted_fsmae", "region_weighted_spectrum_mae"),
    ("iae", "spectrum_iae"),
    ("weighted_iae", "region_weighted_spectrum_iae"),
]

for (input_name, expected_canonical) in test_cases
    try
        result = ENEEGMA._canonical_metric_name(input_name)
        if result != expected_canonical
            println("✗ FAILED: _canonical_metric_name('$input_name') should return '$expected_canonical', got '$result'")
            exit(1)
        end
        println("✓ _canonical_metric_name('$input_name') → '$result'")
    catch e
        println("✗ FAILED: _canonical_metric_name('$input_name') threw error: $e")
        exit(1)
    end
end

# Test 5: get_metric_function works
println("\nTest 5: get_metric_function dispatch")
ls_mae = LossSettings(Dict{String, Any}("loss_function" => "spectrum_mae"))
ls_region_mae = LossSettings(Dict{String, Any}("loss_function" => "region_weighted_spectrum_mae"))
ls_iae = LossSettings(Dict{String, Any}("loss_function" => "spectrum_iae"))
ls_region_iae = LossSettings(Dict{String, Any}("loss_function" => "region_weighted_spectrum_iae"))

try
    f_mae = ENEEGMA.get_metric_function(ls_mae)
    println("✓ get_metric_function(spectrum_mae) returns function")
    
    f_region_mae = ENEEGMA.get_metric_function(ls_region_mae)
    println("✓ get_metric_function(region_weighted_spectrum_mae) returns function")
    
    f_iae = ENEEGMA.get_metric_function(ls_iae)
    println("✓ get_metric_function(spectrum_iae) returns function")
    
    f_region_iae = ENEEGMA.get_metric_function(ls_region_iae)
    println("✓ get_metric_function(region_weighted_spectrum_iae) returns function")
catch e
    println("✗ FAILED: get_metric_function threw error: $e")
    exit(1)
end

# Test 6: get_loss_function takes loss_settings parameter
println("\nTest 6: get_loss_function signature")
try
    loss_fun = ENEEGMA.get_loss_function(ls_mae)
    println("✓ get_loss_function(loss_settings) works")
    println("  Returns: $(typeof(loss_fun))")
catch e
    println("✗ FAILED: get_loss_function threw error: $e")
    exit(1)
end

# Test 7: Default loss_function is region_weighted_spectrum_mae
println("\nTest 7: Default loss function")
ls_default = LossSettings(Dict{String, Any}())
if ls_default.loss_function != "region_weighted_spectrum_mae"
    println("✗ FAILED: Default loss_function should be 'region_weighted_spectrum_mae', got '$(ls_default.loss_function)'")
    exit(1)
else
    println("✓ Default loss_function: $(ls_default.loss_function)")
end

# Test 8: Case insensitivity
println("\nTest 8: Case insensitivity")
case_tests = [
    "SPECTRUM_MAE",
    "Spectrum_Mae",
    "REGION_WEIGHTED_SPECTRUM_MAE",
    "Region_Weighted_Spectrum_Mae",
]

for case_variant in case_tests
    try
        result = ENEEGMA._canonical_metric_name(case_variant)
        println("✓ _canonical_metric_name('$case_variant') → '$result'")
    catch e
        println("✗ FAILED: Case variant '$case_variant' failed: $e")
        exit(1)
    end
end

println("\n" * sep)
println("✓ ALL LOSS FUNCTION TESTS PASSED")
println(sep)
println("\nCanonical Loss Functions:")
println("  1. spectrum_mae              - Uniform MAE (ignores ROI masks)")
println("  2. region_weighted_spectrum_mae - ROI/background weighted MAE")
println("  3. spectrum_iae              - Uniform IAE (ignores ROI masks)")
println("  4. region_weighted_spectrum_iae - ROI/background weighted IAE")
println("\nKey Distinction:")
println("  - spectrum_mae: All frequency bins have equal weight")
println("  - region_weighted_spectrum_mae: ROI regions weighted separately from background")
println("  - Setting roi_weight==bg_weight does NOT make bins equally weighted")
println("    (it makes regions equally weighted, not bins)")
