using ENEEGMA

println("=== Testing NodeData implementation ===\n")

# Test 1: Module loading
println("Test 1: Module loading")
try
    println("✓ ENEEGMA loaded successfully")
    println("  NodeData exported: ", isdefined(ENEEGMA, :NodeData))
    println("  Data exported: ", isdefined(ENEEGMA, :Data))
catch e
    println("✗ Module load failed: ", e)
    exit(1)
end

# Test 2: Single-node settings
println("\nTest 2: Single-node settings")
try
    settings = ENEEGMA.create_default_settings()
    settings.data_settings.target_channel = "IC3"
    ENEEGMA.check_settings(settings)
    println("✓ Single-node settings validated")
    println("  target_channel: ", settings.data_settings.target_channel)
    println("  Type: ", typeof(settings.data_settings.target_channel))
catch e
    println("✗ Single-node test failed: ", e)
    exit(1)
end

# Test 3: Multi-node settings
println("\nTest 3: Multi-node settings")
try
    settings = ENEEGMA.create_default_settings()
    settings.network_settings.n_nodes = 2
    settings.network_settings.node_names = ["C", "M"]
    settings.network_settings.node_models = ["MPR", "MPR"]
    settings.network_settings.node_coords = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]
    settings.network_settings.network_conn = [0.0 1.0; 1.0 0.0]
    settings.network_settings.network_conn_funcs = ["alpha1" "alpha1"; "alpha1" "alpha1"]
    settings.network_settings.network_delay = [0.0 5.0; 5.0 0.0]
    settings.network_settings.sensory_input_conn = [1, 0]
    
    settings.data_settings.target_channel = Dict("C" => "IC", "M" => "emgr")
    ENEEGMA.check_settings(settings)
    println("✓ Multi-node settings validated")
    println("  target_channel: ", settings.data_settings.target_channel)
    println("  Type: ", typeof(settings.data_settings.target_channel))
    println("  node_names: ", settings.network_settings.node_names)
catch e
    println("✗ Multi-node test failed: ", e)
    exit(1)
end

# Test 4: Validation - invalid node names
println("\nTest 4: Validation - invalid node names")
try
    settings = ENEEGMA.create_default_settings()
    settings.network_settings.n_nodes = 2
    settings.network_settings.node_names = ["C", "M"]
    settings.network_settings.node_models = ["MPR", "MPR"]
    settings.network_settings.node_coords = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]
    settings.network_settings.network_conn = [0.0 1.0; 1.0 0.0]
    settings.network_settings.network_conn_funcs = ["alpha1" "alpha1"; "alpha1" "alpha1"]
    settings.network_settings.network_delay = [0.0 5.0; 5.0 0.0]
    settings.network_settings.sensory_input_conn = [1, 0]
    
    settings.data_settings.target_channel = Dict("C" => "IC", "InvalidNode" => "emgr")
    ENEEGMA.check_settings(settings)
    println("✗ Should have caught invalid node name")
    exit(1)
catch e
    if contains(string(e), "InvalidNode")
        println("✓ Correctly caught invalid node name")
        println("  Error: ", e)
    else
        println("✗ Wrong error: ", e)
        exit(1)
    end
end

# Test 5: NodeData struct creation
println("\nTest 5: NodeData struct creation")
try
    node_data = ENEEGMA.NodeData(
        channel="IC",
        signal=[1.0, 2.0, 3.0],
        freqs=[0.0, 1.0, 2.0],
        powers=[10.0, 20.0, 30.0],
        measurement_noise_std=0.5
    )
    println("✓ NodeData created successfully")
    println("  channel: ", node_data.channel)
    println("  signal length: ", length(node_data.signal))
    println("  freqs length: ", length(node_data.freqs))
    println("  measurement_noise_std: ", node_data.measurement_noise_std)
catch e
    println("✗ NodeData creation failed: ", e)
    exit(1)
end

# Test 6: Data struct with Dict of NodeData
println("\nTest 6: Data struct with Dict of NodeData")
try
    node_data_dict = Dict(
        "C" => ENEEGMA.NodeData(
            channel="IC",
            signal=[1.0, 2.0, 3.0],
            freqs=[0.0, 1.0, 2.0],
            powers=[10.0, 20.0, 30.0],
            measurement_noise_std=0.5
        ),
        "M" => ENEEGMA.NodeData(
            channel="emgr",
            signal=[4.0, 5.0, 6.0],
            freqs=[0.0, 1.0, 2.0],
            powers=[40.0, 50.0, 60.0],
            measurement_noise_std=0.3
        )
    )
    
    data = ENEEGMA.Data(
        node_data=node_data_dict,
        sampling_rate=256.0,
        times=[0.0, 0.01, 0.02]
    )
    
    println("✓ Data created successfully")
    println("  Number of nodes: ", length(data.node_data))
    println("  Node names: ", collect(keys(data.node_data)))
    println("  Sampling rate: ", data.sampling_rate)
    println("  Times length: ", length(data.times))
    
    # Test access pattern
    c_data = data.node_data["C"]
    println("  Node 'C' channel: ", c_data.channel)
    println("  Node 'C' signal length: ", length(c_data.signal))
    
catch e
    println("✗ Data struct test failed: ", e)
    exit(1)
end

println("\n=== All tests passed! ===")
