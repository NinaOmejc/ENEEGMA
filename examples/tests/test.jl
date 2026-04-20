using ENEEGMA

# settings
settings = ENEEGMA.load_settings(".\\examples\\example_settings_2nodes.json")
settings.network_settings.node_models = ["MPR", "MPR"];

# build model
model = ENEEGMA.build_network(settings)