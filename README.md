# ENEEGMA

**Grammar-based brain network modeling and optimization coupled to EEG data**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Julia Version](https://img.shields.io/badge/Julia-1.9+-blue.svg)](https://julialang.org/)

## Overview

ENEEGMA is a Julia package for constructing, simulating, and optimizing networks of coupled neural mass models. It enables:

1. **Grammar-based model sampling**: Generate diverse network architectures from formal grammars
2. **Network construction**: Build multi-node networks with flexible connectivity and coupling
3. **Efficient simulation**: Solve coupled stochastic and deterministic differential equations
4. **Parameter optimization**: Fit network parameters to empirical EEG data using state-of-the-art optimizers

The package is designed for **reproducible research** with complete configuration tracking and output documentation.

## Key Features

### 🎯 Grammar-based Sampling
- Sample diverse model structures automatically from formal grammars
- Explore large hypothesis spaces systematically
- Compose models from parameterized building blocks

### 🧠 Network Construction
- Flexible multi-node network building
- Support for known neural mass models (Jansen-Rit, Wong-Wang, etc.)
- Custom coupling functions and connectivity patterns
- Sensory input modeling

### 🚀 Efficient Simulation
- Coupled ODE/SDE integration using DifferentialEquations.jl
- Multiple random initializations per run
- Sampling configuration and parameter exploration
- Easy result export and visualization

### 📊 Parameter Optimization
- Power spectral density (PSD) likelihood matching
- Multi-algorithm optimization (Adam, LBFGS, evolutionary)
- Constraint handling and parameter bounds
- Complete loss tracking and diagnostics

### 📝 Reproducibility
- **Every output includes settings.json** with full configuration
- Complete traceability from data to results
- Easy replication and manuscript supplementary materials

## Installation

```julia
# Add to your Julia environment
using Pkg
Pkg.add(url="https://github.com/NinaOmejc/ENEEGMA.jl.git")

# Or development mode
Pkg.develop(url="https://github.com/NinaOmejc/ENEEGMA.jl.git")
```

## Quick Start

### 1. Create Settings with Defaults

All settings have sensible defaults:

```julia
using ENEEGMA

# Create default settings for a 2-node network
settings = create_default_settings(
    network_name="MyNetwork",
    n_nodes=2,
    tspan=(0.0, 1000.0),  # milliseconds
    dt=0.1                 # integration step
)

# Customize as needed
settings["general_settings"]["verbose"] = true
settings["network_settings"]["node_models"] = ["JansenRit", "JansenRit"]
settings["simulation_settings"]["n_runs"] = 5

# Save for reproducibility
save_settings_to_json(settings, "my_settings.json")
```

### 2. Load Settings and Build Network

```julia
# Load from file (best for reproducibility)
settings = manage_settings("my_settings.json")

# Build the network
network = Network(settings=settings)
network = build_nodes!(network)
network = construct_network_connectivity!(network)
network = build_network_ode!(network)
```

### 3. Run Simulations

```julia
# Simulate with configured parameters
results = simulate_network(network)

# Results automatically include settings and metadata
```

### 4. Optimize Parameters (Optional)

```julia
# Load target EEG data
target_data = load_target_psd("eeg_data.csv")

# Run optimization
opt_result = optimize_network(network, target_data, settings)

# Results include full history and settings
```

## Examples

Three working examples demonstrate the full workflow:

### Example 1: Settings Configuration
```bash
julia examples/example1_settings.jl
```
Learn how to create, customize, and save settings with sensible defaults.

### Example 2: Grammar Sampling & Simulation
```bash
julia examples/example2_sampling_simulation.jl
```
Build a network and run simulations with multiple random initializations.

### Example 3: Parameter Optimization
```bash
julia examples/example3_optimization.jl
```
Configure and understand the parameter optimization workflow.

## Architecture

```
Grammar (formal rules)
        ↓
   Sampling & Construction
        ↓
   Node Models (populations, dynamics)
        ↓
   Network Coupling (connectivity, delays)
        ↓
   Coupled ODE System
        ↓
   Simulation / Optimization
        ↓
   EEG Output
```

## Core Modules

| Module | Purpose |
|--------|---------|
| **types/** | Core type definitions (Settings, Population, Node, Network) |
| **grammar/** | Grammar-based model sampling and parsing |
| **build/** | Network and node construction functions |
| **simulate/** | ODE integration and simulation |
| **optimize/** | Loss functions and parameter optimization |
| **utils/** | Utilities (I/O, settings management, spectral analysis) |

## Settings Structure

All configurations are managed through a hierarchical settings dictionary:

```json
{
  "general_settings": {
    "network_name": "string",
    "path_out": "string",
    "verbose": boolean,
    "seed": int or null
  },
  "network_settings": {
    "n_nodes": int,
    "node_names": ["N1", "N2", ...],
    "node_models": ["Model1", "Model2", ...],
    "network_conn": [[connectivity matrix]],
    "network_delay": [[delay matrix]]
  },
  "simulation_settings": {
    "tspan": [start, end],
    "dt": float,
    "n_runs": int,
    "solver": "Tsit5",
    "solver_kwargs": {...}
  },
  "optimization_settings": {
    "loss_settings": {...},
    "optimizer_settings": {...}
  }
}
```

**Every output includes this complete settings file**, enabling full reproducibility.

## Best Practices

1. **Always save settings**
   ```julia
   save_settings_to_json(settings, "experiment_settings.json")
   ```

2. **Use version control for settings**
   ```bash
   git add experiment_settings.json
   git commit -m "Settings for Experiment X"
   ```

3. **Print summaries**
   ```julia
   print_settings_summary(settings)
   ```

4. **Document changes**
   - What changed and why
   - How it affects results
   - Links to commits/manuscripts

5. **Compare configurations**
   - Use settings.json for manuscript supplements
   - Share exact configurations with collaborators
   - Enables reproduction by others

## Dependencies

### Core Packages
- `DifferentialEquations.jl` - ODE/SDE integration
- `Symbolics.jl` - Symbolic mathematics
- `Optimization.jl` - Parameter optimization
- `DataFrames.jl` - Data management
- `Plots.jl` - Visualization

### Minimal Set
ENEEGMA uses only essential packages (~24 total) for lightweight installation and fast load times.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature-name`)
3. Make changes with clear commit messages
4. Submit a pull request

## Citation

If you use ENEEGMA in your research, please cite:

```bibtex
@software{eneegma2026,
  author = {Omejc, Nina},
  title = {ENEEGMA: Grammar-based brain network modeling and EEG optimization},
  year = {2026},
  url = {https://github.com/NinaOmejc/ENEEGMA}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details

## Support

For questions, issues, or suggestions:
- Open an issue on GitHub
- Check existing issues for solutions
- Review example files for common use cases

## Acknowledgments

Built on excellent Julia packages including:
- DifferentialEquations.jl
- Symbolics.jl
- Optimization.jl

## Roadmap

- [ ] Stochastic noise coupling
- [ ] GPU acceleration
- [ ] Interactive visualization
- [ ] Extended model library
- [ ] Connectivity inference tools
- [ ] Spectral decomposition analysis

---

**Happy modeling!** 🧠⚡
