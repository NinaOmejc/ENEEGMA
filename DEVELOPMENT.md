# ENEEGMA Development Guide

## Repository Structure

```
ENEEGMA/
├── src/                          # Source code (main package)
│   ├── ENEEGMA.jl               # Main module file (package entry point)
│   ├── types/                   # Type definitions
│   │   ├── abstract_types.jl    # Abstract base types
│   │   ├── settings.jl          # Settings data structures
│   │   ├── params.jl            # Parameter definitions
│   │   ├── variables.jl         # Variable/state definitions
│   │   └── network_types.jl     # Network-related types
│   │
│   ├── grammar/                 # Grammar-based model sampling
│   │   ├── grammar.jl           # Core grammar engine
│   │   ├── grammar_utils.jl     # Grammar utilities
│   │   └── known_models_parse_trees.jl  # Predefined model grammars
│   │
│   ├── build/                   # Network and node construction
│   │   ├── build_node.jl        # Node building functions
│   │   ├── build_population.jl  # Population dynamics
│   │   ├── build_network.jl     # Network assembly
│   │   ├── build_utils.jl       # Build utilities
│   │   ├── connectivity_functions.jl    # Coupling functions
│   │   ├── connectivity_motifs.jl       # Network patterns
│   │   ├── input_dynamics.jl    # Sensory input configuration
│   │   ├── output_dynamics.jl   # EEG output mapping
│   │   └── known_node_models.jl # Predefined node models
│   │
│   ├── simulate/                # Simulation execution
│   │   ├── simulate_network.jl  # Full network simulation
│   │   └── simulate_node.jl     # Single node simulation
│   │
│   ├── optimize/                # Parameter optimization
│   │   ├── losses.jl            # Loss function definitions
│   │   ├── optimization_utils.jl# Optimization utilities
│   │   ├── optimize_network.jl  # Main optimization routine
│   │   ├── adam_nograd.jl       # Adam optimizer variants
│   │   ├── reparametrization.jl # Parameter transformations
│   │   ├── target_preparation.jl# Target data handling
│   │   ├── evaluation_utils.jl  # Fit evaluation
│   │   ├── save_optimization_results.jl  # Results I/O
│   │   └── hyperparameter_sweep.jl      # Hyperparameter tuning
│   │
│   └── utils/                   # Utility functions
│       ├── utils.jl             # General utilities
│       ├── settings_manager.jl  # Settings loading/handling
│       ├── defaults.jl          # Default settings and helpers
│       ├── io.jl                # File I/O
│       ├── spectral_transforms.jl       # Signal processing
│       ├── extract_brain_source.jl      # Source extraction
│       └── smooth_norm.jl       # Smoothing/normalization
│
├── examples/                     # Working examples
│   ├── example1_settings.jl     # Settings configuration walkthrough
│   ├── example2_sampling_simulation.jl  # Grammar + simulation
│   └── example3_optimization.jl # Parameter optimization workflow
│
├── test/                        # Test suite
│   └── runtests.jl              # Main test file
│
├── docs/                        # Documentation
│   └── (to be populated)
│
├── Project.toml                 # Julia package manifest
├── Manifest.toml                # Dependency versions (generated)
├── README.md                    # Main documentation
├── LICENSE                      # MIT License
├── .gitignore                   # Git ignore patterns
└── DEVELOPMENT.md               # This file
```

## Key Design Principles

### 1. Settings-First Design
All functionality is controlled through a comprehensive `Settings` object that captures:
- Network architecture (connectivity, delays, coupling functions)
- Simulation parameters (solver, time spans, sampling)
- Optimization configuration (loss functions, optimizers, bounds)
- General options (verbosity, output paths, seeds)

**Best practice**: Always save settings.json alongside results for reproducibility.

### 2. Type System
Core types follow a hierarchy:
- `AbstractSettings` - Base for all setting types
- `Settings` - Complete configuration object
- `Population` - Neural population (e.g., pyramidal cells)
- `Node` - Collections of populations with internal coupling
- `Network` - Multiple nodes with inter-node connectivity

### 3. Modular Functions
Each module exports specific functionality:
- **types/**: Type definitions and constructors
- **grammar/**: Sampling operations on grammars
- **build/**: Network creation from specifications
- **simulate/**: ODE integration and sampling
- **optimize/**: Loss computation and optimization loops
- **utils/**: General-purpose utilities

### 4. Minimal Dependencies
Only 24 essential packages used:
- Core math: LinearAlgebra, Statistics, Random
- Scientific computing: DifferentialEquations, Symbolics, Optimization
- Data handling: DataFrames, CSV, JSON
- Visualization: Plots (optional)

This keeps load time fast and installation simple.

## Workflow

### Typical User Workflow

```julia
using ENEEGMA

# 1. Create configuration (with defaults)
settings = create_default_settings(network_name="MyNet", n_nodes=3)
settings["network_settings"]["node_models"] = ["JansenRit", "JansenRit", "JansenRit"]
save_settings_to_json(settings, "my_config.json")

# 2. Load and prepare
settings = manage_settings("my_config.json")
network = build_network(settings)

# 3. Simulate
results = simulate_network(network)

# 4. Optimize (optional)
data = load_target_psd("eeg_data.csv")
opt_results = optimize_network(network, data, settings)
```

### Developer Workflow

```julia
# Install in dev mode
using Pkg
Pkg.develop(path="/path/to/ENEEGMA")

# Make changes
# (edit files in src/)

# Test changes
using Revise
using ENEEGMA
# Changes auto-reload

# Run tests
Pkg.test()

# When satisfied, commit and push
```

## Output Format Convention

**Every output should include settings.json:**

```julia
results_dict = Dict(
    "metadata" => Dict(...),
    "settings" => settings_dict,  # ← Always include this!
    "simulation_results" => [...],
    "optimized_params" => {...}
)

save(results_dict, "results.json")
```

This ensures:
- ✓ Complete reproducibility
- ✓ Easy parameter checking
- ✓ Manuscript supplementary materials
- ✓ Collaboration and sharing

## Code Style Guidelines

### Naming Conventions
- **Functions**: `snake_case` for multi-word (e.g., `build_network!`)
- **Types**: `PascalCase` (e.g., `NetworkSettings`)
- **Constants**: `CONSTANT_CASE` if global (e.g., `DEFAULT_SOLVER`)
- **Boolean functions**: Ending in `!` for in-place mutations

### Docstring Format
```julia
"""
    my_function(arg1::Type, arg2::Type)::ReturnType

Brief description in one line.

Extended description explaining the purpose, context, and important details.

# Arguments
- `arg1::Type`: Description
- `arg2::Type`: Description

# Returns
- `output::ReturnType`: Description

# Examples
```julia
result = my_function(value1, value2)
```
"""
```

### Documentation
- Module-level docstrings in each file
- Function docstring for every public function
- Inline comments explaining "why", not "what"

## Testing

Run tests with:
```bash
cd ENEEGMA
julia --project -e 'using Pkg; Pkg.test()'
```

Test file: `test/runtests.jl`

## Common Tasks

### Adding a New Function

1. **Decide the module** (build/, optimize/, etc.)
2. **Write the function** with docstring
3. **Add to module exports** in the relevant ENEEGMA.jl section
4. **Add tests** for the new function
5. **Update examples** if user-facing

### Adding a New Model Type

1. **Extend grammar** in `grammar/known_models_parse_trees.jl`
2. **Implement ODEs** in `build/known_node_models.jl`
3. **Add defaults** in `build/build_utils.jl`
4. **Create example** in `examples/`

### Fixing a Bug

1. **Write test** that reproduces the bug
2. **Fix** the code
3. **Verify** test passes
4. **Make commit** with clear message

## Performance Considerations

1. **Preallocation**: Use `similar()` for arrays
2. **Type stability**: Avoid changing variable types
3. **Solver selection**: Choose appropriate ODE solver (Tsit5 default)
4. **Parallelization**: Use `pmap` for multiple runs

## Future Enhancements

- [ ] GPU acceleration (CUDA.jl)
- [ ] Parallel optimization across nodes
- [ ] Interactive visualization dashboards
- [ ] Extended model library
- [ ] Connectivity inference tools
- [ ] Spectral decomposition analysis

## Contact & Support

- **Issues**: GitHub Issues page
- **Discussions**: GitHub Discussions
- **Email**: nina.omejc@example.com

---
Last updated: April 2026
