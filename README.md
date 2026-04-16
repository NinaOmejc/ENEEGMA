# ENEEGMA

**ENEEGMA - Exploring Neural EEG Model Architectures - is a Julia package for constructing, simulating, and optimizing networks of neural population models.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Julia Version](https://img.shields.io/badge/Julia-1.9+-blue.svg)](https://julialang.org/)

## Overview

![ENEEGMA workflow](ENEEGMA_workflow.png)

It enables:

1. **Grammar-based model generation**: Generate diverse single population models from formal grammar
2. **Network construction**: Build multi-node networks with flexible connectivity
3. **Efficient simulation**: Solve coupled stochastic or deterministic differential equations
4. **Parameter optimization**: Fit network parameters to empirical EEG data using state-of-the-art optimizers

## Installation
The package is not yet registered, so install directly from GitHub.

```julia
# Add to your Julia environment
using Pkg
Pkg.add(url="https://github.com/NinaOmejc/ENEEGMA.git")

# Or development mode
Pkg.develop(url="https://github.com/NinaOmejc/ENEEGMA.git")
```

## Examples

Five comprehensive examples demonstrate the complete workflow from configuration to optimization:

### Example 1: Settings Configuration
Learn how to load, customize, and save settings with sensible defaults.
```bash
julia examples/example1_settings.jl
```
For a full list of settings and explanations, see [settings_info.md](settings_info.md).

### Example 2: Neural Population Model Simulation
Build a network and run simulations with multiple random initializations.
```bash
julia examples/example2_simulation.jl
```

### Example 3: Grammar-based Model Sampling
Generate diverse candidate models using the formal grammar and explore their properties.
```bash
julia examples/example3_grammar_sampling.jl
```

### Example 4a: Parameter Optimization of Canonical Model
Optimize parameters for a fixed canonical population model using empirical EEG data.
```bash
julia examples/example4a_optimization_of_canonical_model.jl
```

### Example 4b: Parameter Optimization of Sampled Model
Optimize parameters for models generated from grammar sampling.
```bash
julia examples/example4b_optimization_of_sampled_model.jl
```

### Example 5: Hyperparameter Optimization
Perform hyperparameter sweep to optimize loss function weights and other optimization settings.
```bash
julia examples/example5_hyperparam_optimization.jl
```


## Related paper

If you use ENEEGMA in your research, please cite:

```bibtex
@article {Omejc2026.04.10.717643,
	author = {Omejc, Nina and Roman, Sabin and Todorovski, Ljup{\v c}o and D{\v z}eroski, Sa{\v s}o},
	title = {Neural Population Models for EEG: From Canonical Models to Alternative Model Structures},
	elocation-id = {2026.04.10.717643},
	year = {2026},
	doi = {10.64898/2026.04.10.717643},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2026/04/14/2026.04.10.717643},
	eprint = {https://www.biorxiv.org/content/early/2026/04/14/2026.04.10.717643.full.pdf},
	journal = {bioRxiv}
}

```

## License

MIT License - see [LICENSE](LICENSE) file for details


