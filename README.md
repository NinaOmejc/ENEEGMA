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

The package includes five examples covering the complete workflow:

- **Settings Configuration** ([example1_settings.jl](examples/example1_settings.jl)): Load, customize, and save settings with sensible defaults. See [settings_info.md](settings_info.md) for full documentation.
- **Neural Population Model Simulation** ([example2_simulation.jl](examples/example2_simulation.jl)): Build networks and run simulations with multiple random initializations.
- **Grammar-based Model Sampling** ([example3_grammar_sampling.jl](examples/example3_grammar_sampling.jl)): Generate diverse candidate models using formal grammar.
- **Parameter Optimization - Canonical** ([example4a_optimization_of_canonical_model.jl](examples/example4a_optimization_of_canonical_model.jl)): Optimize parameters for fixed canonical models.
- **Parameter Optimization - Sampled** ([example4b_optimization_of_sampled_model.jl](examples/example4b_optimization_of_sampled_model.jl)): Optimize parameters for grammar-generated models.
- **Hyperparameter Optimization** ([example5_hyperparam_optimization.jl](examples/example5_hyperparam_optimization.jl)): Perform hyperparameter sweep to optimize loss function weights.

## Related paper

For a detailed description of the framework, see [the paper](https://www.biorxiv.org/content/early/2026/04/14/2026.04.10.717643.full.pdf). If you use ENEEGMA in your research, please cite:

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


