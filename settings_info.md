# ENEEGMA Settings Reference

Complete documentation of all configuration settings for ENEEGMA network building, simulation, and optimization.

---

## Table of Contents

1. [General Settings](#general-settings)
2. [Network Settings](#network-settings)
3. [Simulation Settings](#simulation-settings)
4. [Sampling Settings](#sampling-settings)
5. [Data Settings](#data-settings)
   - [PSD Settings](#psd-settings)
6. [Optimization Settings](#optimization-settings)
   - [Loss Settings](#loss-settings)
   - [Optimizer Settings](#optimizer-settings)
   - [Hyperparameter Sweep Settings](#hyperparameter-sweep-settings)

---

## General Settings

Top-level experiment and output configuration.

| Setting | Type | Default | Options | Description |
|---------|------|---------|---------|-------------|
| `exp_name` | String | `"example-exp"` | Any string | Experiment/project name. Used for output file naming and directory structure. |
| `path_out` | String | `"./results"` | Valid file path | Base directory where all outputs are saved. Created if it doesn't exist. |
| `verbosity_level` | Int | `1` | `0`, `1`, `2` | Logging verbosity: 0=silent, 1=minimal, 2=detailed. |
| `seed` | Int or null | `null` | Any integer or `null` | Master random seed for reproducibility. If `null`, behavior is non-deterministic. |
| `make_plots` | Bool | `true` | `true`, `false` | Whether to generate visualization plots during simulation/optimization. |
| `save_model_formats` | Array[String] | `["tex"]` | `"tex"`, `"pdf"`, `"png"` | Output formats for exporting network equations and diagrams. |

---

## Network Settings

Neural network topology and dynamics configuration.

| Setting | Type | Constraints | Default | Description |
|---------|------|-------------|---------|-------------|
| `name` | String | Any string | `"example-net"` | Network name. Used for output directory naming. |
| `n_nodes` | Int | > 0, required | `1` | Number of nodes/populations in the network. |
| `node_names` | Array[String] | Length must equal `n_nodes` | `["N1", "N2", ...]` | Names for each node. Auto-generated if omitted. |
| `node_models` | Array[String/RuleTree] | Length must equal `n_nodes` | `["MPR", "MPR", ...]` | Model type for each node (e.g., "MPR"=multi-population ramp model, "WC"=Wilson-Cowan, "FHN"=FitzHugh-Nagumo). Can also be RuleTree grammar objects. |
| `node_coords` | Array[Array[Float, Float, Float]] | Length must equal `n_nodes` | `[[0, 10i, 0] for i=1:n_nodes]` | 3D coordinates `[x, y, z]` for each node. Used for visualization. |
| `network_conn` | Matrix[Float] | Shape `(n_nodes, n_nodes)` | `zeros(n_nodes, n_nodes)` | Connection strength matrix. Element `[i,j]` is the weight from node `i` to node `j`. |
| `network_conn_funcs` | Matrix[String] | Shape `(n_nodes, n_nodes)` | Diagonal: `""`, Off-diagonal: `"linear"` | Connection function strings (e.g., "linear", "sigmoid"). Maps connection dynamics. Default: "linear" for inter-node connections, "" (no self-connection) on diagonal. |
| `network_delay` | Matrix[Float] | Shape `(n_nodes, n_nodes)` | `zeros(n_nodes, n_nodes)` | Synaptic delay (ms) for each connection. Element `[i,j]` is delay from node `i` to `j`. |
| `sensory_input_conn` | Array[Int] | Length must equal `n_nodes` | `ones(n_nodes)` | Binary vector indicating which nodes receive sensory input (1=receives, 0=no input). Default: all nodes receive input. |
| `sensory_input_func` | String | Valid Julia expression | `"rand(Normal(0.0, 1.0))"` | Function string for sensory input (e.g., `"sin(t)"`, `"randn()"`). Can reference time `t`. |
| `sensory_seed` | Int or null | Any integer or `null` | `null` | Random seed for sensory input generation. If `null`, uses global seed or non-deterministic. |
| `init_seed` | Int or null | Any integer or `null` | `null` | Random seed for initial condition sampling. Allows independent control separate from sensory input randomness. If `null`, uses global seed or non-deterministic. |
| `eeg_output` | String | Valid Julia expression or empty | `""` | EEG measurement function (e.g., which states to record). Empty string means no EEG output. |

---

## Simulation Settings

ODE solver and time-stepping configuration.

| Setting | Type | Constraints | Default | Description |
|---------|------|-------------|---------|-------------|
| `tspan` | Array[Float, Float] | `[t_start, t_end]`, `t_start < t_end` | `[0.0, 60.0]` | Simulation time span in milliseconds. |
| `dt` | Float or null | > 0 or `null` | `0.0001` | Fixed time step (ms). Required for stochastic solvers. `null` for adaptive solvers. |
| `saveat` | Float | > 0 | `0.00390625` | Output sampling rate (ms). Default ~256 Hz for good PSD resolution. |
| `solver` | String | See solver list below | `"Tsit5"` | ODE solver algorithm to use. |
| `abstol` | Float or null | > 0 or `null` | `null` | Absolute tolerance for adaptive solvers. `null` uses solver default. |
| `reltol` | Float or null | > 0 or `null` | `null` | Relative tolerance for adaptive solvers. |
| `maxiters` | Int or null | > 0 or `null` | `null` | Maximum iterations per step. `null` uses solver default. |

**Available Solvers:**
- **Non-stiff ODE:** `Tsit5`, `RK4`, `BS3`, `DP5`, `Vern6`, `Vern7`, `Vern8`, `Vern9`
- **Stiff ODE:** `Rosenbrock23`, `Rodas4`, `Rodas5`, `TRBDF2`, `KenCarp4`, `QNDF`, `Rodas4P`, `Rodas5P`
- **Adaptive:** `AutoTsit5(Rosenbrock23)`, `AutoTsit5(Rodas5)`, `AutoVern7(Rodas5)`, `AutoVern9(Rodas5)`
- **SDE:** `EM`, `SOSRI`, `EulerHeun`, `SRA1`, `SRA3`, `SRIW1`, `SRIW2`, `SRI`, `RKMil`, `ImplicitEM`, `ISSEM`, `ISSEulerHeun`, `ImplicitRKMil`
- **DDE:** `MethodOfSteps(Tsit5)`, `MethodOfSteps(RK4)`, `MethodOfSteps(Vern7)`, `MethodOfSteps(Vern9)`, `MethodOfSteps(Rosenbrock23)`, `MethodOfSteps(TRBDF2)`, `MethodOfSteps(Rodas5)`
- **SDDE:** `ImplicitEM`, `LambaEM`, `RKMil`, `SOSRI`, `EulerHeun`

---

## Sampling Settings

Grammar-based network topology sampling configuration.

| Setting | Type | Constraints | Default | Description |
|---------|------|-------------|---------|-------------|
| `grammar_file` | String | Valid file path | `DEFAULT_GRAMMAR` | Path to grammar file (`.cfg` format). Defaults to `grammars/default_grammar.cfg` relative to package directory (automatically resolved via `pkgdir(@__MODULE__)`). Can specify custom absolute or relative paths. |
| `n_samples` | Int | > 0 | `10` | Number of network topologies to sample from the grammar. |
| `only_unique` | Bool | `true`, `false` | `true` | Filter out duplicate samples. |
| `grammar_seed` | Int or null | Any integer or `null` | `null` | Random seed for grammar rule selection. If `null`, uses global seed or non-deterministic. |

---

## Data Settings

Target data input and metadata configuration.

| Setting | Type | Constraints | Default | Description |
|---------|------|-------------|---------|-------------|
| `data_file` | String or null | Valid file path or `null` | `examples/example_data_rest.csv` | Path to data CSV file. Relative paths resolved from examples folder. |
| `target_channel` | String or null | Valid channel name or `null` | `"IC3"` | Which channel/column to use as optimization target. |
| `task_type` | String or null | Valid task name or `null` | `"rest"` | Task associated with the data (e.g., "rest", "task", "ssvep"). |
| `fs` | Float or null | > 0 or `null` | `256.0` | Sampling frequency of input data (Hz). |
| `data_columns` | Array[String] or null | Valid column names or `null` | `null` | Which columns/channels to load from data file. `null` loads all. |
| `estimate_measurement_noise` | Bool | `true`, `false` | `true` | Whether to estimate measurement noise directly from data. |
| `spectral_roi_definition_mode` | String | `"auto"`, `"manual"` | `"auto"` | How to define region of interest (ROI): `:auto`=peak detection, `:manual`=manual bands. |
| `spectral_roi_auto_peak_sensitivity` | Float | 0.0–1.0 | `0.3` | Sensitivity for automatic peak detection (0=loose, 1=strict). |
| `spectral_roi_manual` | Array[Array[Float, Float]] | `[[fmin, fmax], ...]` | `[[7.5, 14.0]]` | Manual frequency bands for ROI definition (e.g., alpha: 8-12 Hz). |
| `measurement_noise_std` | Float | ≥ 0 | `0.0` | Measurement noise standard deviation. 0=no noise. |

### Nested PSD Settings

PSD preprocessing configuration (nested under `psd`):

| Setting | Type | Constraints | Default | Description |
|---------|------|-------------|---------|-------------|
| `preproc_pipeline` | String | Processing pipeline spec | `"log10"` | PSD preprocessing: `"log"`, `"log10"`, `"log2"`, `"none"`. Can chain with `-`. |
| `welch_window_sec` | Float | > 0 | `2.0` | Welch window duration (seconds). |
| `welch_overlap` | Float | 0–0.99 | `0.1` | Welch window overlap fraction. |
| `welch_nperseg` | Int | ≥ 0 | `0` | Welch samples per segment. 0=auto. |
| `welch_nfft` | Int | ≥ 0 | `0` | FFT size. 0=auto. |
| `noise_avg_reps` | Int | ≥ 1 | `1` | Number of noise averages for loss computation. |
| `window_size` | Int | > 0 | `5` | Savitzky-Golay window size (samples). |
| `smooth_poly_order` | Int | ≥ 0 | `2` | Savitzky-Golay polynomial order. |
| `rel_eps` | Float | > 0 | `1e-12` | Relative epsilon for numerical stability. |
| `smooth_sigma` | Float | > 0 | `1.0` | Gaussian smoothing sigma. |
| `noise_seed` | Int or null | Any integer or `null` | `42` | Random seed for PSD noise generation. 42=deterministic, `null`=random. |

---

## Optimization Settings

Parameter optimization configuration.

| Setting | Type | Constraints | Default | Description |
|---------|------|-------------|---------|-------------|
| `method` | String | `"CMAES"` | `"CMAES"` | Optimization method. Only CMAES (Covariance Matrix Adaptation Evolution Strategy) currently supported. |
| `param_bound_scaling_level` | String | `"low"`, `"medium"`, `"high"`, `"ultra"`, `"empirical"`, `"unbounded"` | `"medium"` | Parameter bounds scaling level. Scales literature-based parameter bounds by level-specific factors. |
| `empirical_bounds_table_path` | String or null | Valid file path or `null` | `grammars/empirical_parameter_values.csv` | Path to CSV file containing empirical parameter bounds. If relative, resolved from package directory via `pkgdir(@__MODULE__)`. Absolute paths are used as-is. |
| `empirical_lower_bound_column` | String | Column name | `5perc` | Column name in empirical bounds table for lower bound values (e.g., 5th percentile). |
| `empirical_upper_bound_column` | String | Column name | `95perc` | Column name in empirical bounds table for upper bound values (e.g., 95th percentile). |
| `reparametrize` | Bool | `true`, `false` | `true` | Whether to use reparameterization strategy for parameter scaling. |
| `reparam_strategy` | String | `"typed"`, `"none"` | `"typed"` | Reparameterization strategy: `"typed"`=type-specific scaling, `"none"`=disabled. |
| `n_restarts` | Int | ≥ 1 | `1` | Number of independent optimization restarts. |
| `maxiters` | Int | > 0 | `100000` | Maximum iterations per optimization run. |
| `time_limit_minutes` | Int | > 0 | `120` | Time limit per optimization run (minutes). |
| `save_optimization_history` | Bool | `true`, `false` | `false` | Save iteration-by-iteration optimization history. |
| `save_modeled_psd` | Bool | `true`, `false` | `false` | Save computed PSD from optimized model. |
| `include_settings_in_results_output` | Bool | `true`, `false` | `true` | Include settings configuration in results output files. |

### Loss Settings

Loss function configuration for optimization.

| Setting | Type | Constraints | Default | Description |
|---------|------|-------------|---------|-------------|
| `fmin` | Float | > 0 | `1.0` | Minimum frequency (Hz) for PSD analysis in loss computation. |
| `fmax` | Float | > `fmin` | `45.0` | Maximum frequency (Hz) for PSD analysis in loss computation. |
| `roi_weight` | Float | ≥ 0 | `1.0` | Weight for region of interest (ROI) in loss computation. |
| `bg_weight` | Float | ≥ 0 | `1.0` | Weight for background activity in loss computation. |
| `loss_abstol` | Float | > 0 | `1e-3` | Absolute tolerance for loss convergence criterion. |
| `loss_reltol` | Float | > 0 | `1e-3` | Relative tolerance for loss convergence criterion. |
| `abs_target_loss` | Float | ≥ 0 | `0.01` | Absolute loss target for early stopping. |

### Optimizer Settings

CMAES optimizer-specific configuration.

| Setting | Type | Constraints | Default | Description |
|---------|------|-------------|---------|-------------|
| `population_size` | Int | > 0 | `-1` | Population size for CMA-ES evolution strategy. `-1` uses auto-scaling based on problem dimension. |
| `sigma0` | Float | > 0 or -1 | `-1.0` | Initial step-size sigma. `-1.0` uses auto-scaling. |

### Hyperparameter Sweep Settings

Grid search and hyperparameter sweep configuration (auto-populated with sensible defaults if not specified).

Default hyperparameters swept (if no config provided):
- `optimization_settings.param_bound_scaling_level`: `["medium", "high"]`
- `optimization_settings.optimizer_settings.sigma0`: `[2.0, 8.0]`
- `optimization_settings.optimizer_settings.population_size`: `[100, 150]`

---

## Examples

### Minimal Configuration
```json
{
  "general_settings": {
    "exp_name": "TestRun"
  },
  "network_settings": {
    "n_nodes": 2
  },
  "simulation_settings": {
    "tspan": [0, 1000]
  },
  "optimization_settings": {
    "method": "CMAES"
  }
}
```

### With Grammar Sampling
```json
{
  "sampling_settings": {
    "grammar_file": "grammars/default_grammar.cfg",
    "n_samples": 50,
    "grammar_seed": 42
  }
}
```

### With Data Optimization
```json
{
  "data_settings": {
    "data_file": "example_data_rest.csv",
    "target_channel": "IC3",
    "fs": 256.0,
    "spectral_roi_definition_mode": "auto",
    "psd": {
      "preproc_pipeline": "log10",
      "welch_window_sec": 2.0,
      "welch_overlap": 0.1,
      "noise_seed": 42
    }
  },
  "optimization_settings": {
    "method": "CMAES",
    "n_restarts": 5,
    "param_bound_scaling_level": "high",
    "loss_settings": {
      "fmin": 1.0,
      "fmax": 45.0,
      "roi_weight": 1.0,
      "bg_weight": 1.0
    },
    "optimizer_settings": {
      "population_size": 100,
      "sigma0": 5.0
    }
  }
}
```

### Hyperparameter Sweep
```json
{
  "hyperparameter_sweep": {
    "optimization_settings.param_bound_scaling_level": ["medium", "high"],
    "optimization_settings.optimizer_settings.sigma0": [2.0, 5.0, 8.0],
    "optimization_settings.optimizer_settings.population_size": [80, 120]
  }
}
```

---

## Notes

- **Seed Hierarchy:** `seed` (general) affects all randomness globally. Specific seeds (`sensory_seed`, `grammar_seed`, `loss_noise_seed`) override the global seed for their respective components.
- **Missing Sections:** Omitted sections use their defaults from `settings.jl`.
