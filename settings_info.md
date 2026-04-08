# ENEEGMA Settings Reference

Complete documentation of all configuration settings for ENEEGMA network building, simulation, and optimization.

---

## Table of Contents

1. [General Settings](#general-settings)
2. [Network Settings](#network-settings)
3. [Simulation Settings](#simulation-settings)
4. [Sampling Settings](#sampling-settings)
5. [Data Settings](#data-settings)
6. [Optimization Settings](#optimization-settings)
   - [Loss Settings](#loss-settings)
   - [Optimizer Settings](#optimizer-settings)
   - [Hyperparameter Sweep Settings](#hyperparameter-sweep-settings)

---

## General Settings

Top-level experiment and output configuration.

| Setting | Type | Default | Options | Description |
|---------|------|---------|---------|-------------|
| `exp_name` | String | `"SimpleNetwork"` | Any string | Experiment/project name. Used for output file naming. |
| `path_out` | String | `"./results"` | Valid file path | Directory where all outputs are saved. Created if it doesn't exist. |
| `verbosity_level` | Int | `1` | `0`, `1`, `2` | Logging verbosity: 0=silent, 1=minimal, 2=detailed. |
| `seed` | Int or null | `null` | Any integer or `null` | Master random seed. Sets Julia's global RNG state for reproducibility. If `null`, behavior is non-deterministic. |
| `make_plots` | Bool | `false` | `true`, `false` | Whether to generate visualization plots during simulation/optimization. |
| `save_model_formats` | Array[String] | `["tex"]` | `"tex"`, `"pdf"`, `"png"` | Output formats for exporting network equations and diagrams. |

---

## Network Settings

Neural network topology and dynamics configuration.

| Setting | Type | Constraints | Default | Description |
|---------|------|-------------|---------|-------------|
| `n_nodes` | Int | > 0, required | `1` | Number of nodes/populations in the network. |
| `node_names` | Array[String] | Length must equal `n_nodes` | `["N1", "N2", ...]` | Names for each node. Auto-generated if omitted. |
| `node_models` | Array[String] | Length must equal `n_nodes` | `["WC", "WC", ...]` | Model type for each node (e.g., "WC"=Wilson-Cowan, "FHN"=FitzHugh-Nagumo). |
| `node_coords` | Array[Array[Float, Float, Float]] | Length must equal `n_nodes` | `[[0,10i,0] for i=1:n_nodes]` | 3D coordinates `[x, y, z]` for each node. Used for visualization. |
| `network_conn` | Matrix[Float] | Shape `(n_nodes, n_nodes)` | `zeros(n_nodes, n_nodes)` | Connection strength matrix. Element `[i,j]` is the weight from node `i` to node `j`. |
| `network_conn_funcs` | Matrix[String] | Shape `(n_nodes, n_nodes)` | `fill("", n_nodes, n_nodes)` | Connection function strings (e.g., "sigmoid", "linear"). Maps connection dynamics. |
| `network_delay` | Matrix[Float] | Shape `(n_nodes, n_nodes)` | `zeros(n_nodes, n_nodes)` | Synaptic delay (ms) for each connection. Element `[i,j]` is delay from node `i` to `j`. |
| `sensory_input_conn` | Array[Int] | Length must equal `n_nodes` | `ones(n_nodes)` | Binary vector indicating which nodes receive sensory input (1=receives, 0=no input). |
| `sensory_input_func` | String | Valid Julia expression | `"rand(Normal(0.0, 1.0))"` | Function string for sensory input (e.g., `"sin(t)"`, `"randn()"`). Can reference time `t`. |
| `sensory_seed` | Int or null | Any integer or `null` | `null` | Random seed for sensory input generation. If `null`, uses global seed or non-deterministic. |
| `eeg_output` | String | Valid Julia expression or empty | `""` | EEG measurement function (e.g., which states to record). Empty string means no EEG output. |

---

## Simulation Settings

ODE solver and time-stepping configuration.

| Setting | Type | Constraints | Default | Description |
|---------|------|-------------|---------|-------------|
| `n_runs` | Int | > 0 | `1` | Number of independent simulation runs with different initial conditions. |
| `tspan` | Array[Float, Float] | `[t_start, t_end]`, `t_start < t_end` | `[0.0, 10.0]` | Simulation time span in milliseconds. |
| `dt` | Float or null | > 0 or `null` | `0.001` | Fixed time step (ms). Required for stochastic solvers. `null` for adaptive solvers. |
| `saveat` | Float | > 0 | `0.001` | Output sampling rate (ms). Solutions recorded at this interval. |
| `solver` | String | See solver list below | `"Tsit5"` | ODE solver algorithm to use. |
| `solver_kwargs.abstol` | Float or null | > 0 or `null` | `null` | Absolute tolerance for adaptive solvers. `null` uses solver default. |
| `solver_kwargs.reltol` | Float or null | > 0 or `null` | `null` | Relative tolerance for adaptive solvers. |
| `solver_kwargs.maxiters` | Int or null | > 0 or `null` | `null` | Maximum iterations per step. `null` uses solver default. |

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
| `grammar_file` | String | Valid file path | Constructed from `path_grammar`+`fname_grammar` | Path to grammar file (`.cfg` format). |
| `n_samples` | Int | > 0 | `10` | Number of network topologies to sample from the grammar. |
| `only_unique` | Bool | `true`, `false` | `true` | Filter out duplicate samples. |
| `max_resample_attempts` | Int | > 0 | `100` | Maximum attempts to resample if duplicates found. |
| `grammar_seed` | Int or null | Any integer or `null` | `null` | Random seed for grammar rule selection. If `null`, uses global seed or non-deterministic. |

---

## Data Settings

Target data input and metadata configuration.

| Setting | Type | Constraints | Default | Description |
|---------|------|-------------|---------|-------------|
| `data_path` | String or null | Valid file path or `null` | `""` | Directory containing input data file. |
| `data_fname` | String or null | Valid filename or `null` | `""` | Name of the input data file. |
| `fs` | Float or null | > 0 or `null` | `null` | Sampling frequency of input data (Hz). |
| `data_columns` | Array[String] or null | Valid column names or `null` | `null` | Which columns/channels to load from data file. |
| `target_channel` | String or null | Valid channel name or `null` | `null` | Which channel to use as optimization target. |

---

## Optimization Settings

Parameter optimization configuration.

| Setting | Type | Constraints | Default | Description |
|---------|------|-------------|---------|-------------|
| `method` | String | `"CMAES"` | `"CMAES"` | Optimization method. Only CMAES (Covariance Matrix Adaptation Evolution Strategy) currently supported. |
| `loss` | String | See loss functions below | `"fspb"` | Loss function to minimize. |
| `n_restarts` | Int | ≥ 1 | `1` | Number of independent optimization restarts. |
| `maxiters` | Int | > 0 | `2000` | Maximum iterations per optimization run. |
| `time_limit_minutes` | Int | > 0 | `120` | Time limit per optimization run (minutes). |
| `reparametrize` | Bool | `true`, `false` | `false` | Whether to use reparameterization strategy. |
| `reparam_strategy` | String | `"typed"`, `"none"` | `"typed"` | Reparameterization strategy (parameter grouping). |
| `param_range_level` | String | `"low"`, `"medium"`, `"high"`, `"ultra"`, `"empirical"`, `"unbounded"` | `"high"` | Parameter bounds strategy. |
| `empirical_param_table_path` | String or null | Valid file path or `null` | `null` | Path to empirical parameter table for bounds. |
| `empirical_lb_col` | String | Column name | `"q1"` | Column in empirical table for lower bounds. |
| `empirical_ub_col` | String | Column name | `"q3"` | Column in empirical table for upper bounds. |
| `abs_target_loss` | Float | ≥ 0 | `0.01` | Absolute loss target for early stopping. |
| `component_fit` | String | `"all"`, `"fspb"`, `"ssvep"` | `"all"` | Which loss component(s) to optimize. |
| `save_optimization_history` | Bool | `true`, `false` | `false` | Save iteration-by-iteration optimization history. |
| `save_all_optim_restarts_results` | Bool | `true`, `false` | `true` | Save results from all restarts (vs. best only). |
| `save_modeled_psd` | Bool | `true`, `false` | `false` | Save computed PSD from optimized model. |

### Loss Settings

Loss function configuration for optimization.

| Setting | Type | Constraints | Default | Description |
|---------|------|-------------|---------|-------------|
| `loss` | String | `"fspb"`, `"ssvep"`, `"peakbg"`, etc. | `"fspb"` | Which loss component to use. |
| `fbands` | Array[String] | Band names | `["delta", "theta", "alpha", "betalow", "betahigh"]` | Frequency bands for PSD analysis. |
| `fmin` | Float | > 0 | `1.0` | Minimum frequency (Hz) for PSD analysis. |
| `fmax` | Float | > `fmin` | `48.0` | Maximum frequency (Hz) for PSD analysis. |
| `psd_preproc` | String | Preprocessing pipe | `"log10"` | PSD preprocessing: `"log"`, `"log10"`, `"log2"`, `"none"`. Can chain with `-`. |
| `psd_window_size` | Int | > 0 | `5` | Savitzky-Golay window size (samples). |
| `psd_poly_order` | Int | > 0 | `2` | Savitzky-Golay polynomial order. |
| `psd_rel_eps` | Float | > 0 | `1e-12` | Regularization for relative normalization. |
| `psd_smooth_sigma` | Float | > 0 | `1.0` | Gaussian smoothing sigma. |
| `psd_welch_window_sec` | Float | > 0 | `2.0` | Welch window length (seconds). |
| `psd_welch_overlap` | Float | 0–0.99 | `0.5` | Welch window overlap fraction. |
| `psd_welch_nperseg` | Int | > 0 or 0 | `0` | Welch samples per segment. 0=auto. |
| `psd_welch_nfft` | Int | > 0 or 0 | `0` | FFT size. 0=auto. |
| `psd_noise_avg_reps` | Int | ≥ 1 | `1` | Number of noise averages for loss. |
| `sigma_meas` | Float | ≥ 0 | `0.0` | Measurement noise std dev. 0=no noise. |
| `auto_initialize_sigma_meas` | Bool | `true`, `false` | `true` | Auto-initialize measurement noise from data. |
| `loss_noise_seed` | Int or null | Any integer or `null` | `null` | Random seed for loss measurement noise. If `null`, non-deterministic. |
| `peak_bandwidth_hz` | Float | > 0 | `6.0` | Frequency bandwidth for peak detection (Hz). |
| `peak_prominence_db` | Float | Any | `0.5` | Prominence threshold for peak detection (dB). |
| `peak_min_frequency_hz` | Float | ≥ 0 | `5.0` | Minimum frequency for peak detection (Hz). |
| `peak_max_frequency_hz` | Float | ≥ 0 | `45.0` | Maximum frequency for peak detection (Hz). |
| `max_peak_windows` | Int | ≥ 0 | `2` | Maximum number of peak windows. |
| `fspb_enabled` | Bool | `true`, `false` | `true` | Enable FSPB (frequency-specific peak background) loss. |
| `weight_fspb` | Float | ≥ 0 | `1.0` | Weight for FSPB loss component. |
| `weight_ssvep` | Float | ≥ 0 | `1.0` | Weight for SSVEP loss component. |
| `weight_background` | Float | ≥ 0 | `0.4` | Weight for background activity component. |
| `ssvep_enabled` | Bool | `true`, `false` | `true` | Enable SSVEP (steady-state visual evoked potential) loss. |
| `ssvep_stim_freq_hz` | Float | > 0 | `5.0` | SSVEP stimulus frequency (Hz). |
| `ssvep_n_harmonics` | Int | ≥ 1 | `3` | Number of harmonics to include in SSVEP. |
| `ssvep_bandwidth_hz` | Float | ≥ 0 | `0.5` | Bandwidth around each harmonic (Hz). |
| `ssvep_harmonic_decay` | Float | 0–1 | `0.7` | Decay factor for harmonic amplitudes. |
| `max_abs_signal` | Float | > 0 | `100.0` | Clip signals to this max absolute value. |
| `max_rms_growth` | Float | > 0 | `100.0` | Max allowed RMS growth between time windows. |

### Optimizer Settings

CMAES optimizer-specific configuration.

| Setting | Type | Constraints | Default | Description |
|---------|------|-------------|---------|-------------|
| `population_size` | Int | > 0 | `50` | Population size for CMA-ES evolution strategy. |
| `sigma0` | Float | > 0 or -1 | `-1.0` | Initial step-size sigma. `-1` uses auto-scaling. |
| `K` | Float | > 0 | `0.5` | Step control parameter. |
| `n_samples` | Int | > 0 | `100` | Number of samples per iteration. |
| `learning_rate` | Float | > 0 | `0.1` | Learning rate for covariance updates. |

### Hyperparameter Sweep Settings

Grid search and hyperparameter sweep configuration.

| Setting | Type | Constraints | Default | Description |
|---------|------|-------------|---------|-------------|
| `param_range_levels` | Array[String] | Valid level names | `["high"]` | Parameter range levels to sweep. |
| `sigma0_mode` | String | `"auto"`, `"absolute"` | `"auto"` | Scaling mode for initial sigma. |
| `population_grid` | Array[Int] | > 0 | `[50]` | Population sizes to test. |
| `restart_grid` | Array[Int] | > 0 | `[1]` | Restart counts to test. |
| `sigma_values_override` | Array[Float] | > 0 or null | `null` | Override sigma0 values. |
| `hyperparameter_axes` | Array[Dict] | See format below | `[]` | Multi-dimensional hyperparameter grid. |
| `save_results` | String | `"best"`, `"all"`, `"none"` | `"best"` | Which results to save. |

**Hyperparameter Axes Format:**
```json
"hyperparameter_axes": [
  {
    "hyperparameter": ["param_name1", "param_name2"],
    "values": [value1, value2, value3]
  }
]
```

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

### With Sampling
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
    "data_path": "data/",
    "data_fname": "subject01.csv",
    "fs": 250.0,
    "target_channel": "IC1"
  },
  "optimization_settings": {
    "method": "CMAES",
    "n_restarts": 5,
    "loss_settings": {
      "loss": "fspb",
      "fbands": ["alpha", "beta"],
      "sigma_meas": 0.1
    }
  }
}
```

---

## Notes

- **Seed Hierarchy:** `seed` (general) affects all randomness globally. Specific seeds (`sensory_seed`, `grammar_seed`, `loss_noise_seed`) override the global seed for their respective components.
- **Missing Sections:** Omitted sections use their defaults from `settings.jl`.
- **Type Conversions:** JSON numbers are automatically converted to appropriate Julia types.
- **File Paths:** Can be relative (expanded to current working directory) or absolute.
