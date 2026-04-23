# ENEEGMA Settings Reference

Complete reference for the current settings model implemented in [`src/types/settings.jl`](src/types/settings.jl).

---

## Table of Contents

1. [General Settings](#general-settings)
2. [Network Settings](#network-settings)
3. [Simulation Settings](#simulation-settings)
4. [Sampling Settings](#sampling-settings)
5. [Data Settings](#data-settings)
6. [PSD Settings](#psd-settings)
7. [Optimization Settings](#optimization-settings)
8. [Loss Settings](#loss-settings)
9. [Optimizer Settings](#optimizer-settings)
10. [Hyperparameter Sweep Settings](#hyperparameter-sweep-settings)
11. [Examples](#examples)
12. [Notes](#notes)

---

## General Settings

Top-level experiment and output configuration.

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `exp_name` | String | `"example-exp"` | Experiment/project name used in output names and folders. |
| `path_out` | String | `"./results"` | Base directory for all outputs. Created automatically if missing. |
| `save_model_formats` | Array[String] | `["tex"]` | Export formats for model/equation outputs. |
| `make_plots` | Bool | `true` | Whether plots should be generated. |
| `verbosity_level` | Int | `1` | Logging verbosity. Clamped to `0`, `1`, or `2`. |
| `seed` | Int or `null` | `null` | Global seed for reproducibility. |

---

## Network Settings

Network topology and node-level configuration.

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `name` | String | `"example-net"` | Network name. |
| `n_nodes` | Int | `1` | Number of nodes/populations. |
| `node_names` | Array[String] | `["N1", ..., "Nn"]` | Node names. Auto-generated if omitted. |
| `node_models` | Array[String or RuleTree] | `fill("MPR", n_nodes)` | Model assigned to each node. |
| `node_coords` | Array[Array[Float64, Float64, Float64]] | `[[0.0, 10.0*i, 0.0] ...]` | 3D node coordinates. |
| `network_conn` | Matrix[Float64] | `zeros(n_nodes, n_nodes)` | Connectivity weights. |
| `network_conn_funcs` | Matrix[String] | off-diagonal `"linear"`, diagonal `""` | Connection function names. |
| `network_delay` | Matrix[Float64] | `zeros(n_nodes, n_nodes)` | Connection delays in ms. |
| `sensory_input_conn` | Array[Int] | `ones(Int, n_nodes)` | Per-node sensory input flags. |
| `sensory_input_func` | String | `"rand(Normal(0.0, 1.0))"` | Sensory input expression. |
| `sensory_seed` | Int or `null` | `null` | Seed for sensory input randomness. |
| `init_seed` | Int or `null` | `null` | Seed for initial condition sampling. |
| `eeg_output` | Dict[String, String] or String | `{}` | Optional node-to-expression mapping for EEG output extraction. Empty dict means use node defaults. A single string is accepted as a backward-compatible shortcut for the first node. |

---

## Simulation Settings

Solver and sampling configuration.

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `tspan` | Array[Float64, Float64] | `[0.0, 60.0]` | Simulation time span in ms. |
| `dt` | Float64 or `null` | `0.0001` | Fixed time step in ms. Useful for stochastic solvers. |
| `saveat` | Float64 | `0.00390625` | Output sampling interval in ms. This corresponds to about 256 Hz. |
| `abstol` | Float64 or `null` | `null` | Absolute solver tolerance. |
| `reltol` | Float64 or `null` | `null` | Relative solver tolerance. |
| `solver` | String or `null` | `"Tsit5"` | Solver algorithm name. |
| `maxiters` | Int or `null` | `null` | Maximum solver iterations. |

---

## Sampling Settings

Grammar-based network sampling.

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `grammar_file` | String | `grammars/default_grammar.cfg` | Grammar file path. |
| `n_samples` | Int | `10` | Number of samples to generate. |
| `only_unique` | Bool | `true` | Whether duplicate samples are removed. |
| `grammar_seed` | Int or `null` | `null` | Seed for grammar sampling. |

---

## Data Settings

Input data and spectral analysis metadata.

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `data_file` | String or `null` | `examples/example_data_rest.csv` when available | Path to the input CSV file. Relative paths are resolved from the package `examples` folder. |
| `target_channel` | String, Dict[String, String], or `null` | `"IC3"` | Channel used as target data. For multi-node fits, use a dict like `{"N1": "IC3", "N2": "IC4"}`. |
| `task_type` | String or `null` | `"rest"` | Data/task label. |
| `fs` | Float64 or `null` | `256.0` | Sampling frequency of the input data in Hz. |
| `data_columns` | Array[String] or `null` | `null` | Explicit columns to load from the CSV. |
| `estimate_measurement_noise` | Bool | `true` | Whether per-node measurement noise should be estimated from the data. |
| `spectral_roi_definition_mode` | String or Symbol | `"manual"` | ROI mode. Supported values are `auto` and `manual`. |
| `spectral_roi_auto_peak_sensitivity` | Float64 | `0.3` | Sensitivity used by automatic peak detection. |
| `spectral_roi_manual` | Array of bands or Dict[String, Array of bands] | `[[7.5, 14.0]]` | Manual ROI bands. A single array applies to all nodes; a dict enables per-node ROI bands. |

### Accepted `spectral_roi_manual` Forms

Shared ROI for all nodes:

```julia
[(9.0, 14.0)]
```

Per-node ROI:

```julia
Dict(
    "N1" => [(9.0, 14.0)],
    "N2" => Tuple{Float64, Float64}[]
)
```

Internally, `DataSettings.spectral_roi_manual` is normalized to:

```julia
Dict{String, Vector{Tuple{Float64, Float64}}}
```

---

## PSD Settings

Nested under `data_settings.psd`.

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `preproc_pipeline` | String | `"log10"` | PSD preprocessing pipeline string, for example `"log10"` or `"relative-log10"`. |
| `welch_window_sec` | Float64 | `2.0` | Welch window duration in seconds. |
| `welch_overlap` | Float64 | `0.1` | Welch overlap fraction. Clamped to `[0.0, 0.99]`. |
| `welch_nperseg` | Int | `0` | Welch segment length. `0` means auto. |
| `welch_nfft` | Int | `0` | FFT size. `0` means auto. |
| `noise_avg_reps` | Int | `1` | Number of noisy PSD repetitions to average. |
| `window_size` | Int | `5` | Savitzky-Golay window size. |
| `smooth_poly_order` | Int | `2` | Savitzky-Golay polynomial order. |
| `rel_eps` | Float64 | `1e-12` | Relative epsilon for numerical stability. |
| `smooth_sigma` | Float64 | `1.0` | Gaussian smoothing sigma. |
| `transient_period_duration` | Float64 | `2.0` | Initial transient duration to discard before PSD and metric computation, in seconds. |
| `noise_seed` | Int or `null` | `42` | Seed for synthetic measurement-noise injection during PSD averaging. |

`workspace` also exists on the runtime struct but is an internal cache, not a user-facing JSON setting.

---

## Optimization Settings

Top-level optimization workflow configuration.

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `method` | String | `"CMAES"` | Optimization method. Only `"CMAES"` is currently supported. |
| `param_bound_scaling_level` | String | `"medium"` | Bounds scaling mode. Supported values include `low`, `medium`, `high`, `ultra`, `empirical`, `unbounded`. |
| `empirical_bounds_table_path` | String or `null` | `grammars/empirical_parameter_values.csv` | CSV path for empirical bounds. |
| `empirical_lower_bound_column` | String | `"5perc"` | Lower-bound column name in the empirical bounds CSV. |
| `empirical_upper_bound_column` | String | `"95perc"` | Upper-bound column name in the empirical bounds CSV. |
| `save_optimization_history` | Bool | `false` | Whether optimization history is saved. |
| `save_modeled_psd` | Bool | `false` | Whether modeled PSD output is saved. |
| `include_settings_in_results_output` | Bool | `true` | Whether settings are included in result output files. |
| `reparametrize` | Bool | `true` | Whether reparameterization is enabled. |
| `reparam_strategy` | String or Symbol | `"typed"` | Reparameterization strategy. Supported values are `typed` and `none`. |
| `reparam_type_scales` | Dict[String, Float64] | `{}` | Optional type-specific reparameterization scales. Keys are normalized internally to lowercase symbols. |
| `n_restarts` | Int | `1` | Number of optimization restarts. More you have, the better chance for global optimum; increase for better optimization (but longer runtime). |
| `maxiters` | Int | `100000` | Maximum optimization iterations. |
| `time_limit_minutes` | Int | `120` | Time limit per optimization run in minutes. |
| `output_dir` | String or `null` | `null` | Output directory for the optimization job. Usually set internally. |

The following nested settings live inside `optimization_settings`:

- `loss_settings`
- `optimizer_settings`
- `hyperparameter_sweep`

---

## Loss Settings

Nested under `optimization_settings.loss_settings`.

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `fmin` | Float64 | `1.0` | Minimum frequency used for loss computation. |
| `fmax` | Float64 | `45.0` | Maximum frequency used for loss computation. |
| `roi_weight` | Float64 | `1.0` | ROI loss weight. |
| `bg_weight` | Float64 | `1.0` | Background loss weight. |
| `loss_abstol` | Float64 | `1e-3` | Absolute tolerance for loss convergence. |
| `loss_reltol` | Float64 | `1e-3` | Relative tolerance for loss convergence. |
| `abs_target_loss` | Float64 | `0.01` | Absolute target loss for early stopping. |

---

## Optimizer Settings

Nested under `optimization_settings.optimizer_settings`.

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `population_size` | Int | `-1` | CMA-ES population size. `-1` means auto. |
| `sigma0` | Float64 | `-1.0` | Initial CMA-ES sigma. `-1.0` means auto. |

---

## Hyperparameter Sweep Settings

Nested under `optimization_settings.hyperparameter_sweep` in the `Settings` object.

In JSON, both of these forms are accepted:

```json
{
  "optimization_settings": {
    "hyperparameter_sweep": {
      "optimization_settings.param_bound_scaling_level": ["medium", "high"]
    }
  }
}
```

and the legacy top-level form:

```json
{
  "hyperparameter_sweep": {
    "optimization_settings.param_bound_scaling_level": ["medium", "high"]
  }
}
```

If omitted, the following defaults are used:

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

### Data-Driven Optimization

```json
{
  "data_settings": {
    "data_file": "example_data_rest.csv",
    "target_channel": {
      "N1": "IC3",
      "N2": "IC4"
    },
    "fs": 256.0,
    "spectral_roi_definition_mode": "manual",
    "spectral_roi_manual": {
      "N1": [[9.0, 14.0]],
      "N2": []
    },
    "psd": {
      "preproc_pipeline": "log10",
      "welch_window_sec": 2.0,
      "welch_overlap": 0.1,
      "transient_period_duration": 2.0,
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
  "optimization_settings": {
    "hyperparameter_sweep": {
      "optimization_settings.param_bound_scaling_level": ["medium", "high"],
      "optimization_settings.optimizer_settings.sigma0": [2.0, 5.0, 8.0],
      "optimization_settings.optimizer_settings.population_size": [80, 120]
    }
  }
}
```

---

## Notes

- `settings_info.md` documents the current runtime/settings model, not every historical alias.
- Old noise-related settings like `measurement_noise_std` and `loss_noise_seed` are no longer active user settings.
- Per-node measurement noise is estimated from data and stored in `NodeData`, not in `DataSettings`.
- The global `seed` can be overridden by component-specific seeds such as `sensory_seed`, `grammar_seed`, and `data_settings.psd.noise_seed`.
