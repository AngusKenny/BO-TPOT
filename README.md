# Bayesian optimisation for TPOT (BO-TPOT)
An add-on for the Tree-based Pipeline Optimisation Tool (TPOT), a Python library for automated machine learning. By employing the Optuna hyperparameter optimisation Python library, Bayesian optimisation can be applied to an existing TPOT pipeline in an attempt to improve its quality by intelligently searching real-valued hyperparameter spaces.

## Introduction
[notes]
- description of tpot
- tpot uses grid search on a discrete parameter space, even for real-valued params
- once a pipeline is found, can use bayesian optimisation to improve in a real parameter space

## Description of operation 
The operation of BO-TPOT can be separated into three main methods:

1. `get_tpot_data` which generates the initial TPOT pipelines from the problem data;
2. `run_bo` which performs a single instance of Bayesian optimisation after a pre-determined point in the `get_tpot_data` process, to allow comparison; and,
3. `run_tpot_bo_alt` which performs an alternating series of TPOT and Bayesian optimisation operations.

All three of these are found in the file`tpot_tools.py` and may be run independently of each other, however [2] and [3] rely on previously generated pipelines and require that [1] has been run _at some stage_.

The processing of parameters and running and tracking of these methods can be performed easily using the `TestHandler` class in `tpot_tools.py`, but they can also be accessed separately for use in other code.

The following is a detailed description of these three methods and their parameters and operation.

### `run_tpot_bo_alt`:
This method uses TPOT to generate the initial pipelines for use as control data for comparison, and to save the other methods time generating pipelines.

The table below gives all parameter arguments, their type and default values:

| Parameter          | Type     | Default                   | 
|:-------------------|:--------:|--------------------------:|
|`tot_gens`         | int      | 100                       |
|`pop_size`         | int      | 100                       |
|`stop_gen`         | int      | 80                        |
|`n_runs`           | int      | 1                         |
|`start_seed`       | int      | 42                        |
|`prob_list`        | list     | `[]`                        |
|`data_dir`         | string   |`'Data'`                  |
|`results_dir`      | string   |`'Results'`               |
|`tpot_config_dict` | dict     |`default_tpot_config_dict`|
|`n_jobs`           | int      | -1                        |
|`vprint`           | `Vprint`|`u.Vprint(1)`             |
|`pipe_eval_timeout`| int      | 5                         |



## Running BO-TPOT


## Processing results