# Bayesian Optimisation for TPOT (BO-TPOT)
An add-on for the Tree-based Pipeline Optimisation Tool (TPOT), a Python library for automated machine learning. By employing the Optuna hyperparameter optimisation Python library, Bayesian optimisation can be applied to an existing TPOT pipeline in an attempt to improve its quality by intelligently searching real-valued hyperparameter spaces.

## Introduction
[notes to fill in later]
- description of tpot
- tpot uses grid search on a discrete parameter space, even for real-valued params
- once a pipeline is found, can use bayesian optimisation to improve in a real parameter space

## Description of Operation 
The operation of BO-TPOT can be separated into three main methods:

1. `get_tpot_data` which generates the initial TPOT pipelines from the problem data;
2. `run_bo` which performs a single instance of Bayesian optimisation after a pre-determined point in the `get_tpot_data` process, to allow comparison; and,
3. `run_tpot_bo_alt` which performs an alternating series of TPOT and Bayesian optimisation operations.

All three of these are found in the file`tpot_tools.py` and may be run independently of each other, however [2] and [3] rely on previously generated pipelines and require that [1] has been run _at some stage_.

The processing of parameters and running and tracking of these methods can be performed easily using the `TestHandler` class in `tpot_tools.py`, but they can also be accessed separately for use in other code.

The following is a detailed description of these three methods and their parameters and operation.

### `get_tpot_data`:
This method uses TPOT to generate the initial pipelines for use as control data for comparison, and to save the other methods time generating pipelines.

#### Parameters:

The table below gives all parameter arguments, their type and default values:

| Parameter          | Type     | Default                   | 
|:-------------------|:--------:|--------------------------:|
|`tot_gens`         | int      | 100                       |
|`pop_size`         | int      | 100                       |
|`stop_gen`         | int      | 80                        |
|`n_runs`           | int      | 1                         |
|`start_seed`       | int      | 42                        |
|`prob_list`        | list     | `[]`                     |
|`data_dir`         | string   |`'Data'`                  |
|`results_dir`      | string   |`'Results'`               |
|`tpot_config_dict` | dict     |`default_tpot_config_dict`|
|`n_jobs`           | int      | -1                        |
|`vprint`           | `Vprint`|`u.Vprint(1)`             |
|`pipe_eval_timeout`| int      | 5                         |

#### Operation:
The parameter `prob_list` allows the user to specify a set of problems for batch processing as a list of strings. A problem is specified by the file name (without extension) for its data file, within the `data_dir` directory. For example, if the data for the two required problems are held in `./Data/prob1.data` and `./Data/prob2.data` then `data_dir` would be `'Data'` and `prob_list` would be `['prob1', 'prob2']`. If `prob_list` is supplied as the empty list `[]`, then `data_dir` is searched for all files with the extension `.data` and will process all of them.

Once the problem data is loaded using the `load_data` method in `utils.py`, the directory specified by `results_dir` is checked to see if there exists any previous run data. If such data exists, then it is kept and the current run is created in `./<results_dir>/<problem>/Run_<i+1>/` where `i` is the number of the most recent run. Run numbers are padded with a single zero, and if no run data exists, then the first run is created in `./<results_dir>/<problem>/Run_00/`. The random seed for each run is calculated as `start_seed + <run_no>` (assuming starting with run 0).

In order to generate the pipes, a TPOT regressor object is created with the following parameters:
```python
tpot = TPOTRegressor(generations=tot_gens-1,
                      population_size=pop_size, 
                      mutation_rate=0.9, 
                      crossover_rate=0.1, 
                      cv=5,
                      verbosity=tpot_verb, 
                      config_dict=tpot_config_copy, 
                      random_state=seed, 
                      n_jobs=n_jobs,
                      warm_start=False,
                      max_eval_time_mins=pipe_eval_timeout)
```

Having created the TPOT object, it is fitted to the training data for the specified number of generations. Here, we use `tot_gens-1` because the methods are compared by number of pipeline evaluations and when counting generations, TPOT does not include generation 0 - which still does need to be evaluated. Mutation and crossover rates are standard values, and the rest of the parameters are discussed below in the 'Running BO-TPOT' section of this document.

After fitting, `tpot.evaluated_individuals_` provides a dictionary of all pipelines that were evaulated, along with their CV scores. The keys for this dictionary are the string representations of each pipeline, which use a system of nested brackets to indicate the structure of the tree that represents it. For example, given operators `OpA` with a single input and 2 parameters, `OpB` with two inputs and one parameter and `OpC` with one input and one parameter, the string
```text
OpA(OpB(input_matrix, OpC(input_matrix, OpC__paramC1=0.5), OpB__paramB1=catX), OpA__paramA1=True, OpA__paramA2=2)
```
would represent the tree
```text
.
+-- OpA
    +-- OpB
    |   |-- input_matrix
    |   +-- OpC
    |   |    |-- input_matrix
    |   |     `- paramC1
    |    `- paramB1
    |-- paramA1
     `- paramA2
```
TPOT operator parameters can be real-valued, integer, categorical or boolean and the naming convention is `<operator name>__<parameter name>`.

The evaluated individuals dictionary is written in its entirety to the file:

`./<data_dir>/<problem>/<run_dir>/tpot/tpot_pipes.out` 

in the format:

`<TPOT pipe string>;<generation of pipe>;<CV value of pipe>`

The best pipeline at `stop_gen` is extracted and a `PipelinePopOpt` object (more info below) is created in order to use its `get_matching_structures` method to find all pipelines in the dictionary that have the same structure as this interrim best pipeline, and all matching pipelines (if they exist) are written to the file:
`./<data_dir>/<problem>/<run_dir>/tpot/matching_pipes.out`
in the same format as above.

Finally, the details of the run, including relevant parameters, times taken and the best pipelines at `stop_gen` and `tot_gens-1` is output to the file:

`./<data_dir>/<problem>/<run_dir>/tpot/tpot_progress.out` 

### `run_bo`
This method takes data previously generated by `get_tpot_data`

## Running BO-TPOT


## Processing Results