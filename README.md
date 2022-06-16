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

---

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
    |   |     `- paramC1=0.5
    |    `- paramB1=catX
    |-- paramA1=True
     `- paramA2=2
```
TPOT operator parameters can be real-valued, integer, categorical or boolean and the naming convention is `<operator name>__<parameter name>`.

The evaluated individuals dictionary is written in its entirety to the file:

`./<data_dir>/<problem>/<run_dir>/tpot/tpot_pipes.out` 

in the format:

`<TPOT pipe string>;<generation of pipe>;<CV value of pipe>`

The best pipeline at `stop_gen` is extracted and a `PipelinePopOpt` object (more info below) is created in order to use its `get_matching_structures` method to find all pipelines in the dictionary that have the same structure as this interrim best pipeline, and all matching pipelines (if they exist) are written to the file:
`./<data_dir>/<problem>/<run_dir>/tpot/matching_pipes.out`
in the same format as above. A pipeline is said to have a matching structure if it has the same operators, in the same order, but with different parameters. For example, the pipeline:
```text
OpA(OpB(input_matrix, OpC(input_matrix, OpC__paramC1=0.7), OpB__paramB1=catY), OpA__paramA1=False, OpA__paramA2=4)
```
would be considered to have the same structure as the example pipeline given above.

Finally, the details of the run, including relevant parameters, times taken and the best pipelines at `stop_gen` and `tot_gens-1` is output to the file:

`./<data_dir>/<problem>/<run_dir>/tpot/tpot_progress.out` 

---

### `run_bo`:
This method takes data previously generated by `get_tpot_data` and uses Bayesian optimisation techniques driven by the Optuna hyperparamter optimisation library for Python to improve the best pipeline that was found within a certain number of generations by TPOT.

#### Parameters:

The table below gives all parameter arguments, their type and default values:

| Parameter              | Type     | Default                   | 
|:-----------------------|:--------:|--------------------------:|
|`run_list`             | list     | `[]`                     |
|`optuna_timeout_trials`| int      | 100                       |
|`force_bo_evals`       | int      | `None`                   |
|`ignore_results`       | boolean  | True                      |
|`prob_list`            | list     | `[]`                     |
|`data_dir`             | string   |`'Data'`                  |
|`results_dir`          | string   |`'Results'`               |
|`tpot_config_dict`     | dict     |`default_tpot_config_dict`|
|`n_jobs`               | int      | -1                        |
|`vprint`               | `Vprint`|`u.Vprint(1)`             |
|`real_vals`            | boolean  | True                      |
|`pipe_eval_timeout`    | int      | 5                         |

### Operation
As with above, the problems to be processed are specified using `prob_list`. However, unlike with `get_tpot_data`, new directories are not created for runs, rather existing run directories are searched for in the path `./<results_dir>/<problem>/`. If a particular set of runs is required, then this can be specified using the `run_list` parameter, e.g., `run_list=[0,4,7]` would process runs 0, 4 and 7 only.

The `tpot_progress.out` file from the current run is processed using `get_run_data` from `utils.py` to obtain the random seed, `stop_gen` and all other relevant information. This information is used to compute the number of Bayesian optimisation evaluations to be performed with the formula:

`n_bo_evals = (n_tot_gens - tpot_stop_gen) * pop_size`

This ensures that the total number of function evaluations between `run_bo` and `get_tpot_data` are the same, to enable comparison between the two methods. For debugging (or other) purposes, this value can be overriden using the parameter `force_bo_evals`.

The method `get_progress_pop` from `utils.py` is used to load the entire set of evaluated pipelines for `tpot_stop_gen-1` generations (again, this `-1` is because TPOT does not count generation 0) from the previous `get_tpot_data` run. From this set, the best pipeline is selected and a TPOT object is created with the following parameters:
```python
tpot = TPOTRegressor(generations=0,
                      population_size=1, 
                      mutation_rate=0.9, 
                      crossover_rate=0.1, 
                      cv=5,
                      verbosity=tpot_verb, 
                      config_dict=tpot_config_copy, 
                      random_state=seed, 
                      n_jobs=n_jobs,
                      warm_start=True,
                      max_eval_time_mins=pipe_eval_timeout)
```
This is similar to above, with a few main differences. Firstly, zero generations are used so because we just want the TPOT object for the purposes of evaluating pipes, generating them. Secondly, population size is set at 1, because we only want to evaluate one pipeline at a time. The configuration dictionary `tpot_config_copy` is a deepcopy of `tpot_config_dict` that is created at the start of each run, as Python dictionaries store information by reference, so any change made in one run will be carried over until the next. Finally, `warm_start` is set to `True`, otherwise TPOT deletes all parameter sets (psets) and current population data after it finishes its operation.

The newly created TPOT object is initialised using the `tpot._fit_init` method which generates the pset and initial population. The `tpot.evaluated_individuals_` dictionary is replaced by the full dictionary of loaded pipelines and a `PipelinePopOpt` object `po` is created, which contains most of the methods needed to interact with TPOT objects and their populations. This is used to find all pipelines with matching structures and produce a new dictionary consisting of only these matching pipelines, which is then substituted for `tpot.evaluated_individuals_`.

A new individual is created using the `creator.Individual.from_string` method from the DEAP libraray, using the string representing the previously identified best pipeline and `tpot._pset` as its arguments, and substituted as the only element in the `tpot._pop` list. The TPOT object is now reinitialised by "fitting" it to the training data; however, because there already exist entries in the evaluated individuals dictionary which match the current population, it does not evaluate the pipeline, and just uses the CV value provided by the evaluated individuals dictionary. This reinitialised TPOT object is then used to reinitialise `po`.

The method `string_to_params` from `utils.py` is used to convert the matching pipelines to sets of parameter-value pairs and passed, along with the training data, to `po.optimise`.  This method sets up an Optuna study using the NSGAII sampler with the matching pipelines as initial seed samples. This study must be set up as a maximisation problem, as TPOT reports its CV values as negative[^1]. A callback object is also established in this method to determine when the optimisation process should stop. We are only counting the number of pipeline evaluations, not number of Optuna "trials", therefore the stopping condition should be when the size of `tpot.evaluated_individuals_` is equal to the number of evaluations required (plus however many matching pipelines we started with), as TPOT only adds to this dictionary if it performs a full evaluation of a pipeline. In cases where there is a discrete search space (e.g., when all hyperparameters are categorical or bounded integers) it is possible that the entire search space will be exhausted before the required dictionary size is achieved, causing Optuna to run indefinitely. In these cases, a second stopping condition is triggered, which says that if there have been `optuna_timeout_trials` trials without any change in the size of `tpot.evaluated_individuals_`, then the `study.stop()` method should be called. The counter for this trigger is reset every time the size of the dictionary is increased.

An `Objective` object is created with the training data and other relevant arguments, and passed to the Optuna `study.optimize` method. Each call to `Objective` first makes a call to `make_hp_space_real` or `make_hp_space_discrete` (depending on the value for the `real_vals` boolean flag) from `optuna_hp_spaces.py` with the current trial object and parameter names for the hyperparameters for arguments. These methods establish the hyperparameter space and distributions which Optuna should suggest values from. The values were taken directly from the [TPOT configuration files on GitHub](https://github.com/EpistasisLab/tpot/blob/master/tpot/config/regressor.py) (with some inference made as to the required distribution types).

Having constructed the search space and suggested values for each hyperparameter from the appropriate distributions, the new pipeline needs to be evaluated. The existing TPOT infrastructure is used to achieve this, and to explain how this happens, it is worth first digging a little deeper into the way in which TPOT structures its populations and performs its evaluations.

TPOT is built on top of the [Distributed Evolutionary Algorithms in Python (DEAP)](https://github.com/deap) library, and uses many of its data structures and methods in its operation. When the `warm_start` flag is set to `True`, the most recent population is accessible via the attribute `tpot._pop` which is a list of `deap.creator.Individual` objects. As there are so many different types of evolutionary algorithms, all requring different types of "individual" to function, DEAP allows the `Individual` class to be completely customisable through the use of its `Toolbox` class. In the case of TPOT, the `Individual` class maintains a `deap.gp.PrimitiveTree` and a `deap.creator.FitnessMulti` object.

A `PrimitiveTree` is a tree-based data structure made up of two data types: `deap.gp.Primitive` and `deap.gp.Terminal`. The `Primitive` objects are analogous to the internal nodes of the tree, representing the pipeline's operators, and the `Terminal` objects are analogous to the leaf nodes of the tree, representing the pipeline's inputs and hyperparameters. Within the `PrimitiveTree` object, the `Primitive` and `Terminal` objects are organised into an indexable list, using a _preorder_-type tree-traversal. Taking the example from the previous section, the list representing its `PrimitiveTree` would look like:
```text
[ Primitive(OpA),
  Primitive(OpB),
  Terminal(input_matrix),
  Primitive(OpC),
  Terminal(input_matrix),
  Terminal(OpC__paramC1=0.5),
  Terminal(OpB__paramB1=catX),
  Terminal(OpA__paramA1=True),
  Terminal(OpA__paramA2=2) ]
```

A common issue with many GP-based approaches is that, after a few generations, the trees produced can become very complex and unwieldy. In order to keep this "bloat" to a minimum, TPOT adopts as bi-objective approach where, as well as minimising the cross-validation error, the number of operators in the tree is also minimised. Both of these metrics are tracked by a tuple inside the `FitnessMulti` object for each individual in the population. Every time an individual is evaluated, this tuple is created or updated, which allows it to serve a second purpose: to act as a determinant as to whether an individual in the population should be evaluated or not. When the `tpot._evaluate_individuals` method is called, it checks `tpot._pop[i].fitness` for each `i` in the population and collates a list of any individuals which do not have valid tuple and evaluates them, avoiding wasting resources by evaluating individuals in the population that have already been evaluated. This is especially useful when TPOT is mutating an individual as, once its `PrimitiveTree` is edited, it can be marked for re-evaluation next time `tpot._evaluate_individuals` is called by deleting the fitness tuple.

One final element that should be mentioned in this digression is the `deap.gp.PrimitiveSetTyped` class. This class contains a list of all of the `Primitive` objects, and their related `Terminal` objects that can be used by DEAP to solve a GP problem - as well as all of the possible values each `Terminal` object can take. At its core, TPOT is a grid-search algorithm, selecting its hyperparameters from a discrete set of choices. Even the "real-valued" hyperparameter spaces are descretised using the `np.arrange` method, making it possible to maintain an exhaustive list of all possible values that can be selcted from. This "master list" is maintained in the attribute `tpot._pset` and it is referenced any time an individual in the population is evaluated.



[notes]
- important to distinguish between an evaluation and a trial
## Running BO-TPOT


## Processing Results




## Footnotes
[^1]: In a previous version of the code, the Optuna study was set up as a minimisation problem, which negated the CV values returned by the TPOT pipeline evaluation. However, this created some confusion and, as a result, a bug was found where the CV of the initial samples given to Optuna was not negated when the study was set up. This meant that Optuna could never find a solution with a CV value better than its initial samples, because they were input as negative values, and any subsequent CV score from the TPOT pipeline evaluations were negated and stored as positive ones in the model. This did not mean that no improvement on the pipeline was possible with BO-TPOT, as any evaluated pipeline was stored in the TPOT evaluated individuals dictionary, along with its CV value, so if there was a better pipeline that was found, it would still be there - it just meant that, in an expected improvement sense, the "bar" was being set too high and Optuna did not know whether the new pipeline was actually an improvement or not. In order to remove this confusion and avoid any further problems, the entire process was converted to a maximisation problem for both TPOT and BO.