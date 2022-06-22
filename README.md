# Bayesian Optimisation for TPOT (BO-TPOT)
An add-on for the [Tree-based Pipeline Optimisation Tool (TPOT)](http://epistasislab.github.io/tpot/), a Python library for automated machine learning. By employing [Optuna](https://optuna.org/), a hyperparameter optimisation Python library, Bayesian optimisation can be applied to an existing TPOT pipeline in an attempt to improve its quality by intelligently searching real-valued hyperparameter spaces.

## Dependencies
[TPOT](http://epistasislab.github.io/tpot/), [PyTorch](http://pytorch.org), [Optuna](https://optuna.org/)

## Introduction
There are many different types of machine learning models, each of which has its own unique set of hyperparameters. These hyperparameters can be real values, integers or even categories, and control various aspects of the model like learning rate, different thresholds, etc. Machine learning models can be employed singularly, or combined; using the output of one model as the input to the next, harnessing the different strengths of multiple models simultaneously. When combined in this manner, the models are collectively known as a _pipeline_.

Three main decisions to consider when designing machine learning pipelines are:
- which models to select; 
- what order to apply them in; and, 
- what the values of their respective hyperparameters should be. 

TPOT is a Python libarary, built on top of the [Distributed Evolutionary Algorithms in Python (DEAP)](https://github.com/deap) library, which is designed to automate this machine learning pipeline development process. It represents pipelines as tree-based data structures and constructs them using the genetic programming (GP) methods provided by DEAP. 

One significant limitation of TPOT is that for all its GP bells-and-whistles, it is still essentially a grid-search method, and as such cannot handle continuous hyperparameter spaces. Many hyperparameters are real-valued however, but TPOT circumvents this issue by descretising the continuous search space with a certain granularity. Although this is an effective method of handling this issue, the fact still remains that unless the global optimum value for a given hyperparameter lies on the exact point of descretisation, TPOT will never find it.

[Optuna](https://optuna.org/) is a hyperparameter optimisation library for Python which uses Bayesian optimisation to tune hyperparameter values. It uses a Tree-based Parzen Estimator (TPE)-based surrogate model, built on historical information, to estimate and suggest hyperparameter values, which are then evaluated and the model subsequently updated. In contrast to TPOT,  Optuna has no limitations on the type of values the hyperparameters can take, however it is not as effective at selecting models and constructing pipelines as TPOT. 

We can think of TPOT as an effective tool for pipeline _exploration_ and Optuna as effective for pipeline _exploitation_. By using Optuna to _fine-tune_ the coarser results produced by TPOT, the BO-TPOT algorithm is able to harness the strengths of both of these powerful tools.

## Description of Operation 
The operation of BO-TPOT can be separated into three main processes:

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
|`tot_gens`          | int      | 100                       |
|`pop_size`          | int      | 100                       |
|`stop_gen`          | int      | 80                        |
|`n_runs`            | int      | 1                         |
|`start_seed`        | int      | 42                        |
|`prob_list`         | list     | `[]`                      |
|`data_dir`          | string   |`'Data'`                   |
|`results_dir`       | string   |`'Results'`                |
|`tpot_config_dict`  | dict     |`default_tpot_config_dict` |
|`n_jobs`            | int      | -1                        |
|`vprint`            | `Vprint` |`u.Vprint(1)`              |
|`pipe_eval_timeout` | int      | 5                         |

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

After fitting, `tpot.evaluated_individuals_` provides a dictionary of all pipelines that were evaulated, along with their cross-validation error (CV) scores. The keys for this dictionary are the string representations of each pipeline, which use a system of nested brackets to indicate the structure of the tree that represents it. For example, given operators `OpA` with a single input and 2 parameters, `OpB` with two inputs and one parameter and `OpC` with one input and one parameter, the string
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
TPOT operator parameters can be real-valued, integer, categorical or boolean and the naming convention is `<operator name>__<parameter name>=<parameter value>`.

The evaluated individuals dictionary is written in its entirety to the file:

`./<results_dir>/<problem>/<run_dir>/tpot/tpot_pipes.out` 

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

`./<results_dir>/<problem>/<run_dir>/tpot/tpot_progress.out` 

In the case where `get_tpot_data` is run with a single problem and for a single run, the run number of the recently created data is returned so it can be passed directly to `run_bo` or `run_tpot_bo_alt`.

---

### `run_bo`:
This method takes data previously generated by `get_tpot_data` and uses Bayesian optimisation techniques driven by the Optuna hyperparamter optimisation library for Python to improve the best pipeline that was found within a certain number of generations by TPOT.

#### Parameters:

The table below gives all parameter arguments, their type and default values:

| Parameter              | Type     | Default                   | 
|:-----------------------|:--------:|--------------------------:|
|`run_list`              | list     | `[]`                      |
|`optuna_timeout_trials` | int      | 100                       |
|`force_bo_evals`        | int      | `None`                    |
|`ignore_results`        | boolean  | True                      |
|`prob_list`             | list     | `[]`                      |
|`data_dir`              | string   |`'Data'`                   |
|`results_dir`           | string   |`'Results'`                |
|`tpot_config_dict`      | dict     |`default_tpot_config_dict` |
|`n_jobs`                | int      | -1                        |
|`vprint`                | `Vprint` |`u.Vprint(1)`              |
|`real_vals`             | boolean  | True                      |
|`pipe_eval_timeout`     | int      | 5                         |

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
                      n_jobs=1,
                      warm_start=True,
                      max_eval_time_mins=pipe_eval_timeout)
```
This is similar to above, with a few main differences. Firstly, zero generations are used so because we just want the TPOT object for the purposes of evaluating pipes, generating them. Secondly, population size is set at 1, because we only want to evaluate one pipeline at a time - it is for this reason we set `n_jobs` to 1 as well. The configuration dictionary `tpot_config_copy` is a deepcopy of `tpot_config_dict` that is created at the start of each run, as Python dictionaries store information by reference, so any change made in one run will be carried over until the next. Finally, `warm_start` is set to `True`, otherwise TPOT deletes all parameter sets (psets) and current population data after it finishes its operation.

The newly created TPOT object is initialised using the `tpot._fit_init` method. The `tpot.evaluated_individuals_` dictionary is replaced by the full dictionary of loaded pipelines and a `PipelinePopOpt` object `po` is created, which contains most of the methods needed to interact with TPOT objects and their populations. This is used to find all pipelines with matching structures and produce a new dictionary consisting of only these matching pipelines, which is then substituted for `tpot.evaluated_individuals_`.

TPOT is built on top of the [Distributed Evolutionary Algorithms in Python (DEAP)](https://github.com/deap) library, and uses many of its data structures and methods in its operation. A new individual is created using the `creator.Individual.from_string` method from this libraray, using the string representing the previously identified best pipeline and `tpot._pset` as its arguments, and substituted as the only element in the `tpot._pop` list. The TPOT object is now reinitialised by "fitting" it to the training data; however, because there already exist entries in the evaluated individuals dictionary which match the current population, it does not evaluate the pipeline, and just uses the CV value provided by the evaluated individuals dictionary. This reinitialised TPOT object is then used to reinitialise `po`.

The method `string_to_params` from `utils.py` is used to convert the matching pipelines to sets of parameter-value pairs and passed, along with the training data, to `po.optimise`.  This method sets up an Optuna study using the multivariate Tree-based Parzen Estimator (TPE) sampler with the matching pipelines as initial seed samples. This study must be set up as a maximisation problem, as TPOT reports its CV values as negative[^1]. When processing the matching pipelines as the initial seeds for the Optuna study, a dictionary of the distribution for each parameter in the pipeline must be generated. This is done using the `make_optuna_trial_real` and `make_optuna_trial_discrete`. Due to a precision quirk in the way TPOT descretises continuous search spaces, sometimes it will provide hyperparameters such as `RandomForestRegressor__max_features=0.7500000000000001`. When working in continuous hyperparameter spaces, this is no problem; however, when `real_vals` is set to `False`, it is important to ensure that any numerical values are rounded to an appropriate precision, because Optuna will not accept the value `0.7500000000000001` as being part of the discrete distribution `[..., 0.7, 0.75, 8, ...]`.

A callback object is also established in this method to determine when the optimisation process should stop. We are only counting the number of pipeline evaluations, not number of Optuna "trials", therefore the stopping condition should be when the size of `tpot.evaluated_individuals_` is equal to the number of evaluations required (plus however many matching pipelines we started with), as TPOT only adds to this dictionary if it performs a full evaluation of a pipeline. In cases where there is a discrete search space (e.g., when all hyperparameters are categorical or bounded integers) it is possible that the entire search space will be exhausted before the required dictionary size is achieved, causing Optuna to run indefinitely. In these cases, a second stopping condition is triggered, which says that if there have been `optuna_timeout_trials` trials without any change in the size of `tpot.evaluated_individuals_`, then the `study.stop()` method should be called. The counter for this trigger is reset every time the size of the dictionary is increased.

An `Objective` object is created with the training data and other relevant arguments, and passed to the Optuna `study.optimize` method. Each call to `Objective` first makes a call to `make_hp_space_real` or `make_hp_space_discrete` (depending on the value for the `real_vals` boolean flag) from `optuna_hp_spaces.py` with the current trial object and parameter names for the hyperparameters for arguments. These methods establish the hyperparameter space and distributions which Optuna should suggest values from. The values were taken directly from the [TPOT configuration files on GitHub](https://github.com/EpistasisLab/tpot/blob/master/tpot/config/regressor.py) (with some inference made as to the required distribution types). 

Having constructed the search space, Optuna samples values for each hyperparameter from the appropriate distributions, creating a new pipeline which needs to be evaluated. The existing TPOT infrastructure is used to achieve this, and to explain how this happens it is worth first digging a little deeper into the way in which TPOT structures its populations and performs its evaluations.

When the `warm_start` flag is set to `True`, the most recent population is accessible via the attribute `tpot._pop` which is a list of `deap.creator.Individual` objects. As there are so many different types of evolutionary algorithms, all requring different types of "individual" to function, DEAP allows the `Individual` class to be completely customisable through the use of its `Toolbox` class. In the case of TPOT, the `Individual` class maintains a `deap.gp.PrimitiveTree` and a `deap.creator.FitnessMulti` object.

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

A common issue with many GP-based approaches is that, after a few generations, the trees produced can become very complex and unwieldy. In order to keep this "bloat" to a minimum, TPOT adopts as bi-objective approach where, as well as minimising the cross-validation error, the number of operators in the tree is also minimised. Both of these metrics are tracked by a tuple inside the `FitnessMulti` object for each individual in the population. Every time an individual is evaluated, this tuple is created or updated, which allows it to serve a second purpose: to act as a determinant as to whether an individual in the population should be evaluated or not. When the `tpot._evaluate_individuals` method is called, it checks `tpot._pop[i].fitness` for each `i` in the population and collates a list of any individuals which do not have valid tuple and evaluates them, avoiding wasting resources by evaluating individuals in the population that have already been evaluated. This is especially useful when TPOT is mutating an individual as, once its `PrimitiveTree` is edited, it can be marked for re-evaluation next time `tpot._evaluate_individuals` is called by deleting the fitness tuple. If the tuple exists, then TPOT will simply add the new pipeline to the `tpot.evaluated_individuals_` dictionary with whatever values are present, without evaluation.

One final element that should be mentioned in this digression is the `deap.gp.PrimitiveSetTyped` class. This class contains a list of all of the `Primitive` objects, and their related `Terminal` objects that can be used by DEAP to solve a GP problem - as well as all of the possible values each `Terminal` object can take. At its core, TPOT is a grid-search algorithm, selecting its hyperparameters from a discrete set of choices. Even the "real-valued" hyperparameter spaces are descretised using the `np.arrange` method, making it possible to maintain an exhaustive list of all possible values that can be selcted from. When a TPOT instance is created and initialised using the `tpot._fit_init` or `tpot.fit` methods, it uses the `config_dict` attribute to compile a list of operators, hyperparameters and all of the values those hyperparameters can take. This "master list" is maintained in the attribute `tpot.operators`, and is used to generate a `PrimitiveSetTyped` object which is referenced any time an individual in the population is evaluated.

Back with Optuna...

Although TPOT is a grid-search algorithm requiring a discrete search space, Optuna has no such limitations. This means that Optuna can suggest any value from a distribution, be it categorical, bound (or unbound) integer, uniform real or even log uniform real - the challenge is in making the two methods play nicely with each other. Once the new hyperparameter values have been suggested, `Terminal` objects for each of the new hyperparameters are created and inserted into the `PrimitiveTree` object for the single individual in the population. As well as this, the new hyperparameter values must be added to `tpot.operators` and `tpot._pset`, in order to "trick" TPOT into accepting these new values as part of the original, discrete, set of choices. Having updated these two data structures, the fitness values for the individual are deleted, to mark it for evaluation, and `tpot._evaluate_individuals` is called. This updates the fitness values tuple and the CV value is returned as the score for that Optuna trial. These values are tracked in the `PipelinePopOpt` object and the next trial is performed, until either of the stopping conditions mentioned earlier are met.

Once the stopping conditions are met, the `tpot.evaluated_individuals_` dictionary is written to the file:

`./<results_dir>/<problem>/<run_dir>/bo/bo_pipes.out`

using the same convention as before, and the general run information is written to the file:

`./<results_dir>/<problem>/<run_dir>/bo/bo_progress.out`

---

### `run_tpot_bo_alt`:
This method is similar in operation to `run_bo_alt`, except that instead of performing a single long TPOT execution and then a single long execution of BO to improve on it, the algorithm alternates between TPOT and BO in an interative fashion.

#### Parameters:

The table below gives all parameter arguments, their type and default values:

| Parameter              | Type     | Default                   | 
|:-----------------------|:--------:|--------------------------:|
|`n_iters`               | int      | 10                        |
|`run_list`              | list     | `[]`                      |
|`optuna_timeout_trials` | int      | 100                       |
|`force_bo_evals`        | int      | `None`                    |
|`ignore_results`        | boolean  | True                      |
|`prob_list`             | list     | `[]`                      |
|`data_dir`              | string   |`'Data'`                   |
|`results_dir`           | string   |`'Results'`                |
|`tpot_config_dict`      | dict     |`default_tpot_config_dict` |
|`n_jobs`                | int      | -1                        |
|`vprint`                | `Vprint` |`u.Vprint(1)`              |
|`real_vals`             | boolean  | True                      |
|`pipe_eval_timeout`     | int      | 5                         |

### Operation
The initialisation and specification of problems, runs, etc. for `run_tpot_bo_alt` is very similar to `run_bo`, except it does not load the previous TPOT data all the way up until the original stopping generation. Instead, it uses the input variable `n_iters` to split the available computing budget into `n_iters` "chunks". For example, if the initial TPOT run had a population size of 100, went for 100 generations and had an initial stopping generation of 80, this would mean that `run_bo` would have had pipeline 2000 evaluations to carry out its optimisation on the single best pipeline after TPOT generation 80. If `n_iters` is set at 10, this would mean that `run_tpot_bo_alt` would run for 10 iterations, each iteration consisting of 8 TPOT generations (`n_tpot_gens`) and 200 BO evaluations (`n_bo_evals`).

To begin with, a TPOT object is created with the following parameters:
```python
tpot = TPOTRegressor(generations=0,
                      population_size=pop_size, 
                      mutation_rate=0.9, 
                      crossover_rate=0.1, 
                      cv=5,
                      verbosity=tpot_verb, 
                      config_dict=main_tpot_config_dict, 
                      random_state=seed, 
                      n_jobs=n_jobs,
                      warm_start=True,
                      max_eval_time_mins=pipe_eval_timeout)
```
and initialised using `tpot._fit_init` to generate the `tpot._pset`. This "master" TPOT object has a population size of `pop_size` and maintains the main population throughout the optimisation process. A "master" `PipelinePopOpt` object is also created to give access to the methods that interact with the TPOT object and provide an interface between TPOT and Optuna.

With each iteration, `tpot.fit` is called, using the training data as input arguments, for `n_tpot_gens` generations (taking care to ensure that for the first iteration it runs for `n_tpot_gens-1` generations to account for the fact that TPOT doesn't include generation 0 in its count)[^2]. The best pipeline from the evolved population is identified, along with any others in the `tpot.evaluated_individuals_` dictionary that have the same structure. A new TPOT object is created using the parameters:
```python
bo_tpot = TPOTRegressor(generations=0,
                      population_size=1, 
                      mutation_rate=0.9, 
                      crossover_rate=0.1, 
                      cv=5,
                      verbosity=tpot_verb, 
                      config_dict=bo_tpot_config_dict, 
                      random_state=seed, 
                      n_jobs=1,
                      warm_start=True,
                      max_eval_time_mins=pipe_eval_timeout)
```
and initialised with `bo_tpot._fit_init`. 

A new deep copy of the `config_dict` is used when establishing this TPOT object, as any changes made to the dictionary should not be carried over from iteration to iteration. A new `PipelinePopOpt` object is also created to manage this TPOT object. The hyperparameters of the pipeline to be optimised with BO are added to `bo_tpot._pset` and `bo_tpot.operators`, as they might contain values suggested by the BO step in previous iterations, which are not present in the original `config_dict`. The pipeline to be optimised (and other pipelines with matching structures) is then added to the `bo_tpot.evaluated_individuals_` dictionary and fit to the training data, before being optimised using Optuna in the same way as `run_bo`, except only for `n_bo_evals` (or until the other stopping condition is met).

If the BO step is a success (i.e., produces a pipeline with better CV error than any in the TPOT population) the main `tpot._pset` and `tpot.operators` are updated with the new hyperparameter values and the previous best pipeline is replaced by the new, improved pipeline; if no improvement is found, the main population stays the same. The current progress is recorded in the file:

`./<data_dir>/<problem>/<run_dir>/alt/alt_progress.out`

the TPOT and BO evaluated pipes are written to:

`./<data_dir>/<problem>/<run_dir>/alt/alt_tpot_pipes.out`

and

`./<data_dir>/<problem>/<run_dir>/alt/alt_bo_pipes.out`

respectively, and it moves on to the next iteration.

## Running BO-TPOT
As mentioned above, it is possible to run the `get_tpot_data`, `run_bo` and `run_tpot_bo_alt` processes individually, by calling their respective methods with the appropriate parameter arguments, but the simplest way is to execute the script in the file`run_tpot_tests.py`. At the start of this script is a parameters dictionary `params` which the user can populate and is given to a `TestHandler` object which sets up and runs all the required tests, in accordance with the specified parameters, automatically.

The table below gives the parameters in this dictionary and their type, followed by a description of their functions:

| Parameter              | Type       |
|:-----------------------|:----------:|
|`CLEAN_DATA`            | boolean    |
|`RUN_TPOT`              | boolean    |
|`RUN_BO`                | boolean    |
|`RUN_ALT`               | boolean    |
|`VERBOSITY`             | int        |
|`DATA_DIR`              | string     |
|`RESULTS_DIR`           | string     |
|`RUNS`                  | int / list | 
|`PROBLEMS`              | list       |
|`TPOT_CONFIG_DICT`      | dict       |
|`nJOBS`                 | int        |
|`REAL_VALS`             | boolean    |
|`PIPE_EVAL_TIMEOUT`     | int        | 
|`START_SEED`            | int        | 
|`POP_SIZE`              | int        | 
|`nTOTAL_GENS`           | int        | 
|`STOP_GEN`              | int        | 
|`OPTUNA_TIMEOUT_TRIALS` | int        |
|`nALT_ITERS`            | int        |

### `CLEAN_DATA`:
This parameter allows the user to clear **all** data in run directories that are to be written to, when starting an execution. This parameter depends on the values of `RUN_TPOT`, `RUN_BO`, `RUN_ALT`, `RUNS` and `PROBLEMS`, and will only delete data from whatever is flagged to execute in the current instance. When set to `True`, it will display the list of runs and problems it is about to clear, and whether the `run_bo` or `run_tpot_bo_alt` (or both) data will be removed, and will ask for confirmation[^3]. Confirming with `y` will remove the specified data directories and continue with the execution; cancelling with `n` (or any other input) will exit the script. Once the data has been cleared it will ask for a second confirmation to ensure the user wants to continue executing the script. Selecting `y` (or any other input) here will continue as normal, selecting `n` here will exit without processing anything else, meaning this parameter can be used as an easy way to clean the `RESULTS_DIR` directory back to the initial TPOT data if needed. If this parameter is set to `False`, any existing data will still be overwritten, but only as-and-when new data is produced. If `RUN_TPOT` is set to `True` then this step is ignored, as generated TPOT data is never affected.

### `RUN_TPOT`/`RUN_BO`/`RUN_ALT`:
Boolean flag to control which of `get_tpot_data`, `run_bo` and `run_tpot_bo_alt` are executed. Set to `False` to skip any of the processes, however keep in mind that both `run_bo` and `run_tpot_bo_alt` require that `get_tpot_data` has been run _at some stage_.

### `VERBOSITY`:
There are four levels of verbosity that can be set:
| Level | Description                             |
|:-----:|:----------------------------------------|
| 0     | all output off - print errors only*     |
| 1     | progress information                    |
| 2     | debug information                       |
| 3     | show everything, including all warnings |

(\* some warnings from the scikit-learn library may still appear)

The verbosity output is controlled by the `Vprint` object from `utils.py`. For example, to restrict a particular output to verbosity level 2 or above, instead of using the `print` command, `Vprint.v2` would be used with the same syntax as `print`.

### `DATA_DIR`/`RESULTS_DIR`:
Strings to specify the path that data should be drawn from, and results written to, respectively.

### `PROBLEMS`:
This is a list which specifies the problems to be batch-processed, in exactly the same way as `prob_list` is described above. If this is set to the empty list `[]`, then `DATA_DIR` is searched for appropriate problem data, and all datasets found are processed.

### `TPOT_CONFIG_DICT`:
The configuration dictionary to be used by TPOT containing all of the operators, their hyperparameters and all possible values they can take. These must all be finite and discrete, expressed as lists, integer ranges or real ranges that have been descretised using `np.arrange`. The default dictionary cannot be used directly from the TPOT library as any changes to it will persist across runs; meaning that a new (deep) copy must be made for each new TPOT object instance.

The file `tpot_config.py` provides a copy of the default TPOT configuration dictionary as `default_tpot_config_dict`. Also present in this file is `reduced_tpot_config_dict`, a configuration dictionary with a reduced subset of the default operators, useful for running small and controllable tests when debugging.

### `nJOBS`:
This parameter allows the user to specify the number of parallel jobs to run whenever TPOT has a population of solutions to simultaneously evaluate. This value is not used when creating the TPOT object used to evaluate pipelines during the BO step, or when setting up an Optuna study, as there is no advantage gained by parallelisation in these instances. When set to -1, the maximum number of cores available will be used.

### `REAL_VALS`:
This boolean flag determines the type of search space the hyperparameters should be drawn from in the BO step. If set to `False` the `make_hp_space_discrete` method from `optuna_hp_spaces.py` is called which allows Optuna to suggest hyperparameter values from the exact same set of values contained in `default_tpot_config_dict`.  When set to `True`, Optuna is able to suggest hyperparameter values from continuous distributions during the BO step.

While there may still be some benefit to adding a BO step using a discrete hyperparameter space - harnessing the power of BO over simple random grid-search - it is undoubtedly the fact that it is able to produce pipelines with hyperparameter values sampled from continous spaces that gives BO-TPOT the edge over canonical TPOT. Therefore, it is highly recommended to run BO-TPOT with `REAL_VALS` set to `True`.

### `PIPE_EVAL_TIMEOUT`:
This parameter allows the user to determine the maximum time allowed for a single pipeline evaluation by TPOT in minutes. The default TPOT value is 5, and it is suggested to use this value.

### `RUNS`:
This parameter allows the user to specify how many runs they want to execute or, if no new TPOT data is being generated, what specific runs they wish to apply `run_bo` and/or `run_tpot_bo_alt` to. It can be specified as an `int` or `list`. If `RUN_TPOT` is set to `True` this parameter must be set as an `int` and not a `list`, as `get_tpot_data` does not overwrite previous runs; rather it will create a new run directory with its number subsequent to the number of the most recent run. The algorithm determines if run data exists by whether there are any run directories (of the form `Run_XX`, where `XX` is the run number, padded with a zero for numbers less than 10) so when starting from scratch, or after changing parameter values between runs, make sure that the problem directories are clear before executing. If new TPOT data is generated, the new run number is returned to the main script and passed to `run_bo` and/or `run_tpot_bo_alt`, if they are being executed.

In the case where `RUN_TPOT` is set to `False`, this parameter may be given as a list, or range, indicating the specific runs to process. If given as an `int` then the run list will be taken as `range(RUNS)`.

### `START_SEED`:
This parameter allows the user to specify the starting seed for the first run, with subsequent runs incrementing this seed by 1 each time. It only applies to `get_tpot_data` as `run_bo` and `run_tpot_bo_alt` will acquire this information from the `tpot_progress.out` file for each run. Be careful when setting the start seed, that you are not duplicating the results of a previous run.

### `POP_SIZE`:
This parameter allows the user to specify the population size when generating the initial TPOT data. It only applies to `get_tpot_data` as `run_bo` and `run_tpot_bo_alt` will acquire this information from the `tpot_progress.out` file for each run.

### `nTOTAL_GENS`:
This parameter allows the user to specify the total number of generations when producing the initial TPOT data. It only applies to `get_tpot_data` as `run_bo` and `run_tpot_bo_alt` will acquire this information from the `tpot_progress.out` file for each run.

### `STOP_GEN`:
This parameter allows the user to specify the point at which the BO step should commence for the `run_bo` process (and is used to calculate how much computational budget should be allocated to the BO step for each iteration in `run_tpot_bo_alt`).  It only applies to `get_tpot_data` as `run_bo` and `run_tpot_bo_alt` will acquire this information from the `tpot_progress.out` file for each run.

### `OPTUNA_TIMEOUT_TRIALS`:
As mentioned above, the main stopping condition for the BO step is when the size of the `tpot.evaluated_individuals_` dictionary indicates that the requried number of evaluations have been performed. In cases where it is possible to exhaustively search the entire hyperparameter space before this size is reached, this can result in Optuna running indefinitely. Therefore a second stopping condition is applied when no change is made to the size of the `tpot.evaluated_individuals_` dictionary within some specified number of trials. This parameter specifies this number of trials, its suggested value is 100.

### `nALT_ITERS`:
This parameter allows the user to specify the number of iterations that `run_tpot_bo_alt` should run for. It is used to divide the total computational budget between alternating instances of the TPOT and BO steps.

### Tracking runs:
Along with simplifying the process of running tests with the three aspects of BO-TPOT, the script in `run_tpot_tests.py` also automatically tracks the progress of the runs.

Each time this script is executed, a file in `./<RESULTS_DIR>/` called `progress.out` is created with the date and time the execution was started, the values of the parameters in the `params` dictionary (if the default TPOT configuration dictionary is used, then it will just say `default_tpot_config_dict`, otherwise it will give the full dictionary). As well as this general execution information, for each problem and run, the seed is given, and the outcome of any processes run, along with the time taken (if successful). If any process fails or crashes, then the error message and stack traceback will be written to this file, but the execution itself will contine with the next process or run. This helps to diagnose any minor or incidental bugs, without having to cancel the entire set of runs.

## Processing Results
The file `process_full_results.py` provides a script that automatically processes the results produced by BO-TPOT, computes the statistics and generates plots for the data. By default, the script searches each problem sub-directory in the results directory to find runs which have valid results. It performs a check on each run to ensure that the parameters used to generate the results are the same as those used in the first valid run, if there is a mis-match in parameter values, or if the results themselves are invalid, it will skip the run and not include it in the final statistics or plots.

A plot is generated for the control TPOT data, `run_bo` and `run_tpot_bo_alt`(if available) and tracks the mean value against pipeline evaluation calls across all valid runs for each problem, with +/- standard deviation shaded. As well as these plots, a box plot is also generated showing the number of pipelines with the same structure at the `STOP_GEN` cut-off point. All plots are saved as `.PNG ` files to `./<RESULTS_DIR>/<problem>/Plots/`. 

Along with the plots, a separate file is generated with the statistics of the processed runs. This file gives the date and time the results were processed, then for each problem it gives the list of runs that were processed and then a semicolon-separated table with the best, worst, median, mean and standard deviation of all the processed runs for that problem. The file is saved to:

`./<RESULTS_DIR>/stats.out`

At the top of the script is a dictionary of parameters that can be set; the table below gives the parameters in this dictionary and their type, followed by a description of their functions:

| Parameter              | Type       |
|:-----------------------|:----------:|
|`RESULTS_DIR`           | string     |
|`PROBLEMS`              | list       |
|`RUN_LIST`              | list       |
|`SAVE_PLOTS`            | boolean    |
|`PLOT_ALT`              | boolean    |
|`PLOT_MIN_MAX`          | boolean    |
|`SKIP_PLOT_INIT`        | int        |

### `RESULTS_DIR`:
String to specify the the name of the directory that results data should be drawn from.

### `PROBLEMS`:
This is a list which specifies the problems to be batch-processed, in exactly the same way as it is described above. If this is set to the empty list `[]`, then `RESULTS_DIR` is searched for existing results, and all results found are processed.

### `RUN_LIST`:
This parameters allows the user to specify the runs to be processed. If this is set to the empty list `[]` it will process all valid runs it can find under the problem directory, otherwise this parameter can be give as a list or range of specific run numbers.

### `SAVE_PLOTS`:
When this parameter is set to `True`, the plots generated by the script will be saved as `.PNG` files to `./<RESULTS_DIR>/<problem>/Plots/`. If set to `False`, the plots will be displayed but not saved.

### `PLOT_ALT`:
If it is not required to plot the results of `run_tpot_bo_alt`, this parameter should be set to `False`.

### `PLOT_MIN_MAX`:
As well as the plots tracking the mean with shaded +/- standard deviation, setting this parameter to `True` will also provide plots tracking the mean with shaded best and worst.

### `SKIP_PLOT_INIT`:
If there is a very large gap between the CV scores of the initial solutions and later ones, plotting the entire set of results can mean it is hard to see any small changes after it starts converging. This parameter allows the user to specify a number of initial solutions to skip, making the plot much easier to read.

## Footnotes
[^1]: In a previous version of the code, the Optuna study was set up as a minimisation problem, which negated the CV values returned by the TPOT pipeline evaluation. However, this created some confusion and, as a result, a bug was found where the CV of the initial samples given to Optuna was not negated when the study was set up. This meant that Optuna could never find a solution with a CV value better than its initial samples, because they were input as negative values, and any subsequent CV score from the TPOT pipeline evaluations were negated and stored as positive ones in the model. This did not mean that no improvement on the pipeline was possible with BO-TPOT, as any evaluated pipeline was stored in the TPOT evaluated individuals dictionary, along with its CV value, so if there was a better pipeline that was found, it would still be there - it just meant that, in an expected improvement sense, the "bar" was being set too high and Optuna did not know whether the new pipeline was actually an improvement or not. In order to remove this confusion and avoid any further problems, the entire process was converted to a maximisation problem for both TPOT and BO.

[^2]: To save a little bit of computational effort in the first iteration, some of the original TPOT data is used, but as the population for subsequent iterations is potentially changed by each BO step, all other TPOT data must be generated in real-time.

[^3]: Somer older versions of the Spyder IDE for the Anaconda distribution cannot handle user input from the keyboard. If you are using the Spyder IDE and it keeps crashing when asking for confirmation, ensure this parameter is set to `False` or run the main script from a terminal console or other IDE.