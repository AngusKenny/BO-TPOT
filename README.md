# Augmented TPOT (aTPOT) Suite
This set of tools augments the [Tree-based Pipeline Optimisation Tool (TPOT)](http://epistasislab.github.io/tpot/) Python library for automated machine learning. It does so by employing a number of techniques, such as Bayesian optimization (BO) or optimal computing budget allocation (OCBA) in either a post hoc or in hoc fashion. 
<!---(By employing [Optuna](https://optuna.org/), a hyperparameter optimisation Python library, Bayesian optimisation can be applied to an existing TPOT pipeline to improve its quality of prediction by identifying real-valued hyperparameters of the machine learning components of the pipeline. --->


## Dependencies
[TPOT](http://epistasislab.github.io/tpot/), [PyTorch](http://pytorch.org), [Optuna](https://optuna.org/) - and various packages from [Anaconda Distribution](http://anaconda.org).

## Introduction
Regression problems are regularly encountered in practice where there is a need to identify a model that captures the relationship between independent variables/features and the dependent variable/response. The performance of the model is often characterised by the complexity of the model and the accuracy of prediction. There are many different types of machine learning models that can be used to deal with such problems and each of such machine learnbing models have their own unique set of hyperparameters. These hyperparameters can be real values, integers or even categories, and controls the predictive performance of the model. Machine learning models can be employed singularly, or combined; using the output of one model as the input to the next, harnessing the different strengths of multiple models simultaneously. When combined in this manner, the models are collectively known as a _pipeline_. Over the last decade there has been significant research effort dedicated towards automating the process of identifying promising machine learning pipelines.

There are three key decisions to consider when designing machine learning pipelines:
- which models to select; 
- how they are organised/structured, relative to each other; and, 
- what are the values of their respective hyperparameters. 

TPOT is a Python libarary, built on top of the [Distributed Evolutionary Algorithms in Python (DEAP)](https://github.com/deap) library, which is designed to automate design of machine learning pipelines. It represents pipelines as tree-based data structures and constructs them using the genetic programming (GP) methods provided by DEAP. 

One significant limitation of TPOT is that for all its GP bells-and-whistles, it is still employs a grid-search to assign the hyperparameters to the constituent machine learning models. Since many hyperparameters are real-valued, TPOT descretises the continuous search spaces with a certain granularity. Although such a discretization is effective method to reduce the search space, the fact still remains that unless the global optimum value for a given hyperparameter lies on the exact point of descretisation, TPOT will never be able to find it.

[Optuna](https://optuna.org/) is a hyperparameter optimisation library for Python which uses Bayesian optimisation to tune hyperparameter values. It uses a Tree-based Parzen Estimator (TPE)-based surrogate model, built on historical information, to estimate and suggest hyperparameter values, which are then evaluated and the model subsequently updated. Unlike TPOT,  Optuna has no limitations on the type of values the hyperparameters can take. However it is not as effective at selecting models or structuring models in a pipeline as TPOT. 

We can think of TPOT as an effective tool for pipeline _exploration_ and BO as effective tool for pipeline _exploitation_. By using Optuna to _fine-tune_ the coarser results produced by TPOT, the algorithims in the BO-TPOT suite can potentially harness the strengths of both of these powerful tools. The question then becomes, "which pipelines should be selected for improvement by BO?", and exploring the many possibilities for this choice is the purpose of most of the tools in this suite.

Additionally, TPOT has some limitations itself, particularly in the way it selects its active population for each generation using a non-dominated sort, based on CV error and number of operators. Biasing the selection towards pipelines with fewer operators is important to prevent the GP from running out of control; however, doing so can severely limit the diversity of the population, meaning a smaller region of the search space is explored. We suggest an alternative method of selecting the active population, based on the optimal computing budget allocation (OCBA) algorithm, which takes into account the mean and standard deviation of similar pipelines when making its selection.

---

## Tools in the aTPOT suite
There are two main classes of tools in the aTPOT suite:

1. _Post hoc_ methods. These are applied after an initial TPOT search (typically, 80 generations) has completed. They include:
    - `TPOT-BO-S`, which selects a single candidate (based on CV error) to improve with BO;
    - `TPOT-BO-H`, which allocates a BO budget to many candidates, proportional to the number of hyper-parameters it contains, using a successive halving strategy to reduce the number of candidates in subsequent generations;
    - `TPOT-BO-O`, which operates similarly to `TPOT-BO-H`, however BO budget is allocated according to OCBA principles (specifically [OCBA-m for subset selection](https://github.com/CLAHRCWessex/subset-selection-problem)), reducing the number of candidates in each subsequent generation.  
&nbsp;

2. _In hoc_ methods. These are applied in tandem with TPOT, starting from generation zero. They include:
    - `TPOT-BASE`, which provides an interface to perform a base-line search using the un-altered TPOT algorithm, and store its output in a format compatible with the rest of the aTPOT suite;
    - `TPOT-BO-ALT`, which performs a prescribed number of alternating TPOT and BO optimisation steps;
    - `TPOT-BO-AUTO`, which automatically decides whether to perform a TPOT or BO step, based on gradient information;
    - `oTPOT-BASE`, which replaces the non-dominated sort-based population selection mechanism that TPOT uses, in favour of one based on OCBA principles.

---
## File organisation
Each tool has the option to write tracking and output data to a file. There are different conventions and types of data tracked for each individual tool, however they all follow the same over-arching file organisation structure:

`<root>/<results dir>/<problem name>/<method name>/Seed_<seed number>/`

**NB:** previous versions (i.e., before the change of name from BO-TPOT) followed a different file organisation convention. Any data generated using the deprecated convention can be easily updated by running the `BO-TPOT_to_aTPOT_file_organisation_patch.py` script from the root directory.

---

## Pipeline structures
During its operation, TPOT can be thought of as searching a hierarchy of two distinct classes of spaces. The first is the space of all possible combinations of operators, which it explores by using genetic programming to recombine and mutate tree representations of previously evaluated pipelines. The second is the sub-space of all possible hyper-parameter combinations for each unique combination of operators, which it explores using a grid-based search (having discretised any continuous parameter spaces). 

As it is used for tuning hyper-parameters, BO can only operate on this second sub-space and can only manipulate the value of hyper-parameters, not their relation to each other. Similarly, OCBA-based methods rely on statistical information to function, and therefore need multiple samples from the same concept. Because of this, it is useful to have a method of grouping pipelines together by their pipeline structure. 

A unique pipeline is said to have the same structure as another unique pipeline if both have the same set of hyper-parameters, in the same order, but with at least one disagreeing in value. As a given operator will have the same hyper-parameters, regardless of its position in the pipeline, a pipeline structure can be represented by its operators alone.

The tools in this suite use a tree-bracket representation to denote pipeline structure. For example, given the TPOT pipeline string:

```text
OpA(OpB(input_matrix, OpC(input_matrix, OpC__paramC1=0.5), OpB__paramB1=catX), OpA__paramA1=True, OpA__paramA2=2)
```
which represents the tree:
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
the bracket representation of the pipeline structure would be:

```text
{OpA{OpB{input_matrix}{OpC{input_matrix}}}}
```
Note that the `input_matrix` terminal is still indicated, whereas all hyper-parameter terminals are ignored.

---
## Description of operation
The tools are found as classes in separate Python files in the directory `aTPOT` and may be run independently of each other. However, all of the post hoc methods rely on previously generated pipelines and require that `TPOT-BASE` has been run _at some stage_.

Each class has a constructor and a single `optimize` method, into which the training data `X_train` and `y_train` must be passed. The `optimize` method may also be called with an optional keyword `out_path` which specifies a directory that the output files should be written to (if required). 

When the `optimize` method is called with `out_path` specified, pickling is used to save the progress of the search with each generation, to allow the search to be continued from near where it left off, should it be interrupted. The pickle file is located in `/<out_path>/<method_name>.pickle` - e.g.,`./Results/Prob1/TPOT-BASE/Seed_42/TPOT-BASE.pickle` - and is deleted at the end of a successfully completed search. It should be noted that resuming a search in this way makes exact reproduction of results almost impossible, as the random seed will be reset mid-way through the search. To start the search from scratch after an interruption, simple delete the `.pickle` file.

The processing of parameters and running and tracking of the tools in the aTPOT suite can be performed easily using the `TestHandler` class in the `utils` directory, but they can also be accessed separately for use in customised code.

The following is a detailed description of these three classes and their constructors and `optimize` methods.

---

### `TPOT-BASE`:
This method uses TPOT to generate the initial pipelines for use as control data for comparison.

#### Constructor parameters:

The table below lists all parameter arguments, their type and default values:

| Parameter          | Type     | Default                   | 
|:-------------------|:--------:|--------------------------:|
|`n_gens`            | int      | 100                       |
|`pop_size`          | int      | 100                       |
|`seed`              | int      | 42                        |
|`config_dict`       | dict     |`default_tpot_config_dict` |
|`n_jobs`            | int      | -1                        |
|`vprint`            | `Vprint` |`u.Vprint(1)`              |
|`pipe_eval_timeout` | int      | 5                         |

#### Constructor operation:

When the constructor is called, the class attributes are initialized and a TPOT regressor object is created with the following parameters:

```python
tpot = TPOTRegressor(generations=n_gens-1,
                      population_size=pop_size, 
                      mutation_rate=0.9, 
                      crossover_rate=0.1, 
                      cv=5,
                      verbosity=tpot_verb, 
                      config_dict=config_dict, 
                      random_state=seed, 
                      n_jobs=n_jobs,
                      warm_start=False,
                      max_eval_time_mins=pipe_eval_timeout)
```

#### `optimize` operation:

Having created the TPOT object in the constructor, it is fitted to the training data over the specified number of generations. Here, we use `n_gens-1` because the methods are compared by the number of pipeline evaluations and when counting generations, TPOT does not include generation 0 - and all piplines in generation 0 still need to be evaluated. Mutation and crossover rates are set to standard values, and the rest of the parameters are discussed below in the 'Running BO-TPOT' section of this document.

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
TPOT operator parameters can be continuous-valued, integer, categorical or boolean and the naming convention is `<operator name>__<parameter name>=<parameter value>`.

If `out_path` is specified, the evaluated individuals dictionary is written in its entirety to the file:

`<out_path>/TPOT-BASE.pipes` 

in the format:

`<TPOT pipe string>;<generation of pipe>;<CV value of pipe>`

and the details of the execution are written to:

`<out_path>/TPOT-BASE.progress` 

The generated pipelines are accessible as the class attribute `pipes`.

---

### `TPOT-BO-S`:
This method takes data previously generated by `TPOT-BASE` and uses Bayesian optimisation technique provided by the Optuna hyperparamter optimisation library for Python to improve the best pipeline that was found by TPOT.

#### Constructor parameters:

The table below lists all parameter arguments, their type and default values:

| Parameter              | Type     | Default                   | 
|:-----------------------|:--------:|--------------------------:|
|`init_pipes`            | dict     |                           |
|`seed`                  | int      | 42                        |
|`n_bo_evals`            | int      | 2000                      |
|`discrete_mode`         | bool     | `True`                    |
|`restricted_hps`        | bool     |`False`                    |
|`optuna_timeout_trials` | int      | 100                       |
|`config_dict`           | dict     |`default_tpot_config_dict` |
|`pipe_eval_timeout`     | int      | 5                         |
|`vprint`                | `Vprint` |`u.Vprint(1)`              |

#### Constructor operation:

After initializing the class attributes, a TPOT object is created with the following parameters:
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
This is similar to above, with a few differences. Firstly, zero generations are used as we just want the TPOT object for the purposes of evaluating pipes. Secondly, population size is set at 1, because we only want to evaluate one pipeline at a time - it is for this reason we set `n_jobs` to 1 as well. The configuration dictionary `tpot_config_copy` is a deepcopy of `config_dict` that is created at the start of each run, as Python dictionaries store information by reference, so any change made in one run will be carried over until the next. Finally, `warm_start` is set to `True`, otherwise TPOT deletes all parameter sets (psets) and current population data after it finishes its operation.

The parameter `init_pipes` specifies a population of TPOT pipelines which have been previously evaluated, after which BO should be applied[^1]. This population is searched for the best pipeline, and all other pipelines which have matching structures to the best pipeline. A pipeline is said to have a matching structure if it has the same operators, in the same order, but with different parameters. For example, the pipeline:
```text
OpA(OpB(input_matrix, OpC(input_matrix, OpC__paramC1=0.7), OpB__paramB1=catY), OpA__paramA1=False, OpA__paramA2=4)
```
would be considered to have the same structure as the example pipeline given above in the `TPOT-BASE` operation description.

TPOT is built on top of the [Distributed Evolutionary Algorithms in Python (DEAP)](https://github.com/deap) library, and uses many of its data structures and methods in its operation. A new individual is created using the `creator.Individual.from_string` method from this libraray, using the string representing the previously identified best pipeline and `tpot._pset` as its arguments, and substituted as the only element in the `tpot._pop` list. The TPOT object is now reinitialised by "fitting" it to the training data; however, because there already exist entries in the evaluated individuals dictionary which match the current population, it does not evaluate the pipeline, and just uses the CV value provided by the evaluated individuals dictionary. The `evaluated_individuals_` dictionary is replaced by the set of matching pipelines.

Finally a `TPOT_BO_Handler` object is initialized as well. This class contains all of the methods used by tools in the BO-TPOT suite to interface between TPOT and Optuna, and manage the various populations.

**NB:** if the `restricted_hps` flag is set to `True` then behaves as `TPOT-BO-Sr` (see below)


#### `optimize` operation:

The method `string_to_params` from `tpot_utils.py` is used to convert the matching pipelines to sets of parameter-value pairs and passed, along with the training data, to the `optimize` method of the `TPOT_BO_Handler` object.  This method sets up an Optuna study using the multivariate Tree-based Parzen Estimator (TPE) sampler with the matching pipelines as initial seed samples. This study must be set up as a maximisation problem, as TPOT reports its CV values as negative[^2]. When processing the matching pipelines as the initial seeds for the Optuna study, a dictionary of the distribution for each parameter in the pipeline must be generated. This is done using the `make_optuna_trial_cont` and `make_optuna_trial_discrete`. Due to a precision quirk in the way TPOT descretises continuous search spaces, sometimes it will provide hyperparameters such as `RandomForestRegressor__max_features=0.7500000000000001`. When working in continuous hyperparameter spaces, this is no problem; however, when `real_vals` is set to `False`, it is important to ensure that any numerical values are rounded to an appropriate precision, because Optuna will not accept the value `0.7500000000000001` as being part of the discrete distribution `[..., 0.7, 0.75, 8, ...]`.

A callback object is also established in this method to determine when the optimisation process should stop. We are only counting the number of pipeline evaluations, not number of Optuna "trials", therefore the stopping condition should be when the size of `tpot.evaluated_individuals_` is equal to the number of evaluations required `n_bo_evals` (plus however many matching pipelines we started with), as TPOT only adds to this dictionary if it performs a full evaluation of a pipeline. In cases where there is a discrete search space (e.g., when all hyperparameters are categorical or bounded integers) it is possible that the entire search space will be exhausted before the required dictionary size is achieved, causing Optuna to run indefinitely. In these cases, a second stopping condition is triggered, which says that if there have been `optuna_timeout_trials` trials without any change in the size of `tpot.evaluated_individuals_`, then the `study.stop()` method should be called. The counter for this trigger is reset every time the size of the dictionary is increased.

An `Objective` object is created with the training data and other relevant arguments, and passed to the Optuna `study.optimize` method. Each call to `Objective` first makes a call to `make_hp_space_cont` or `make_hp_space_discrete` (depending on the value for the `discrete_mode` boolean flag) from `bo_utils.py` with the current trial object and parameter names for the hyperparameters for arguments. These methods establish the hyperparameter space and distributions which Optuna should suggest values from. The values were taken directly from the [TPOT configuration files on GitHub](https://github.com/EpistasisLab/tpot/blob/master/tpot/config/regressor.py) (with some inference made as to the required distribution types). 

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

Although TPOT employs a grid-search algorithm requiring a discrete hyperparameter search space, Optuna has no such limitations. This means that Optuna can suggest any value from a distribution, be it categorical, bound (or unbounded) integer, uniform real or even log uniform real - the challenge is in making the two methods interact seamlessly with each other. Once the new hyperparameter values have been suggested, `Terminal` objects for each of the new hyperparameters are created and inserted into the `PrimitiveTree` object for the single individual in the population. As well as this, the new hyperparameter values must be added to `tpot.operators` and `tpot._pset`, in order to "trick" TPOT into accepting these new values as part of the original, discrete, set of choices. Having updated these two data structures, the fitness values for the individual are deleted, to mark it for evaluation, and `tpot._evaluate_individuals` is called. This updates the fitness values tuple and the CV value is returned as the score for that Optuna trial. These values are tracked in the `TPOT_BO_Handler` object and the next trial is performed, until either of the stopping conditions mentioned earlier are met.

Once the stopping conditions are met, (if the `out_path` parameter is set) the `tpot.evaluated_individuals_` dictionary is written to the file:

`<out_path>/TPOT-BO-S.pipes`

using the same convention as before, and the general run information is written to the file:

`<out_path>/TPOT-BO-S.progress`

The generated pipelines are accessible as the class attribute `pipes`.

---

### `TPOT-BO-ALT`:
This method is similar in operation to `TPOT-BO-S`, except that instead of performing a single long TPOT execution and then a single long execution of BO to improve on it, the algorithm alternates between TPOT and BO in an interative fashion.

#### Parameters:

The table below gives all parameter arguments, their type and default values:

| Parameter              | Type     | Default                   | 
|:-----------------------|:--------:|--------------------------:|
|`init_pipes`            | dict     | `{}`                      |
|`n_iters`               | int      | 10                        |
|`pop_size`              | int      | 100                       |
|`n_tpot_gens`           | int      | 8                         |
|`n_bo_evals`            | int      | 200                       |
|`seed`                  | int      | 42                        |
|`discrete_mode`         | boolean  | True                      |
|`optuna_timeout_trials` | int      | 100                       |
|`config_dict`           | dict     |`default_tpot_config_dict` |
|`n_jobs`                | int      | -1                        |
|`pipe_eval_timeout`     | int      | 5                         |
|`vprint`                | `Vprint` |`u.Vprint(1)`              |

#### Constructor operation:

After initializing the class attributes, a TPOT object is created with the following parameters:
```python
tpot = TPOTRegressor(generations=0,
                      population_size=pop_size, 
                      mutation_rate=0.9, 
                      crossover_rate=0.1, 
                      cv=5,
                      verbosity=tpot_verb, 
                      config_dict=config_dict, 
                      random_state=seed, 
                      n_jobs=n_jobs,
                      warm_start=True,
                      max_eval_time_mins=pipe_eval_timeout)
```

The starting population from the `init_pipes` dictionary are created as pipeline objects and added to the TPOT population, with the information from the remaining pipelines being added to the `tpot.evaluated_individuals_` dictionary to speed up the evaluation process[^3]. The TPOT object is initialized with `tpot.fit_init`, to generate the `tpot.pset`, and a "master" `TPOT_BO_Handler` object is also created to give acces to the methods that interact with the TPOT object and provide an interface between TPOT and Optuna.

#### `optimize` operation:

The operation of `TPOT-BO-ALT` is similar to `TPOT-BO-S`, except that instead of performing a single BO step, it performs a series of iterations of TPOT and BO, in an alternating fashion.

With each iteration, `tpot.fit` is called, using the training data as input arguments, for `n_tpot_gens` generations (taking care to ensure that for the first iteration it runs for `n_tpot_gens-1` generations to account for the fact that TPOT doesn't include generation 0 in its count)[^4]. The best pipeline from the evolved population is identified, along with any others in the `tpot.evaluated_individuals_` dictionary that have the same structure. A new TPOT object is created using the parameters:

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

A new deep copy of the `config_dict` is used when establishing this TPOT object, as any changes made to the dictionary should not be carried over from iteration to iteration. A new `TPOT_BO_Handler` object is also created to manage this TPOT object. The hyperparameters of the pipeline to be optimised with BO are added to `bo_tpot._pset` and `bo_tpot.operators`, as they might contain values suggested by the BO step in previous iterations, which are not present in the original `config_dict`. The pipeline to be optimised (and other pipelines with matching structures) is then added to the `bo_tpot.evaluated_individuals_` dictionary and fit to the training data, before being optimised using Optuna in the same way as `TPOT-BO-S`, except only for `n_bo_evals` (or until the other stopping condition is met).

If the BO step is a success (i.e., produces a pipeline with better CV error than any in the TPOT population) the main `tpot._pset` and `tpot.operators` are updated with the new hyperparameter values and the previous best pipeline is replaced by the new, improved pipeline; if no improvement is found, the main population stays the same. 

If the `out_path` parameter is set, the evaluated pipelines are written to:

`<out_path>/TPOT-BO-ALT.pipes`, 

with the format:

`<pipeline>;<iteration>;<generation>;<source>;<cv>`

and it moves onto the next iteration. Finally, the progress is recorded in the file:

`<out_path>/TPOT-BO-ALT.progress`

The generated pipelines are accessible as the class attribute `pipes`. The dictionary for pipeline has a `source` entry to indicate whether it was generated by TPOT (`TPOT-BO-ALT(TPOT)`) or BO (`TPOT-BO-ALT(BO)`).

---

### `TPOT-BO-AUTO`:
This method is very similar in operation to `TPOT-BO-ALT`, except that instead of performing a fixed number of alternations between TPOT and BO, gradient information is used with each TPOT generation (or generation equivalent of BO evaluations) to determine which method should be used for the next generation.

#### Parameters:

The table below gives all parameter arguments, their type and default values:

| Parameter              | Type     | Default                   | 
|:-----------------------|:--------:|--------------------------:|
|`init_pipes`            | dict     | `{}`                      |
|`pop_size`              | int      | 100                       |
|`n_gens`                | int      | 100                       |
|`seed`                  | int      | 42                        |
|`discrete_mode`         | boolean  | True                      |
|`optuna_timeout_trials` | int      | 100                       |
|`config_dict`           | dict     |`default_tpot_config_dict` |
|`n_jobs`                | int      | -1                        |
|`pipe_eval_timeout`     | int      | 5                         |
|`vprint`                | `Vprint` |`u.Vprint(1)`              |


#### Constructor operation:

The constructor for `TPOT-BO-AUTO` operates the same as `TPOT-BO-ALT`.

#### `optimize` operation:

`TPOT-BO-AUTO` operates very similarly to `TPOT-BO-ALT`, except that it removes the arbitrary nature of specifiying a number of iterations to divide the total budget into, instead deciding on a generation-by-generation basis, what method should be used. If TPOT is used then the entire population is evolved for a single generation, if BO is used, then `pop_size` evaluations are performed on the best pipeline and pipelines with matching structures to it.

If the `out_path` parameter is set, the evaluated pipelines are written to:

`<out_path>/TPOT-BO-ALT.pipes`, 

with the format:

`<pipeline>;<generation>;<source>;<cv>`

and gradient information written to:

`<out_path>/TPOT-BO-ALT.pipes`, 

Following that, it moves onto the next iteration. Finally, the progress is recorded in the file:

`<out_path>/TPOT-BO-ALT.progress`

The generated pipelines are accessible as the class attribute `pipes`. The dictionary for pipeline has a `source` entry to indicate whether it was generated by TPOT (`TPOT-BO-AUTO(TPOT)`) or BO (`TPOT-BO-AUTO(BO)`).

---

### `TPOT-BO-H`:
One of the main issues identified from experiments using `TPOT-BO-S`, `TPOT-BO-ALT`, and `TPOT-BO-AUTO` was the fact that only one pipeline is selected for improvement by BO at any one time, which introduces a non-trivial amount of luck. `TPOT-BO-H` aims to address this issue by iteratively applying BO to a sub-population of pipelines, the size of which reduces according to a successive halving-based scheme.

Having completed $nG_s$ generations, assuming the typical values of $nG_s=80$ and $nP=100$, the set of evaluated solutions produced by TPOT $S$, should have a size of around 8,000. If $S$ is ordered by CV value and some subset of the best performing pipelines are picked, it is likely that this subset would have a very low diversity, in terms of pipeline structure - i.e., pipelines with the same operators, ordered in the same way, but with different hyper-parameter values. This is not necessarily because one particular pipeline structure is much more promising than any other, but more an artefact of the manner in which TPOT selects its parent population for each generation. When deciding on which set of pipelines to carry forward to generate new offspring, TPOT performs a non-dominated sort on the set of evaluated pipelines, minimising CV error and number of operators. The number of operators is a metric that approximates the complexity of a given pipeline, and keeping this complexity low is important for the operation of the genetic programming algorithm that TPOT is built on. An unfortunate consequence of using this, rather coarse, metric is that unless the first set of hyper-parameters chosen produces a pipeline with a very low CV error, any offspring with a higher number of operators is unlikely to be chosen for the next generation, and therefore will not have many opportunities to find a set of hyper-parameters that do work. This selection bias has an effect on the diversity of the evaluated pipelines, and can result in TPOT becoming "fixated" on one particular pipeline structure from very early on in the search. By first partitioning $S$ by pipeline structure - producing $Q$ - and then sorting $Q$ by CV error, a much more diverse set of candidates for BO improvment can be obtained. 

Initially, $|Q| = \frac{nP}{2}$ candidate pipeline structures are selected for improvement using `TPOT-BO-S`. With each generation the size of this sub-population is halved and the allocated budget for each generation is calculated, based on how many "halvings" remain until $|Q| = 1$. Let $B_r$ be the total remaining budget, the budget for the current generation can be computed as ${B_g = \frac{B_r}{\lceil\log_2(|Q|)+1 \rceil}}$. As the size of the BO search space increases exponentially with the number of hyper-parameters, this budget is allocated among the pipeline structures in $Q$, proportional to the number of number of hyper-parameters they contain that can take 2 or more values. If $H_q$ is the set of hyper-parameters for structure $q \in Q$, then the budget allocation for $q$ is ${A_q = \lceil \frac{|H_q| \times B_g}{\sum_{r \in Q} |H_r|} \rceil}$.

With each generation, $B_g$ and $A$ are calculated, and used to apply `TPOT-BO-S` to each pipeline structure in $Q$, proportional to the number of hyper-parameters. Once all pipline structures have had BO applied and are ordered by the best evaluated CV, the best $\lceil \frac{|Q|}{2} \rceil$ are selected for the next generation and the total remaining budget $B_r$ is updated. This continues until $|Q| = 1$ at which point the remainder of the budget is applied at once. 

As with other BO-based methods, `TPOT-BO-H` can operate with both discrete and continuous parameter spaces, however there is also a third mode `TPOT-BO-Hs`. This `s` stands for "sequential" and runs `TPOT-BO-H` with the standard TPOT discrete parameter spaces, right until $|Q|=1$, at which point it switches to continuous parameter spaces for the final application of BO. The reasoning behind this is that as the BO budget is being spread across many pipelines, there is not a lot of time to explore the search space and construct meaningful BO models for all of the structures in $Q$, especially those which have very few initial samples. Using a coarser, discrete parameter space initially can help the algorithm explore more efficiently, while switching to a continuous space at the end of the search allows the algorithm to "fine tune" the parameters and search in between the discretised steps.

### **Parameters:**

The table below gives all parameter arguments, their type and default values:

| Parameter              | Type     | Default                   | 
|:-----------------------|:--------:|--------------------------:|
|`init_pipes`            | dict     | `{}`                      |
|`seed`                  | int      | 42                        |
|`pop_size`              | int      | 100                       |
|`n_bo_evals`            | int      | 2000                      |
|`discrete_mode`         | boolean  | True                      |
|`optuna_timeout_trials` | int      | 100                       |
|`config_dict`           | dict     |`default_tpot_config_dict` |
|`pipe_eval_timeout`     | int      | 5                         |
|`n_jobs`                | int      | -1                        |
|`vprint`                | `Vprint` |`u.Vprint(1)`              |

### **Pseudocode:**

**Input:** $D$: training data; $S$: input pipeline set; $nE$: number of BO evaluations; $nP$: TPOT population size; $\rho$: GP parameter set  
**Output:** $S$: evaluated pipeline set  
> $Q$ &larr; top $\frac{nP}{2}$ from partition of $S$ by pipeline structure, ordered by CV  
> $B_r$ &larr; $nE$  
> **while** $B_r > 0$ **do:**  
>> $B_g$ &larr; $\frac{B_r}{\lceil\log_2(|Q|)+1 \rceil}$ **if** $|Q| > 1$ **else** $B_r$  
>> $A$ &larr; allocation of $B_g$ for each structure $q \in Q$, proportional to number of hyper-parameters  
>> **for** $q \in Q$ **do:**  
>>> $q$ &larr; `TPOT-BO-S`$(D,q,A_q,\rho)$  
>>
>> $S$ &larr; $S \bigcup_{q \in Q}q$  
>> $B_r$ &larr; $B_r - \sum_{q \in Q} A_q$  
>> $Q$ &larr; top $\lceil \frac{|Q|}{2} \rceil$ pipeline structures in $Q$, ordered by CV  
>
> **return** S

### **Outputs:**  
If the `out_file` parameter is set when calling the `optimize` method, 2 files are produced as output:

`TPOT-BO-H{d/c/s}.progress` - provides general information about the run

`TPOT-BO-H{d/c/s}.pipes` - provides full list of pipelines evaluated during the entire search, with each semi-colon separated line line taking the form:
>`<pipeline string>;<structure string>;<generation>;<number of structures>;<source>;<evaluated CV error>`  

---

### `TPOT-BO-O`:
The hyper-parameter values of a pipeline can take different variable types, and have varying degrees of effect on its fitness. This implies that allocating computational budget proportional to the number of hyper-parameters alone is unlikely the best strategy, as the number of hyper-parameters alone does not necessarily reflect the size of the search space, or difficulty of the problem (e.g., all hyper-parameters could be True/False). The optimal computing budget allocation (OCBA) algorithm uses the mean and standard deviation for a number of samples from different concepts to determine what proportion of the available budget should be allocated to each one. By using statistical information like this, OCBA prioritises regions of the search space that not only have the best sample mean (and are therefore more likely to have been exploited already), but also potentially promising regions, which have a high level of uncertainty.

In principle, `TPOT-BO-O` functions similarly to `TPOT-BO-H`, with a couple of differences. After $nG_s$ (typically 80) TPOT generations, the set of evaluated solutions $S$ is partitioned into pipeline structures with more than one hyper-parameter and ordered by CV. The top $\frac{nP}{2}$ pipeline structures are selected, producing $Q$, and `TPOT-BO-S` is applied to each one as needed, such that the number of _valid_ (i.e., CV $\neq \infty$) evaluated pipelines for a given structure is $|Q[i]| \geq n_0$, with $n_0$ typically 10. Because TPOT uses a non-dominated sort-based method of selecting its parent population for each generation, the search tends to focus on narrow regions of the search space, with many generated pipeline structures only being evaluated once or twice. To make its allocations, OCBA requires the mean and standard deviation of multiple samples, so these initial evaluations aid in the production of more accurate statistics.

In OCBA, the $\Delta$ parameter is often used to specify how many evaluations are made between each statistics update. This parameter is used in `TPOT-BO-O` as a loose control on how focused the search is. A high value for $\Delta$ means that more budget is available to allocate with each iteration, allowing it to potentially spread across many candidates; while a low value forces OCBA to narrowly allocate computing resources, while also reassessing this allocation more frequently to allow it to change its area of attention if need be. Starting with $\Delta = \frac{nP}{2}$, a similar principle of successive halving as was used in `TPOT-BO-H` is applied to $\Delta$, with the intent of focusing the search more with each generation. 

As with `TPOT-BO-H`, $\Delta$ is halved with every generation and the budget for the current generation is computed based on how many halvings remain until $\Delta=1$. Let $B_r$ be the total remaining computing budget for the entire search, the budget to be allocated in the current generation is calculated as ${B_g = \frac{B_r}{\lceil\log_2(\Delta)+1 \rceil}}$. Once $\Delta=1$, single selections and evaluations are carried out until the budget is fully expended.

As with the other BO-based methods, `TPOT-BO-O` can operate using both discrete and continuous BO parameter spaces.

### **Parameters:**

The table below gives all parameter arguments, their type and default values:

| Parameter              | Type     | Default                   | 
|:-----------------------|:--------:|--------------------------:|
|`init_pipes`            | dict     | `{}`                      |
|`seed`                  | int      | 42                        |
|`pop_size`              | int      | 100                       |
|`n_bo_evals`            | int      | 2000                      |
|`discrete_mode`         | boolean  | True                      |
|`optuna_timeout_trials` | int      | 100                       |
|`config_dict`           | dict     |`default_tpot_config_dict` |
|`pipe_eval_timeout`     | int      | 5                         |
|`n_jobs`                | int      | -1                        |
|`n_0`                   | int      | 10                        |
|`vprint`                | `Vprint` |`u.Vprint(1)`              |

### **Pseudocode:**

**Input:** $D$: training data; $S$: input pipeline set; $nE$: number of BO evaluations; $nP$: TPOT population size; $n_0$: initial number of OCBA samples; $\rho$: GP parameter set  
**Output:** $S$: evaluated pipeline set  
> $Q$ &larr; top $\frac{nP}{2}$ from partition of $S$ by pipeline structure, ordered by CV  
> **for** $q \in Q$ **do:**  
>> $q$ &larr; `TPOT-BO-S`$(D,q,\max(0,n_0-|q|),\rho)$  
>
> $B_r$ &larr; $nE + |S| - \sum_{q \in Q}|q|$  
> $S$ &larr; $S \bigcup_{q \in Q}q$  
> $\Delta$ &larr; $nP$  
> **while** $B_r > 0$ **do:**  
>> $\Delta$ &larr; $\lceil\frac{\Delta}{2} \rceil$  
>> $B_g$ &larr; $\frac{B_r}{\lceil\log_2(\Delta)+1 \rceil}$ **if** $\Delta > 1$ **else** 1  
>> $B_r$ &larr; $B_r - B_g$  
>> **while** $B_g > 0$ **do:**  
>>> $\mu,\sigma$ &larr; mean, standard deviation for each structure $q \in Q$  
>>> $A$ &larr; OCBA budget allocations for $q \in Q$ using parameters $\mu,\sigma,\min(B_g,\Delta)$  
>>> **for** $q \in Q$ **do:**  
>>>> $q$ &larr; `TPOT-BO-S`$(D,q,A_q,\rho)$  
>>>
>>> $S$ &larr; $S \bigcup_{q \in Q}q$  
>>> $B_g$ &larr; $B_g - \sum_{q \in Q} A_q$
>
> **return** $S$

### **Outputs:**  
If the `out_file` parameter is set when calling the `optimize` method, 3 files are produced as output:

`TPOT-BO-O{d/c}.progress` - provides general information about the run

`TPOT-BO-O{d/c}.tracking` - provides generation-by-generation of tracking in the form of an ASCII histogram

`TPOT-BO-O{d/c}.pipes` - provides full list of pipelines evaluated during the entire search, with each semi-colon separated line line taking the form:
>`<pipeline string>;<structure string>;<generation>;<delta value>;<evaluated CV error>`  

---

### `oTPOT-BASE`:
By default, TPOT performs a non-dominated sort on the set of evaluated pipelines, when selecting the parent population for each generation, minimising both CV error and number of operators. While this method ensures pipeline complexity does not run out of control during the search - an important consideration for GP based methods - it can severely limit the diversity of the parent population.

In `oTPOT-BASE` this method of parent population selection is replaced with one based on OCBA principles. With each generation, the set of evaluated solutions $S$, is partitioned by structure, producing $Q$. The mean CV and its standard deviation statistics are computed for each structure in $Q$. Due to the random selection of hyper-parameters by TPOT, the standard deviation of the CV can be **very** high or sometimes zero when all pipelines produced the same CV, or only one pipeline has been evaluated from a given structure. Therefore, the standard deviation is capped at `1E+10` from above and `1e-10` from below. A modified version of the OCBA algorithm is used with a total budget equal to the population size $nP$ to produce allocations $A$ for each pipeline structure in $Q$. The modifications introduce an allowed upper bound for each allocation, ensuring that the allocated budget for a given structure $q \in Q$ does not exceed the number of pipelines exhibiting it, $|q|$.

These allocations are used to select the best $A_q$ pipelines from pipeline structure $q \in Q$ and add them to a new parent population $P$. This population is then evolved and fitted to the input data.

### **Parameters:**

The table below gives all parameter arguments, their type and default values:

| Parameter              | Type     | Default                   | 
|:-----------------------|:--------:|--------------------------:|
|`pop_size`              | int      | 100                       |
|`n_gens`                | int      | 100                       |
|`seed`                  | int      | 42                        |
|`config_dict`           | dict     |`default_tpot_config_dict` |
|`n_jobs`                | int      | -1                        |
|`pipe_eval_timeout`     | int      | 5                         |
|`allow_restart`         | boolean  | True                      |
|`vprint`                | `Vprint` |`u.Vprint(1)`              |


### **Pseudocode:**

**Input:** $D$: training data; $nG_t$: number of generations; $nP$: population size; $\rho$: GP parameter set  
**Output:** $S$: evaluated pipeline set  
> $T$ &larr; `TPOTRegressor` object initialised using $nP$ and $\rho$  
> $S$ &larr; evaluated pipeline set after fitting $T$ on $D$ for 1 generation  
> **for** $nG_t - 2$ generations **do:**  
>> $Q$ &larr; partition of $S$ by pipeline structure  
>> $\mu,\sigma$ &larr; mean, standard deviation for each structure $q \in Q$  
>> $A$ &larr; OCBA budget allocations for $q \in Q$ using parameters $\mu,\sigma,nP,|q|$  
>> $P$ &larr; $\emptyset$ &nbsp;&nbsp;&nbsp;&nbsp; empty TPOT population  
>> **for** $q \in Q$ **do:**  
>>> $P$ &larr; $P\ \cup$ best $A_q$ pipelines of structure $q \in Q$  
>>
>> $T$ &larr; replace parent population in $T$ with $P$  
>> $S$ &larr; update with result of fitting $T$ on $D$ and $S$ for 1 generation  
>
> **return** $S$

**Notes:** Main loop is performed for $nG_t-2$ generations to account for $nP$ initial evaluations and first TPOT fitting.

### **Outputs:**  
If the `out_file` parameter is set when calling the `optimize` method, 3 files are produced as output:

`oTPOT-BASE.progress` - provides general information about the run

`oTPOT-BASE.tracker` - provides tracking information for the selected parent population for each generation, with each semi-colon separated line taking the form:
>`<generation>;<pipeline structure>;<number allocated>;<best evaluated CV error>`  

`oTPOT-BASE.pipes` - provides full list of pipelines evaluated during the entire search, with each semi-colon separated line line taking the form:
>`<pipeline string>;<generation>;<evaluated CV error>`  

---

## Running aTPOT

As mentioned above, it is possible to run the `TPOT-BASE`, `TPOT-BO-S` and `TPOT-BO-ALT` processes individually, by instantiating their classes and calling their respective `optimize` methods with the appropriate training data, but the simplest way is to execute the script in the file `run_tpot_tests.py`. At the start of this script is a parameters dictionary `params` which the user can populate and is given to a `TestHandler` object which sets up and runs all the required tests, in accordance with the specified parameters, automatically.

The table below gives the parameters in this dictionary and their type, followed by a description of their functions:

| Parameter              | Type       |
|:-----------------------|:----------:|
|`CLEAN_DATA`            | boolean    |
|`RUN_TPOT-BASE`         | boolean    |
|`RUN_TPOT-BO-S`         | boolean    |
|`RUN_TPOT-BO-Sr`        | boolean    |
|`RUN_TPOT-BO-ALT`       | boolean    |
|`RUN_TPOT-BO-AUTO`      | boolean    |
|`VERBOSITY`             | int        |
|`DATA_DIR`              | string     |
|`RESULTS_DIR`           | string     |
|`RUNS`                  | int / list | 
|`PROBLEMS`              | list       |
|`TPOT_CONFIG_DICT`      | dict       |
|`nJOBS`                 | int        |
|`DISCRETE_MODE`         | boolean    |
|`PIPE_EVAL_TIMEOUT`     | int        | 
|`START_SEED`            | int        | 
|`POP_SIZE`              | int        | 
|`nTOTAL_GENS`           | int        | 
|`STOP_GEN`              | int        | 
|`OPTUNA_TIMEOUT_TRIALS` | int        |
|`nALT_ITERS`            | int        |

### `CLEAN_DATA`:
This parameter allows the user to clear **all** data in run directories that are to be written to, when starting an execution. This parameter depends on the values of `RUN_TPOT-BASE`, `RUN_TPOT-BO-S`, `RUN_TPOT-BO-ALT`, `RUN_TPOT-BO-AUTO`, `RESTRICT_BO`, `RUNS` and `PROBLEMS`, and will only delete data from whatever is flagged to execute in the current instance. When set to `True`, it will display the list of runs and problems it is about to clear, which data will be removed, and will ask for confirmation. Confirming with `y` will remove the specified data directories and continue with the execution; cancelling with `n` (or any other input) will exit the script. Once the data has been cleared it will ask for a second confirmation to ensure the user wants to continue executing the script. Selecting `y` (or any other input) here will continue as normal, selecting `n` here will exit without processing anything else, meaning this parameter can be used as an easy way to clean the `RESULTS_DIR` directory back to the initial TPOT data if needed. If this parameter is set to `False`, any existing data will still be overwritten, but only as-and-when new data is produced. If `RUN_TPOT-BASE` is set to `True` then this step is ignored, as generated TPOT data is never affected.

### `RUN_TPOT-BASE`/`RUN_TPOT-BO-S`/`RUN_TPOT-BO-Sr`/`RUN_TPOT-BO-ALT`/`RUN_TPOT-BO-AUTO`:
Boolean flag to control which of `TPOT-BASE`, `TPOT-BO-S`, `TPOT-BO-Sr`, `TPOT-BO-ALT` and `TPOT-BO-AUTO` are executed. Set to `False` to skip any of the processes, however keep in mind that both `TPOT-BO-S` requires that `TPOT-BASE` has been executed _at some stage_.

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
This is a list which specifies the problems to be batch-processed. This parameter allows the user to specify a set of problems for batch processing as a list of strings. A problem is specified by the file name (without extension) for its data file, within the `data_dir` directory. For example, if the data for the two required problems are held in `./Data/prob1.data` and `./Data/prob2.data` then `data_dir` would be `'Data'` and `prob_list` would be `['prob1', 'prob2']`. If `prob_list` is supplied as the empty list `[]`, then `data_dir` is searched for all files with the extension `.data` and will process all of them.

### `TPOT_CONFIG_DICT`:
The configuration dictionary to be used by TPOT containing all of the operators, their hyperparameters and all possible values they can take. These must all be finite and discrete, expressed as lists, integer ranges or real ranges that have been descretised using `np.arrange`. The default dictionary cannot be used directly from the TPOT library as any changes to it will persist across runs; meaning that a new (deep) copy must be made for each new TPOT object instance.

The file `tpot_config.py` provides a copy of the default TPOT configuration dictionary as `default_tpot_config_dict`. Also present in this file is `reduced_tpot_config_dict`, a configuration dictionary with a reduced subset of the default operators, useful for running small and controllable tests when debugging.

### `nJOBS`:
This parameter allows the user to specify the number of parallel jobs to run whenever TPOT has a population of solutions to simultaneously evaluate. This value is not used when creating the TPOT object used to evaluate pipelines during the BO step, or when setting up an Optuna study, as there is no advantage gained by parallelisation in these instances. When set to -1, the maximum number of cores available will be used.

### `DISCRETE_MODE`:
This boolean flag determines the type of search space the hyperparameters should be drawn from in the BO step. If set to `True` the `make_hp_space_discrete` method from `optuna_hp_spaces.py` is called which allows Optuna to suggest hyperparameter values from the exact same set of values contained in `default_tpot_config_dict`.  When set to `False`, Optuna is able to suggest hyperparameter values from continuous distributions during the BO step.

While there may still be some benefit to adding a BO step using a discrete hyperparameter space - harnessing the power of BO over simple random grid-search - it is undoubtedly the fact that it is able to produce pipelines with hyperparameter values sampled from continous spaces that gives BO-TPOT the edge over canonical TPOT. Therefore, it is highly recommended to run BO-TPOT with `DISCRETE_MODE` set to `False`.

### `PIPE_EVAL_TIMEOUT`:
This parameter allows the user to determine the maximum time allowed for a single pipeline evaluation by TPOT in minutes. The default TPOT value is 5, and it is suggested to use this value.

### `RUNS`:
This parameter allows the user to specify how many runs they want to execute or, if no new TPOT data is being generated, what specific runs they wish to apply `TPOT-BO-S` to. It can be specified as an `int` or `list`. If `RUN_TPOT-BASE` is set to `True` this parameter must be set as an `int` and not a `list`, as the `TestHandler` does not overwrite previous `TPOT-BASE` runs; rather it will create a new run directory with its number subsequent to the number of the most recent run. The algorithm determines if run data exists by whether there are any run directories (of the form `Run_XX`, where `XX` is the run number, padded with a zero for numbers less than 10) so when starting from scratch, or after changing parameter values between runs, make sure that the problem directories are clear before executing.

In the case where `RUN_TPOT` is set to `False`, this parameter may be given as a list, or range, indicating the specific runs to process. If given as an `int` then the run list will be taken as `range(RUNS)`.

### `START_SEED`:
This parameter allows the user to specify the starting seed for the first run, with subsequent runs incrementing this seed by 1 each time. It only applies to `TPOT-BASE` as the `TestHandler` object will acquire this information from the `TPOT-BASE.progress` file for each run to ensure the same seed is being used for `TPOT-BO-S`, etc. Be careful when setting the start seed, that you are not duplicating the results of a previous run.

### `POP_SIZE`:
This parameter allows the user to specify the population size when instantiating a `TPOTRegressor` object. This only applies to `TPOT-BASE` as `TPOT-BO-S` and others will acquire this information from the `TPOT-BASE.progress` file for each run.

### `nTOTAL_GENS`:
This parameter allows the user to specify the total number of generations when producing the initial TPOT data. This only applies to `TPOT-BASE` as `TPOT-BO-S` and others will acquire this information from the `TPOT-BASE.progress` file for each run.

### `STOP_GEN`:
This parameter allows the user to specify the point at which the BO step should commence for the `TPOT-BO-S` process (and is used to calculate how much computational budget should be allocated to the BO step for each iteration in `TPOT-BO-ALT`).

### `OPTUNA_TIMEOUT_TRIALS`:
As mentioned above, the main stopping condition for the BO step is when the size of the `tpot.evaluated_individuals_` dictionary indicates that the requried number of evaluations have been performed. In cases where it is possible to exhaustively search the entire hyperparameter space before this size is reached, this can result in Optuna running indefinitely. Therefore a second stopping condition is applied when no change is made to the size of the `tpot.evaluated_individuals_` dictionary within some specified number of trials. This parameter specifies this number of trials, its suggested value is 100.

### `nALT_ITERS`:
This parameter allows the user to specify the number of iterations that `run_tpot_bo_alt` should run for. It is used to divide the total computational budget between alternating instances of the TPOT and BO steps.


## `TestHandler` operation

Any of the classes in the BO-TPOT suite can be instantiated as part of a custom solver, written by the user. However, there is also a `TestHandler` class provided which implements methods which handle the general "housekeeping" aspects of running the tests, such as data storage and structuring, progress information and parameter handling. It also implements any error handling while running batch tests, allowing the batch to continue running if there should be any errors. This class is instantiated using a dictionary of parameters as is given in the `run_tpot_tests.py` description above. Once instantitated, the `set_problem` and `set_run` methods can be used to load problem data and generate directories for output files to be saved. Below is a description of the main methods in this class.

### `run_TPOT_BASE`:
This method instantiates the `TPOT_Base` class and runs its `optimize` method using the training data generated by the `set_problem` method. Once complete, the resulting pipelines are written to:

`./<RESULTS_DIR>/<problem>/<run>/TPOT-BASE/TPOT-BASE.pipes`

and the run information is written to:

`./<RESULTS_DIR>/<problem>/<run>/TPOT-BASE/TPOT-BASE.progress`

The generated pipelines are also returned for use with other methods.

### `run_TPOT_BO_S`:
This method instantiates the `TPOT_BO_S` class and runs its `optimize` method using the training data generated by the `set_problem` method and a set of initial pipelines generated by `TPOT-BASE`, passed as `init_pipes`. The `truncate_pop` utility is used to remove all pipelines that come after `STOP_GEN` from `init_pipes`, and this truncated population is passed when instantiating the `TPOT_BO_S`. It is important to truncate the set of pipelines in this way to ensure a fair comparison with `TPOT-BASE`. Once complete, the resulting pipelines are written to:

`./<RESULTS_DIR>/<problem>/<run>/TPOT-BO-S/<mode>/TPOT-BO-S.pipes`

and the run information is written to:

`./<RESULTS_DIR>/<problem>/<run>/TPOT-BO-S/<mode>/TPOT-BO-S.progress`

where `<mode>` is `discrete` if `DISCRETE_MODE` is `True`, and `continuous` otherwise.

If `TPOT-BO-Sr` is required, then `run_TPOT_BO_S` should be run with `restricted_hps=True`, in which case the pipelines are written to:

`./<RESULTS_DIR>/<problem>/<run>/TPOT-BO-Sr/<mode>/TPOT-BO-S.pipes`

and the run information is written to:

`./<RESULTS_DIR>/<problem>/<run>/TPOT-BO-Sr/<mode>/TPOT-BO-S.progress`


### `run_TPOT_BO_ALT`:

This method instantiates the `TPOT_BO_ALT` class and runs its `optimize` method using the training data generated by the `set_problem` method and an optional set of initial pipelines generated by `TPOT-BASE`, passed as `init_pipes`. It divides the total computing budget `POP_SIZE * nTOTAL_GENS` into `nALT_ITERS` chunks, and then uses the `STOP_GEN` parameter to determine the ratio of TPOT to BO evaluations. For example, if `nALT_ITERS=10', 'POP_SIZE=100`, `nTOTAL_GENS=100` and `STOP_GEN=80`, then the total computing budget would be 10,000 evaluations. Dividing this by 10 gives 1,000 evaluations per iteration, and the ratio of TPOT to BO evaluations is `(STOP_GEN)/(nTOTAL_GENS - STOP_GEN)`, meaning 800 evaluations (i.e., 8 generations at 100 population size) for TPOT and 200 evaluations for BO. The optional initial popluation is truncated to `n_tpot_gens-1` to maintain fairness in comparison.

Once complete, the resulting pipelines are written to:

`./<RESULTS_DIR>/<problem>/<run>/TPOT-BO-S/<mode>/TPOT-BO-ALT.pipes`

and the run information is written to:

`./<RESULTS_DIR>/<problem>/<run>/TPOT-BO-S/<mode>/TPOT-BO-ALT.progress`

where `<mode>` is `discrete` if `DISCRETE_MODE` is `True`, and `continuous` otherwise.


### Tracking runs:
Along with simplifying the process of running tests with the three aspects of BO-TPOT, the `TestHandler` class also automatically tracks the progress of the runs.

Each time this class is instantiated, a file in `./<RESULTS_DIR>/` called `BO_TPOT.progress` is created with the date and time the execution was started, the values of the parameters in the `params` dictionary (if the default TPOT configuration dictionary is used, then it will just say `default_tpot_config_dict`, otherwise it will give the full dictionary). As well as this general execution information, for each problem and run, the seed is given, and the outcome of any processes run, along with the time taken (if successful). If any process fails or crashes, then the error message and stack traceback will be written to this file, but the execution itself will contine with the next process or run. This helps to diagnose any minor or incidental bugs, without having to cancel the entire set of runs.


## Processing Results
The file `process_tpot_tests.py` provides a script that automatically processes the results produced by BO-TPOT, computes the statistics and generates plots for the data. By default, the script searches each problem sub-directory in the results directory to find runs which have valid results. It performs a check on each run to ensure that the parameters used to generate the results are the same as those used in the first valid run, if there is a mis-match in parameter values, or if the results themselves are invalid, it will skip the run and not include it in the final statistics or plots.

Depending on what is required, a plot is generated for `TPOT-BASE`, `TPOT-BO-S`, `TPOT-BO-Sr`, `TPOT-BO-ALT` and/or `TPOT-BO-AUTO`. 

Along with the plots, a separate file is generated with the statistics of the processed runs. This file gives the date and time the results were processed, then for each problem it gives the list of runs that were processed and then a semicolon-separated table with the best, worst, median, mean and standard deviation of all the processed runs for that problem. The file is saved to:

`./<RESULTS_DIR>/BO-TPOT.stats`

At the top of the script is a dictionary of parameters that can be set; the table below gives the parameters in this dictionary and their type, followed by a description of their functions:

| Parameter              | Type       |
|:-----------------------|:----------:|
|`RESULTS_DIR`           | string     |
|`PROBLEMS`              | list       |
|`RUN_LIST`              | list       |
|`SAVE_PLOTS`            | boolean    |
|`SAVE_STATS`            | boolean    |
|`PLOT_TPOT-BO-Sr`       | boolean    |
|`PLOT_TPOT-BO-ALT`      | boolean    |
|`PLOT_TPOT-BO-AUTO`     | boolean    |
|`DISCRETE_MODE`         | boolean    |
|`SKIP_PLOT_INIT`        | int        |

### `RESULTS_DIR`:
String to specify the the name of the directory that results data should be drawn from.

### `PROBLEMS`:
This is a list which specifies the problems to be batch-processed, in exactly the same way as it is described above. If this is set to the empty list `[]`, then `RESULTS_DIR` is searched for existing results, and all results found are processed.

### `RUN_LIST`:
This parameters allows the user to specify the runs to be processed. If this is set to the empty list `[]` it will process all valid runs it can find under the problem directory, otherwise this parameter can be give as a list or range of specific run numbers.

### `SAVE_PLOTS`:
When this parameter is set to `True`, the plots generated by the script will be saved as `.PNG` files with a `D` (for discrete) or `C` (for continuous) flag in the filename to `./<RESULTS_DIR>/<problem>/Plots/`. If set to `False`, the plots will be displayed but not saved.

### `SAVE_STATS`:
When this parameter is saved to `True`, the stats are output to the file:

`./<RESULTS_DIR>/BO-TPOT.stats`

### `PLOT_TPOT-BO-Sr`/`PLOT_TPOT-ALT`/`PLOT_TPOT-AUTO`:
By default, the results of `TPOT-BASE` and `TPOT-BO-S` are plotted, but if any of the others are not required, then their flags should be set to `False`.

### `DISCRETE_MODE`:
This boolean flag indicates whether discrete or continuous versions should be plotted.

### `SKIP_PLOT_INIT`:
If there is a very large gap between the CV scores of the initial solutions and later ones, plotting the entire set of results can mean it is hard to see any small changes after it starts converging. This parameter allows the user to specify a number of initial solutions to skip, making the plot much easier to read.


## Footnotes
[^1]: BO will be applied to whatever set of pipelines are passed to `TPOT-BO-S`. When running it to compare with previous `TPOT-BASE` executions at an earlier point, the `TPOT-BASE` population must first be truncated using the `truncate_pop` method in `tpot_utils.py`. This is done automatically by the `TestHandler` class.

[^2]: In a previous version of the code, the Optuna study was set up as a minimisation problem, which negated the CV values returned by the TPOT pipeline evaluation. However, this created some confusion and, as a result, a bug was found where the CV of the initial samples given to Optuna was not negated when the study was set up. This meant that Optuna could never find a solution with a CV value better than its initial samples, because they were input as negative values, and any subsequent CV score from the TPOT pipeline evaluations were negated and stored as positive ones in the model. This did not mean that no improvement on the pipeline was possible with BO-TPOT, as any evaluated pipeline was stored in the TPOT evaluated individuals dictionary, along with its CV value, so if there was a better pipeline that was found, it would still be there - it just meant that, in an expected improvement sense, the "bar" was being set too high and Optuna did not know whether the new pipeline was actually an improvement or not. In order to remove this confusion and avoid any further problems, the entire process was converted to a maximisation problem for both TPOT and BO.

[^3]: As with `TPOT-BO-S`, the entire population of `init_pipes` will be pre-loaded into the `tpot.evaluated_individuals_` dictionary. Therefore, if a fair comparison is required against other techniques, the `truncate_pop` method in `tpot_utils.py` should be used. This is done automatically by the `TestHandler` class.

[^4]: To save a little bit of computational effort in the first iteration, some of the original TPOT data is used, but as the population for subsequent iterations is potentially changed by each BO step, all other TPOT data must be generated in real-time.
