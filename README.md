# BO-TPOT Suite
This set of tools augments the [Tree-based Pipeline Optimisation Tool (TPOT)](http://epistasislab.github.io/tpot/) Python library for automated machine learning. It does so through Bayesian optimization (BO), in either a post hoc or in hoc fashion. By employing [Optuna](https://optuna.org/), a hyperparameter optimisation Python library, BO can be applied to an existing TPOT pipeline to improve its quality of prediction by identifying real-valued hyperparameters of the machine learning components of the pipeline.

## Citing this work
If you use any of the code in this repository for your research, please city the paper:

Kenny, A., Ray, T., Limmer, S., Singh, H.K., Rodemann, T., and Olhofer, M., , “Hybridizing TPOT with bayesian optimization,” in Proceedings of the Genetic and Evolutionary Computation Conference,(Lisbon Portugal), Conditionally accepted 04/2023

## Dependencies
[TPOT](http://epistasislab.github.io/tpot/), [PyTorch](http://pytorch.org), [Optuna](https://optuna.org/) - and various packages from [Anaconda Distribution](http://anaconda.org).

## Introduction
Regression problems are regularly encountered in practice where there is a need to identify a model that captures the relationship between independent variables/features and the dependent variable/response. The performance of the model is often characterised by the complexity of the model and the accuracy of prediction. There are many different types of machine learning models that can be used to deal with such problems and each of such machine learning models have their own unique set of hyperparameters. These hyperparameters can be real values, integers or even categories, and controls the predictive performance of the model. Machine learning models can be employed singularly, or combined; using the output of one model as the input to the next, harnessing the different strengths of multiple models simultaneously. When combined in this manner, the models are collectively known as a _pipeline_. Over the last decade there has been significant research effort dedicated towards automating the process of identifying promising machine learning pipelines.

There are three key decisions to consider when designing machine learning pipelines:
- which models to select; 
- how they are organised/structured, relative to each other; and, 
- what are the values of their respective hyperparameters. 

TPOT is a Python libarary, built on top of the [Distributed Evolutionary Algorithms in Python (DEAP)](https://github.com/deap) library, which is designed to automate design of machine learning pipelines. It represents pipelines as tree-based data structures and constructs them using the genetic programming (GP) methods provided by DEAP. 

One significant limitation of TPOT is that for all its GP bells-and-whistles, it is still employs a random grid-search to assign the hyperparameters to the constituent machine learning models. Since many hyperparameters are real-valued, TPOT descretises the continuous search spaces with a certain granularity. Although such a discretization is effective method to reduce the search space, the fact still remains that unless the global optimum value for a given hyperparameter lies on the exact point of descretisation, TPOT will never be able to find it.

[Optuna](https://optuna.org/) is a hyperparameter optimisation library for Python which uses Bayesian optimisation to tune hyperparameter values. It uses a Tree-based Parzen Estimator (TPE)-based surrogate model, built on historical information, to estimate and suggest hyperparameter values, which are then evaluated and the model subsequently updated. Unlike TPOT,  Optuna has no limitations on the type of values the hyperparameters can take. However it is not as effective at selecting models or structuring models in a pipeline as TPOT. 

We can think of TPOT as an effective tool for pipeline _exploration_ and BO as effective tool for pipeline _exploitation_. By using Optuna to _fine-tune_ the coarser results produced by TPOT, the algorithims in the BO-TPOT suite can potentially harness the strengths of both of these powerful tools. The question then becomes, "which pipelines should be selected for improvement by BO?", and exploring the many possibilities for this choice is the purpose of most of the tools in this suite.

---

## Tools in the BO-TPOT suite
There are four separate tools in the BO-TPOT suite:
- `TPOT-BASE`, which provides an interface to perform a base-line search using the un-altered TPOT algorithm, and store its output in a format compatible with the rest of the BO-POT suite;
- `TPOT-BO-S`, which selects a single candidate (based on smallest CV error) to improve with BO;
- `TPOT-BO-ALT`, which performs a prescribed number of alternating TPOT and BO optimisation steps;
- `TPOT-BO-AUTO`, which automatically decides whether to perform a TPOT or BO step, based on gradient information;

---
## File organisation
Each tool has the option to write tracking and output data to a file. There are different conventions and types of data tracked for each individual tool, however they all follow the same over-arching file organisation structure:

`<root>/<results dir>/<problem name>/<method name>/Seed_<seed number>/`

**NB:** previous versions followed a different file organisation convention. Any data generated using the deprecated convention can be easily updated by running the `file_organisation_patch.py` script from the root directory.

---

## Pipeline structures
During its operation, TPOT can be thought of as searching a hierarchy of two distinct classes of spaces. The first is the space of all possible combinations of operators, which it explores by using genetic programming to recombine and mutate tree representations of previously evaluated pipelines. The second is the sub-space of all possible hyper-parameter combinations for each unique combination of operators, which it explores using a grid-based search (having discretised any continuous parameter spaces). 

As it is used for tuning hyper-parameters, BO can only operate on this second sub-space and can only manipulate the value of hyper-parameters, not their relation to each other. Because of this, it is useful to have a method of grouping pipelines together by their pipeline structure. 

Let $a$, and $b$ be two pipelines. Pipelines $a$ and $b$ are said to be unique to each other if they disagree on at least one hyper-parameter value, or do not share the same configuration of hyper-parameters. A pipeline structure is a partition of the set of evaluated pipelines into subsets, such that every pair of pipelines within a subset are unique and share the same set of hyper-parameters (not necessarily values), in the same configuration. As a given operator will have the same hyper-parameters, regardless of its position in the pipeline, a pipeline structure can be represented by its operators alone.

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
The tools are found as classes in separate Python files in the directory `BO-TPOT` and may be run independently of each other. However, `TPOT-BO-S` relies on previously generated pipelines and requires that `TPOT-BASE` has been run _at some stage_.

Each class has a constructor and a single `optimize` method, into which the training data `X_train` and `y_train` must be passed. The `optimize` method may also be called with an optional keyword `out_path` which specifies a directory that the output files should be written to (if required). 

When the `optimize` method is called with `out_path` specified, pickling is used to save the progress of the search with each generation, to allow the search to be continued from near where it left off, should it be interrupted. The pickle file is located in `/<out_path>/<method_name>.pickle` - e.g.,`./Results/Prob1/TPOT-BASE/Seed_42/TPOT-BASE.pickle` - and is deleted at the end of a successfully completed search. It should be noted that resuming a search in this way makes exact reproduction of results almost impossible, as the random seed will be reset mid-way through the search. To start the search from scratch after an interruption, simple delete the `.pickle` file.

The tools can all be executed to operate on discrete or continuous parameter spaces. A lower case `d` or `c` suffix (discrete or continuous, respectively) at the end of a method name is used to indicate which type of parameter space was used - e.g., `TPOT-BO-Sd` for `TPOT-BO-S` using discrete parameters.

The processing of parameters and running and tracking of the tools in the aTPOT suite can be performed easily using the `TestHandler` class in the `utils` directory, but they can also be accessed separately for use in customised code.

The following is a detailed description of these three classes and their constructors and `optimize` methods.

---

### `TPOT-BASE`:
This method uses the `TPOTRegressor` class, to evolve a set of pipelines over $nG_t$ generations, for use as control data for comparison, $S$.

### **Parameters:**

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


### **Pseudocode:**

**Input:** $D$: training data; $nG_t$: number of generations; $nP$: population size; $\rho$: GP parameter set  
**Output:** $S$: evaluated pipeline set  
> $T$ &larr; `TPOTRegressor` class initialised using $nP$ and $\rho$  
> $S$ &larr; evaluated pipeline set after fitting $T$ on $D$ for $nG_t$ generations  
> **return** $S$

### **Outputs:**  
If the `out_path` parameter is set when calling the `optimize` method, 3 files are produced as output:

`TPOT-BASE.progress` - provides general information about the run

`TPOT-BASE.tracker` - provides tracking information for the selected parent population for each generation, with each semi-colon separated line taking the form:
>`<generation>;<pipeline structure>;<best evaluated CV error>`  

`TPOT-BASE.pipes` - provides full list of pipelines evaluated during the entire search, with each semi-colon separated line line taking the form:
>`<pipeline string>;<generation>;<evaluated CV error>`  

---

### `TPOT-BO-S`:

This post hoc method uses Bayesian optimisation (BO) techniques provided by the Optuna hyper-parameter optimisation library for Python to attempt to improve the pipeline with the best CV error, from an input set $S$. It is used as a stand-alone process to improve on a set of `TPOT-BASE` pipelines, or as a sub-process in other tools when they require BO to be applied to the best pipeline from a given set of pipelines.

Ordered by CV error, the best pipeline $s^* \in S$ is selected along with all pipelines that have a matching structure, producing $\bar{S}$. A BO model $M$ is constructed using $\bar{S}$, as well as an instance of the `TPOTRegressor` class $T$, with population size 1 and `generations = 0`, which is used to peform pipeline evaluations. With each evaluation in the allowed budget, a new pipline $s'$ is constructed using hyper-parameters suggested by $M$. This pipeline is evaluated by $T$ on the input data $T$ and both added to $\bar{S}$ and used to update the surrogate model $M$. Once the required number of BO evaluations has been performed, $S$ is updated with $\bar{S}$ and returned.

Occasionally, the hyper-parameters of the pipeline selected to be improved in the BO step do not have sufficient unique combinations of values. As TPOT does not re-evaluate already evaluated pipelines, this can result in `TPOT-BO-S` falling into an infinite loop. To avoid this, a counter $c$ is maintained which triggers a second stopping condition if no new pipelines are evaluated within 100 consecutive attempts. The counter is reset to 0 with every successful pipeline evaluation, and if it reaches 100, the `TPOT-BO-S` step is prematurely terminated, returning $S$ updated with $\bar{S}$.

`TPOT-BO-S` is also used as a sub-routine in `TPOT-BO-ALT` and `TPOT-BO-AUTO`. Here, the `optimize` method is called without the `out_path` parameter, so it does not generate any files.

### **Parameters:**

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

### **Pseudocode:**

**Input:** $D$: training data; $S$: input pipeline set; $nE$: number of BO evaluations; $\rho$: GP parameter set  
**Output:** $S$: evaluated pipeline set  
> $\bar{S}$ &larr; best pipeline $s^* \in S$, and all pipelines with matching structure  
> $M$ &larr; surrogate model constructed from $\bar{S}$  
> $T$ &larr; `TPOTRegressor` class with population size 1, generations 0 and GP parameters $\rho$  
> $c$ &larr; 0 &nbsp;&nbsp;&nbsp;&nbsp;counter for unsuccessful evaluations  
> $nE_t$ &larr; $nE + |\bar{S}|$  
> **while** $nE_t > 0$ **do:**  
>> $s'$ &larr; pipeline built from hyper-parameters suggested by $M$, evaluated by $T$ on $D$ and $\bar{S}$  
>> **if** $s'$ not evaluated **then:**  
>>> $c$ &larr; $c + 1$  
>>> **if** $c = 100$ **:** **break**  
>>
>> **else:**  
>>> $c$ &larr; 0  
>>> $\bar{S}$ &larr; $\bar{S} \cup s'$  
>>
>> $M$ &larr; update surrogate model $M$ with $s'$
>
> **return** $S \cup \bar{S}$

### **Outputs:**  
If the `out_path` parameter is set when calling the `optimize` method, 2 files are produced as output:

`TPOT-BO-S{d/c}.progress` - provides general information about the run

`TPOT-BO-S{d/c}.pipes` - provides full list of pipelines evaluated during the entire search, with each semi-colon separated line line taking the form:
>`<pipeline string>;<evaluated CV error>`  

---

### `TPOT-BO-ALT`:
`TPOT-BO-S` invests a large portion of its computational budget to improve a single candidate pipeline in its BO step. However, this is only effective when the pipeline selected has enough room for improvement in the first place. As there is no way of knowing this a priori, `TPOT-BO-S` essentially locks-in its choice of pipeline structure and hopes that further improvements materialize during the BO step. An alternative strategy to this is the in hoc method `TPOT-BO-ALT`. 

Starting from generation 0, `TPOT-BO-ALT` divides the total computing budget into $nI$ iterations, with each further divided into a TPOT and `TPOT-BO-S` step. The number of total TPOT generations $nG_t$ and the number of generations per TPOT step $nG_s$ are specified as part of the input and, typically, should be set to allow a fair comparison with other methods. For example, let the TPOT population size $nP$ be 100, and say 100 `TPOT-BASE` generations were performed as a control, with `TPOT-BO-S` performing 2000 evaluations from the 80 generation mark. If $nI = 10$, then appropriate values would be $nG_t= 100$ and $nG_s = \frac{80}{nI} = 8$. Using these values, the number of BO evaluations for the `TPOT-BO-S` step can be calculated as ${nE = (nG_t - nI \times nG_s) \times nP = 200}$. This ensures that the total number of evaluations for all three methods would be 10,000, with the same ratio of TPOT to BO evaluations for both `TPOT-BO-S` and `TPOT-BO-ALT`.

An instance of the `TPOTRegressor` class $T$ is initialised with population size $nP$, from which an initial population of evaluated solutions $S$ is extracted. Then, for each of $nI$ iterations, $S$ is updated with the result of fitting $T$ on the input data $D$ and $S$ for $nG_s$ generations and passed to an instance of `TPOT-BO-S` which applies BO for $nE$ evaluations. In the first iteration $T$ is fit for $nG_s - 1$ generations to account for the initial construction of the population, which must be evaluated using $T$ as well. Once all iterations have completed, $S$ is returned.

#### Parameters:

The table below gives all parameter arguments, their type and default values:

| Parameter              | Type     | Default                   | 
|:-----------------------|:--------:|--------------------------:|
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

### **Pseudocode:**

**Input:** $D$: training data; $nG_t$: total number of TPOT generations; $nP$: TPOT population size; $nI$: number of iterations; $nG_s$: number of generations per TPOT step; $\rho$: GP parameter set  
**Output:** $S$: evaluated pipeline set  
> $T$ &larr; `TPOTRegressor` class initialised with $nP$ and GP parameters $\rho$  
> $nE$ &larr; $(nG_t - nI \times nG_s) \times nP$  
> $S$ &larr; initial evaluated pipeline set from $T$  
> **for** $nI$ iterations **do:**  
>> $S$ &larr; update with result of fitting $T$ on $D$ and $S$ for $nG_s$ generations  
>> $S$ &larr; `TPOT-BO-S`$(D,S,nE,\rho)$  
>
> **return** $S$

### **Outputs:**  
If the `out_path` parameter is set when calling the `optimize` method, 2 files are produced as output:

`TPOT-BO-ALT{d/c}.progress` - provides general information about the run

`TPOT-BO-ALT{d/c}.pipes` - provides full list of pipelines evaluated during the entire search, with each semi-colon separated line line taking the form:
>`<pipeline string>;<iteration>;<generation>;<source>;<evaluated CV error>`  

---

### `TPOT-BO-AUTO`:
While dividing the budget into $nI$ iterations allows `TPOT-BO-ALT` to spread its BO budget across a number of pipelines, the choice of $nI$ is still arbitrary and there is a risk that the budget is not being used as efficently as possible. `TPOT-BO-AUTO` aims to address this by removing this parameter $nI$ and using gradient information to decide whether to use TPOT or `TPOT-BO-S` with each generation (or equivalent $nP$ BO evaluations).

A population of $nP$ pipelines $S$ is initialised with an instance of the `TPOTRegressor` class $T$. The initial gradients $\Delta_T$ and $\Delta_B$ (TPOT and `TPOT-BO-S` steps, respectively) are set to arbitrarily large values, with $\Delta_T > \Delta_B$ to force the TPOT step to execute first. With each generation, the gradients are checked to see which step has the best gradient recently and that step is executed for either 1 TPOT generation or $nP$ BO evaluations. If the gradients are equal, then toggle to the step which was _not_ most recently used. Once the step is completed, $S$ is updated, along with the appropriate gradient, and process is repeated until $nG_t -1$ (-1 to account for the generation of the initial population) generations have elapsed - returning $S$ at the end.

#### Parameters:

The table below gives all parameter arguments, their type and default values:

| Parameter              | Type     | Default                   | 
|:-----------------------|:--------:|--------------------------:|
|`pop_size`              | int      | 100                       |
|`n_gens`                | int      | 100                       |
|`seed`                  | int      | 42                        |
|`discrete_mode`         | boolean  | True                      |
|`optuna_timeout_trials` | int      | 100                       |
|`config_dict`           | dict     |`default_tpot_config_dict` |
|`n_jobs`                | int      | -1                        |
|`pipe_eval_timeout`     | int      | 5                         |
|`vprint`                | `Vprint` |`u.Vprint(1)`              |

### **Pseudocode:**

**Input:** $D$: training data; $nG_t$: total number of TPOT generations; $nP$: TPOT population size; $\rho$: GP parameter set  
**Output:** $S$: evaluated pipeline set  
> $T$ &larr; `TPOTRegressor` class initialised with $nP$ and GP parameters $\rho$  
> $S$ &larr; initial evaluated pipeline set from $T$  
> $\Delta_T,\Delta_B$ &larr; arbitrarily large values such that $\Delta_T > \Delta_B$  
> `DoTPOT` &larr; True  
> **for** $nG_t -1$ iterations **do:**  
>> **if** $\Delta_T = \Delta_B$ **:**  
>>> `DoTPOT` &larr; `!DoTPOT`  
>>
>> **else:**  
>>> `DoTPOT` &larr; $\Delta_T > \Delta_B$  
>>
>> **if** `DoTPOT` = True **:**  
>>> $S$ &larr; update with result of fitting $T$ on $D$ and $S$ for 1 generation  
>> 
>> **else:**  
>>> $S$ &larr; `TPOT-BO-S`$(D,S,nP,\rho)$  
>>
>> $\Delta_T,\Delta_B$ &larr; update gradients  
>
> **return** $S$

### **Outputs:**  
If the `out_path` parameter is set when calling the `optimize` method, 2 files are produced as output:

`TPOT-BO-AUTO{d/c}.progress` - provides general information about the run

`TPOT-BO-AUTO{d/c}.pipes` - provides full list of pipelines evaluated during the entire search, with each semi-colon separated line line taking the form:
>`<pipeline string>;<iteration>;<generation>;<source>;<evaluated CV error>`  
---

## Running BO-TPOT

As mentioned above, it is possible to run the `TPOT-BASE`, `TPOT-BO-S`, `TPOT-BO-ALT` and `TPOT-BO-AUTO` processes individually, by instantiating their classes and calling their respective `optimize` methods with the appropriate training data, but the simplest way is to execute the script in the file `run_BO-TPOT_tests.py`. At the start of this script is a parameters dictionary `params` which the user can populate and is given to a `TestHandler` object which sets up and runs all the required tests, in accordance with the specified parameters, automatically.

The table below gives the parameters in this dictionary and their type, followed by a description of their functions:

| Parameter              | Type       |
|:-----------------------|:----------:|
|`METHOD`                | string     |
|`VERBOSITY`             | int        |
|`DATA_DIR`              | string     |
|`RESULTS_DIR`           | string     |
|`SEEDS`                 | list       | 
|`PROBLEM`               | list       |
|`TPOT_CONFIG_DICT`      | dict       |
|`nJOBS`                 | int        |
|`DISCRETE_MODE`         | boolean    |
|`PIPE_EVAL_TIMEOUT`     | int        | 
|`POP_SIZE`              | int        | 
|`nTOTAL_GENS`           | int        | 
|`STOP_GEN`              | int        | 
|`OPTUNA_TIMEOUT_TRIALS` | int        |
|`nALT_ITERS`            | int        |

### `METHOD`:
String to control which process is executed. It takes the name of the method requred (e.g., `TPOT-BASE`, `TPOT-BO-S`, `TPOT-BO-ALT` or `TPOT-BO-AUTO`) as a string and calls the relevant method from `TestHandler`. Keep in mind that `TPOT-BO-S` requires that `TPOT-BASE` has been executed _at some stage_.

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

### `SEEDS`:
This parameter allows the user to specify the starting random seeds for the runs they want to execute. It is specified as `list` of integers, with each element representing a unique run, the results of which are stored according to seed number. If running `TPOT-BO-S`, then be sure to provide seed numbers for which there is already `TPOT-BASE` data, or else it will skip the current seed.

### `PROBLEM`:
This is a string which specifies the problem to be solved. A problem is specified by the file name (without extension) for its data file, within the `data_dir` directory. For example, if the data for the required problem is held in `./Data/prob1.data`, then `data_dir` would be `'Data'` and `PROBLEM` would be `'prob1'`.

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

Any of the classes in the BO-TPOT suite can be instantiated as part of a custom solver, written by the user. However, there is also a `TestHandler` class provided which implements methods which handle the general "housekeeping" aspects of running the tests, such as data storage and structuring, progress information and parameter handling. It also implements any error handling while running batch tests, allowing the batch to continue running if there should be any errors. This class is instantiated using a dictionary of parameters as is given in the `run_tpot_tests.py` description above. Once instantitated, the `set_problem` and `set_run` methods can be used to load problem data and generate directories for output files to be saved. 


### Tracking runs:
Along with simplifying the process of running tests with the three aspects of BO-TPOT, the `TestHandler` class also automatically tracks the progress of the runs.

Each time this class is instantiated, a file in `./<RESULTS_DIR>/` called `BO_TPOT.progress` is created with the date and time the execution was started, the values of the parameters in the `params` dictionary (if the default TPOT configuration dictionary is used, then it will just say `default_tpot_config_dict`, otherwise it will give the full dictionary). As well as this general execution information, for each problem and run, the seed is given, and the outcome of any processes run, along with the time taken (if successful). If any process fails or crashes, then the error message and stack traceback will be written to this file, but the execution itself will contine with the next process or run. This helps to diagnose any minor or incidental bugs, without having to cancel the entire set of runs.


## Processing Results
The file `post_process.py` provides a script that automatically processes the results produced by BO-TPOT and computes the pairwise statistics for those methods being compared.

The problems required are given as a `list` in the `PROBLEMS` parameter, and methods as a list in the `METHODS` parameter. Seeds can be specified as a `list` of integers in `SEED_LIST`, or passed as an empty `list` to process any seeds for which there is valid information available for all methods required.

The `STOP_GEN` parameter can be used if an additional point in the `TPOT-BASE` should be compared (e.g., after 80 generations to measure any improvements gained by `TPOT-BO-S`), or can be set to `None`, if not required.

If pairwise statistics are required, the `CONFIDENCE_LEVEL` can be set between 0 and 1 to determine the confidence level for win/tie/loss (e.g., 0.05 for a confidence level of 5%).
