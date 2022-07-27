# -*- coding: utf-8 -*-
''' Classes for optuna hyperparameter optimisation
'''

import warnings
import utils as u
import optuna
from optuna._experimental import experimental
from deap.gp import Primitive, Terminal
from optuna_hp_spaces import (make_hp_space_real, 
                              make_hp_space_discrete, 
                              make_optuna_trial_real,
                              make_optuna_trial_discrete)
import numpy as np

class RequiredTrialsCallback(object):
    def __init__(self, n_evals, eval_dict, timeout_trials,vprint=u.Vprint(1)):
        self.n_evals = n_evals
        self.eval_dict = eval_dict
        self.timeout_trials = timeout_trials
        self.timeout_count=0
        self.old_size = len(eval_dict)
        self.vprint = vprint

    def __call__(self, study: optuna.study.Study, 
                 trial: optuna.trial.FrozenTrial) -> None:
        if len(self.eval_dict) >= self.n_evals:
            self.vprint.v2(f"{self.n_evals} unique solutions evaluated in total, "
                      + "stopping optimisation process..\n")
            study.stop()
        
        # check if new solution has been added
        if self.old_size == len(self.eval_dict):
            self.timeout_count = self.timeout_count + 1
            if self.timeout_count >= self.timeout_trials:
                self.vprint.v2(f"{self.timeout_trials} trials without finding a "
                          + "unique solution to evaluate, stopping "
                          + "optimisation process..\n")
                study.stop()
        else:
            self.timeout_count = 0
        
        # update old size
        self.old_size = len(self.eval_dict)
        
class Objective(object):
    ''' Class for optuna optimisation objective function
    '''
    def __init__(self, po, ind_idx, X, y, n_evals, real_vals=True, 
                 seed_samples=[], vprint=u.Vprint(1)):
        ''' Constructor for Objective class

        Parameters
        ----------
        po : PipelinePopOpt
            PipelinePopOpt object to handle interface with TPOT.
        ind_idx : INT
            Index of current pipeline to be optimised.
        X : Dataset
            X training data.
        y : Dataset
            y training data.
        real_vals : Bool, optional
            Optimise using real values (or discrete). The default is True.
        seed_samples : LIST, optional
            List of dictionaries containing initial seed samples. 
            The default is empty.

        Returns
        -------
        None.

        '''
        self.vprint = vprint
        self.po = po
        self.real_vals = real_vals
        self.ind_idx = ind_idx
        self.X = X
        self.y = y
        self.seed_samples = seed_samples
    
    def __call__(self, trial):
        ''' Evaluation call for optuna optimisation

        Parameters
        ----------
        trial : Trial
            Optuna Trial object for optimisation.

        Returns
        -------
        score : FLOAT
            CV score as evaluated by TPOT (TPOT gives negative by default so
            sign is reversed before sending back to optuna.
        '''
        
        trial_params = []

        # set up search space
        param_names = [param[0] for param in self.po.best_params[self.ind_idx]]
        if self.real_vals:
            trial_params = make_hp_space_real(trial, param_names)
        else:
            trial_params = make_hp_space_discrete(trial, param_names)  
        
        self.vprint.v1(f"{u.YELLOW_U}optuna call "
                       + f"{len(self.po.eval_scores[self.ind_idx])+1} "
                       + f"(arch size {len(self.po.tpot.evaluated_individuals_)})"
                       + f":{u.OFF}")
        
        self.vprint.v2(f"params: {trial_params}")
        
        # set new params
        self.po.set_params(self.ind_idx, trial_params)
                
        # evaluate with new params
        score = self.po.evaluate(self.ind_idx, self.X, self.y)
        
        self.po.eval_scores[self.ind_idx].append(score)
        
        # update best
        if score > self.po.best_scores[self.ind_idx]:
            self.po.best_scores[self.ind_idx] = score
            self.po.best_params[self.ind_idx] = trial_params
        
        # update overall best
        if score > self.po.best_score:
            self.po.best_score = score
            self.po.best_pipe = str(self.po.tpot._pop[self.ind_idx])
        
        self.vprint.v1(f"{u.GREEN}* score: {u.OFF}{score}\n\t(best: {self.po.best_scores[self.ind_idx]})\n")
        
        return score


class PipelinePopOpt(object):
    ''' Class to interface with TPOT object and optuna library
    '''
    def __init__(self, tpot, vprint=u.Vprint(1), real_vals=True):
        ''' Constructor for PipelinePopOpt class.
        '''
        self.vprint = vprint
        self.real_vals = real_vals
        self.tpot = tpot
        self.size = len(tpot._pop)
        self.best_score = -1e20
        self.best_pipe = None
        self.best_params = []
        self.init_params = []
        self.eval_scores =[[] for i in range(self.size)]
        self.best_scores = [-1e20 for i in range(self.size)]
        # initialise parameter lists
        for i in range(self.size):
            params = self.get_params(i)
            self.best_params.append(params)
            self.init_params.append(params)
                    
    def evaluate(self, ind_idx, X, y):
        ''' Use TPOT to evaluate a given individual
        '''
                
        # clear fitness values
        del(self.tpot._pop[ind_idx].fitness.values)
        
        # evaluate full population
        # (will only evaluate those with invalid fitness values)
        self.tpot._evaluate_individuals(self.tpot._pop, X, y)
        
        # return evaluated fitness value
        return self.tpot._pop[ind_idx].fitness.values[1]
    
    def add_param_to_pset(self, p_name, p_val):
        '''
        if new value then add to operator value list and update pset
        the operator values list gets reset with every call to fit()
        so we must add it to the operator values list, even if the
        value already exists in the pset from previous evaluations
        '''
        if u.is_number(p_val):
            for op in self.tpot.operators:
                if op.__name__ in p_name:
                    # find parameter type in parameter list (skip index 0)
                    for p_type in op.parameter_types()[0][1:]:
                        if p_type.__name__ == p_name:
                            # if value missing from p_type values list
                            if p_val not in p_type.values:
                                # update values list
                                if type(p_type.values) == np.ndarray:
                                    np.append(p_type.values, p_val)
                                elif type(p_type.values) == list:
                                    p_type.values.append(p_val)
                                else:
                                    self.vprint.v2("Cannot update value list for " 
                                          + p_type.__name__)
                                # update pset new value not already used
                                if (p_type.__name__ + "=" + str(p_val) 
                                    not in self.tpot._pset.context):
                                    self.vprint.v2("adding " + p_type.__name__+ "="
                                          + str(p_val) + " to pset")
                                    self.tpot._pset.addTerminal(
                                        p_val, p_type, 
                                        name=(p_type.__name__ 
                                              + "=" + str(p_val)))    
            
        
    def set_params(self, ind_idx, params):
        ''' Set the parameters of a given individual
        '''
        param_idx = 0
        
        # iterate over deap tree
        for n in range(len(self.tpot._pop[ind_idx])):
            # if primitive or 'input_matrix' then skip
            if (type(self.tpot._pop[ind_idx][n]) == Primitive 
                or self.tpot._pop[ind_idx][n].name == "ARG0"):
                continue
            
            p_name = params[param_idx][0]
            p_val = params[param_idx][1]
            
            # if skipped because only one possible value
            if p_val == 'skip':
                # move to next parameter
                param_idx = param_idx + 1
                continue
            
            # build new Terminal with parameter and replace
            temp = Terminal(
                p_name + "=" + str(p_val), 
                True, 
                self.tpot._pop[ind_idx][n].ret)
            
            self.tpot._pop[ind_idx][n] = temp
            
            self.add_param_to_pset(p_name, p_val)
            
            # move to next parameter
            param_idx = param_idx + 1
    
    def get_params(self, ind_idx):
        ''' Get the parameters of a given individual as parameter list
        '''
        params = []
        
        # iterate over population
        for node in self.tpot._pop[ind_idx]:
            # if primitive or 'input_matrix' then skip
            if type(node) == Primitive or node.name == "ARG0":
                continue
            
            # split string on "=" and append parameter
            split_str = node.name.split("=")
            if not u.is_number(split_str[1]):
                params.append((split_str[0], split_str[1]))
            elif "." in split_str[1] or 'e' in split_str[1]:
                params.append((split_str[0], float(split_str[1])))
            else:
                params.append((split_str[0], int(split_str[1])))
            
        return params
    
    def get_best_evaluated_individual(self):
        ''' Get the string representing the best individual in the current 
            TPOT population, along with its CV value
        '''
        best_cv = -1e20
        best_pipe_str = []
        # find best pipeline
        for k,v in self.tpot.evaluated_individuals_.items():
            if v['internal_cv_score'] > best_cv:
                best_cv = v['internal_cv_score']
                best_pipe_str = k
        
        return best_pipe_str, best_cv
        
    def get_best_pipe_idx(self):
        ''' Get the index of the best individual in the current TPOT 
            population, along with its CV value
        '''
        best_cv = -1e20
        best_pipe_idx = 0
        for i in range(len(self.tpot._pop)):
            if self.tpot._pop[i].fitness.values[1] > best_cv:
                best_cv = self.tpot._pop[i].fitness.values[1]
                best_pipe_idx = i
    
        return best_pipe_idx, best_cv
    
    def get_worst_pipe_idx(self):
        ''' Get the index of the worst individual in the current TPOT 
            population, along with its CV value
        '''
        worst_cv = 1e20
        worst_pipe_idx = 0
        for i in range(len(self.tpot._pop)):
            if self.tpot._pop[i].fitness.values[1] < worst_cv:
                worst_cv = self.tpot._pop[i].fitness.values[1]
                worst_pipe_idx = i
    
        return worst_pipe_idx, worst_cv
    
    def get_matching_structures(self, tgt, stop_gen=None):
        ''' Using tgt as the target string, get all strings representing
            pipelines that have been evaluated by TPOT so far, with matching
            structures
            stop_gen indicates maximum generation to get pipes from
        '''
        matching_strs = []
        # when split on "(", all but last strings are primitives
        tgt_split = tgt.split("(")
        for k,v in self.tpot.evaluated_individuals_.items():
            if k == tgt:
                continue
            k_split = k.split("(")
            # check if same number of operators
            if len(k_split) != len(tgt_split):
                continue
            # check if operators match
            if tgt_split[0:-1] != k_split[0:-1]:
                continue
            if not stop_gen or v['generation'] < stop_gen:
                matching_strs.append(k)
                        
        return matching_strs
    
    def optimise(self, ind_idx, X_train, y_train, n_evals, real_vals=True, seed_samples=[], timeout_trials=1e20):
        ''' Run optuna optimisation        

        Parameters
        ----------
        ind_idx : INT
            Index of individual to optimise.
        X_train : Dataset
            X training data.
        y_train : Dataset
            y training data.
        n_evals : INT
            Number of evaluations to perform.
        seed_samples : LIST, optional
            List of dictionaries containing initial seed samples. 
            The default is empty.
        timeout_trials: INT
            Number of trials allowed without adding to pipeline dictionary
            before stopping early (needed for discrete parameter spaces
                                   especially - but not only)

        Returns
        -------
        study : Study object
            Optuna Study object with trial data.

        '''
        self.vprint.v2("optimising..")
        optuna_verb = optuna.logging.DEBUG if self.vprint.verbosity > 2 else optuna.logging.WARN

        optuna.logging.set_verbosity(verbosity=optuna_verb)
        
        # create NGSAII sampler
        # sampler = optuna.samplers.NSGAIISampler(population_size=100)     
        sampler = optuna.samplers.TPESampler(multivariate=True, warn_independent_sampling=False)     
        
        # create optuna study - we have to maximise because TPOT 
        # returns negative CV value
        study = optuna.create_study(sampler=sampler, direction="maximize")

        # create trial based on seed data and insert to model without evaluation
        for seed_sample in seed_samples:
            if real_vals:
                trial = make_optuna_trial_real(seed_sample[0], seed_sample[1])
            else:
                trial = make_optuna_trial_discrete(seed_sample[0], seed_sample[1])
            study.add_trial(trial)
            
            self.best_scores[ind_idx] = max(self.best_scores[ind_idx],seed_sample[1])
            self.best_score = max(self.best_score,seed_sample[1])
        
        
        
        # account for initial seed trials
        n_evals_adj = n_evals + len(self.tpot.evaluated_individuals_)
        
        # set up callback object to stop when required number of solutions evalutated
        stop_callback = RequiredTrialsCallback(n_evals_adj, self.tpot.evaluated_individuals_, timeout_trials, vprint=self.vprint)
        
        # run optimise method
        study.optimize(Objective(self, ind_idx, X_train, y_train,n_evals_adj,real_vals=real_vals,vprint=self.vprint),callbacks=[stop_callback])