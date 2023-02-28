#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 12:22:43 2022

@author: gus
"""
from config.tpot_config import default_tpot_config_dict
from tpot import TPOTRegressor
from deap import creator
from BO_TPOT.tpot_bo_tools import TPOT_BO_Handler
import utils.tpot_utils as u
import copy
import os
import time

class TPOT_BO_S(object):
    
    def __init__(self,  
                 init_pipes,
                 seed=42,
                 n_bo_evals=2000,
                 discrete_mode=True,
                 restricted_hps=False,
                 optuna_timeout_trials=100,
                 config_dict=default_tpot_config_dict,
                 pipe_eval_timeout=5,
                 source_method=None,
                 vprint=u.Vprint(1)):
        
        self.pipes = {}
        self.n_bo_evals=n_bo_evals
        self.tpot_pipes=copy.deepcopy(init_pipes)
        self.config_dict=copy.deepcopy(config_dict)
        self.discrete_mode=discrete_mode
        self.optuna_timeout_trials=optuna_timeout_trials
        self.seed=seed
        self.pipe_eval_timeout=pipe_eval_timeout
        self.vprint=vprint
        self.type_flag = 'd' if discrete_mode else 'c'
        self.discrete_mode = discrete_mode
        
        self.source_method = f"TPOT-BO-S{self.type_flag}" if source_method == None else f"{source_method}"
        
        # set tpot verbosity to vprint.verbosity + 1 to give more information
        self.tpot_verb = vprint.verbosity + 1 if vprint.verbosity > 0 else 0
        
        # create TPOT object and fit for 0 generations
        self.tpot = TPOTRegressor(generations=0,
                                  population_size=1, 
                                  mutation_rate=0.9, 
                                  crossover_rate=0.1, 
                                  cv=5,
                                  verbosity=self.tpot_verb, 
                                  config_dict=copy.deepcopy(self.config_dict),
                                  random_state=self.seed, 
                                  n_jobs=1,
                                  warm_start=True,
                                  max_eval_time_mins=self.pipe_eval_timeout)
        
        # initialise tpot object to generate pset
        self.tpot._fit_init()
        
        vprint.v2(f"\n{u.CYAN}Transplanting best pipe from previous TPOT set and finding matching pipes..{u.OFF}\n")
        
        # get best from previous pop
        self.best_init_pipe,self.best_init_cv = u.get_best(self.tpot_pipes)
 
        self.pipes = u.get_matching_set(self.best_init_pipe, self.tpot_pipes)   
        
        best_params = u.string_to_params(self.best_init_pipe)
        
        for (p,v) in best_params:
            u.add_param_to_pset(self.tpot,p, v)
        
        # remove generated pipeline and transplant saved from before
        self.tpot._pop = [creator.Individual.from_string(self.best_init_pipe, self.tpot._pset)]        
        
        # initialise tpot object to generate pset
        self.tpot._fit_init()
        
        # replace evaluated individuals dict
        self.tpot.evaluated_individuals_ = copy.deepcopy(self.pipes)
        
        # initialise tpot bo handler
        self.handler = TPOT_BO_Handler(self.tpot, vprint=self.vprint, discrete_mode=self.discrete_mode)

        
    def optimize(self, X_train, y_train, out_path=None):
        t_start = time.time()
        
        self.vprint.v2(f"{u.CYAN}\nfitting tpot model with 0" 
                + f" generations to initialise..\n{u.OFF}")
        
        self.tpot.fit(X_train, y_train)
                
        self.vprint.v1("")
        
        seed_samples = [(u.string_to_params(k), v['internal_cv_score']) for k,v in self.pipes.items()]
        
        self.vprint.v2(f"\n{u.CYAN}{len(seed_samples)} seed samples generated, optimizing for {self.n_bo_evals} evaluations..{u.OFF}\n")
          
        # run bayesian optimisation with seed_dicts as initial samples
        self.handler.optimise(0, X_train, y_train, n_evals=self.n_bo_evals,
                    seed_samples=seed_samples, discrete_mode=self.discrete_mode,
                    timeout_trials=self.optuna_timeout_trials)
        
        for k,v in self.tpot.evaluated_individuals_.items():
            if k not in self.pipes:
                self.pipes[k] = v
                self.pipes[k]['source'] = self.source_method
        
        t_end = time.time()
        
        best_bo_pipe, best_bo_cv = u.get_best(self.pipes, source=self.source_method)
        
        self.vprint.v1(f"\n{u.YELLOW}best pipe found by BO:{u.OFF}")
        self.vprint.v1(f"{best_bo_pipe}\n{u.GREEN} * score:{u.OFF} {best_bo_cv}")
        self.vprint.v1(f"\nTotal time elapsed: {round(t_end-t_start,2)} sec\n")
        
        # if out_path exists then write pipes to file
        if out_path:
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            fname_bo_pipes = os.path.join(out_path,f'TPOT-BO-S{self.type_flag}.pipes')
            # write all evaluated pipes
            with open(fname_bo_pipes, 'w') as f:
                for k,v in self.pipes.items():
                    if v['source'] == f'TPOT-BO-S{self.type_flag}':
                        f.write(f"{k};{v['internal_cv_score']}\n")
                    
        return "Successful"
                    
