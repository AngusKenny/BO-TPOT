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
import numpy as np

class TPOT_BO_EX(object):    
    def __init__(self,  
                 init_pipes,
                 seed=42,
                 pop_size=100,
                 bo_pop_factor=0.5,
                 n_bo_evals=2000,
                 discrete_mode=True,
                 restricted_hps=False,
                 optuna_timeout_trials=100,
                 config_dict=default_tpot_config_dict,
                 pipe_eval_timeout=5,
                 n_structures=20,
                 vprint=u.Vprint(1)):
        
        self.pipes={}
        self.n_structures = n_structures
        self.pop_size = pop_size
        self.n_bo_evals=n_bo_evals
        self.tpot_pipes=copy.deepcopy(init_pipes)
        self.config_dict=copy.deepcopy(config_dict)
        self.restricted_hps=restricted_hps
        self.discrete_mode=discrete_mode
        self.optuna_timeout_trials=optuna_timeout_trials
        self.seed=seed
        self.pipe_eval_timeout=pipe_eval_timeout
        self.vprint=vprint
        self.d_flag = 'd' if discrete_mode else 'c'
        
        # set tpot verbosity to vprint.verbosity + 1 to give more information
        self.tpot_verb = vprint.verbosity + 1 if vprint.verbosity > 0 else 0
        
        for k,v in self.tpot_pipes.items():
            v['source'] = f'TPOT-BASE'
        
        # get unique structures
        self.strucs = u.get_structures(self.tpot_pipes, config_dict=self.config_dict)
        # u_grps = u.get_unique_groups(copy.deepcopy(self.tpot_pipes), config_dict=self.config_dict)
                        
        self.bo_struc_keys = u.get_best_structures(self.strucs, size=n_structures)
        
        vprint.v2(f"\n{u.CYAN}{len(self.bo_struc_keys)} groups in BO set, populating TPOT evaluated dictionary..{u.OFF}\n")
        
        # create TPOT object for each pipe in set and fit for 0 generations
        for i,k in enumerate(self.bo_struc_keys):
            self.pipes.update(self.strucs[k].pipes)

        self.starting_size = len(self.pipes)
        
    def optimize(self, X_train, y_train, out_path=None):
        start_idx = 0
        
        if out_path:
            if not os.path.exists(out_path):
                os.makedirs(out_path)   
            
            fname_improvements = os.path.join(out_path,"TPOT-BO-EX.improvements")
            
            if os.path.exists(fname_improvements):
                with open(fname_improvements, 'r') as f:
                    start_idx = len(f.readlines())
                
        t_start = time.time()
        
        self.vprint.v1("")
        gen = 1
        
        early_finish = ""

        stagnate_cnt = 0
        
        for i,k in enumerate(self.bo_struc_keys[start_idx:self.n_structures]):
            n_hp = len(self.strucs[k].bo_params)
            struc_data = self.strucs[k]                
            
            seed_samples = [(u.string_to_params(k2), v2['internal_cv_score']) for k2,v2 in struc_data.pipes.items()]
            
            self.vprint.v2(f"\n{u.CYAN}Generation {gen}, structure group {i+1} - {len(seed_samples)} seed samples generated, optimizing for {self.n_bo_evals} evaluations..{u.OFF}\n")
            
            tpot = TPOTRegressor(generations=0,
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
            tpot._fit_init()
            
            tpot.evaluated_individuals_ = struc_data.pipes
            
            old_cv = max([v['internal_cv_score'] for v in tpot.evaluated_individuals_.values()])
            print(f"{u.RED}Old CV: {old_cv}{u.OFF}")
            
            # initialise tpot bo handler
            handler = TPOT_BO_Handler(tpot, vprint=self.vprint, discrete_mode=self.discrete_mode)
            new_params = u.string_to_params(struc_data.best)
            
            # update pset of BO tpot object
            for (p,val) in new_params:
                handler.add_param_to_pset(p, val)
            
            # remove generated pipeline and transplant saved from before
            tpot._pop = [creator.Individual.from_string(struc_data.best, tpot._pset)]
            
            tpot.fit(X_train, y_train)
            
            # re-initialise tpot bo handler
            handler = TPOT_BO_Handler(tpot, vprint=self.vprint, discrete_mode=self.discrete_mode)
            
            # run bayesian optimisation with seed_dicts as initial samples
            handler.optimise(0, X_train, y_train, n_evals=self.n_bo_evals,
                                seed_samples=seed_samples, 
                                discrete_mode=self.discrete_mode,
                                skip_params=[],
                                timeout_trials=self.optuna_timeout_trials)
            
            self.strucs.update(copy.deepcopy(tpot.evaluated_individuals_))
            
            new_cv = max([v['internal_cv_score'] for v in tpot.evaluated_individuals_.values()])
            print(f"{u.RED}New CV: {new_cv}{u.OFF}")
            # new_cv = self.strucs[k].cv
            
            if out_path:
                with open(fname_improvements,'a') as f:
                    f.write(f"{k};{old_cv};{new_cv}\n")
        
        t_end = time.time()

        self.vprint.v1(f"\nTotal time elapsed: {round(t_end-t_start,2)} sec\n")
                    
        return f"Successful{early_finish}"
                    