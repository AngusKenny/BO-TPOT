#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 18:14:11 2022

@author: gus
"""

from config.tpot_config import default_tpot_config_dict
from tpot import TPOTRegressor
import utils.tpot_utils as u
import copy
import os
import time

class TPOT_Base(object):
    pipes = None
    
    def __init__(self,
                 n_gens=100,
                 pop_size=100,
                 seed=42,
                 config_dict=default_tpot_config_dict,
                 n_jobs=-1,
                 pipe_eval_timeout=5,
                 vprint=u.Vprint(1)):
        
        self.n_gens=n_gens
        self.pop_size=pop_size
        self.vprint=vprint
        self.seed=seed
        self.config_dict = copy.deepcopy(config_dict)
        self.n_jobs = n_jobs
                
        # set tpot verbosity to vprint.verbosity + 1 to give more information
        tpot_verb = vprint.verbosity + 1 if vprint.verbosity > 0 else 0
        
        # create TPOT object and fit for tot_gens generations
        self.tpot = TPOTRegressor(generations=self.n_gens-1,
                                  population_size=self.pop_size, 
                                  mutation_rate=0.9, 
                                  crossover_rate=0.1, 
                                  cv=5,
                                  verbosity=tpot_verb, 
                                  config_dict = self.config_dict, 
                                  random_state=self.seed, 
                                  n_jobs=self.n_jobs,
                                  warm_start=True,
                                  max_eval_time_mins=pipe_eval_timeout)
        
    def optimize(self, X_train, y_train, out_path=None):
        t_start = time.time()

        self.vprint.v2(f"{u.CYAN}fitting tpot model with {self.tpot.generations}" 
               + " generations (-1 to account for initial evaluations)" 
               + f"..\n{u.WHITE}")
        
        # in case something has been done to pipes externally update tpot pipes
        self.tpot.evaluated_individuals_ = copy.deepcopy(self.pipes)
        
        # fit tpot model to training data
        self.tpot.fit(X_train, y_train)
        
        # copy evaluated individuals dictionary
        self.pipes = copy.deepcopy(self.tpot.evaluated_individuals_)
        
        for k,v in self.pipes.items():
            v['source'] = 'TPOT-BASE'
        
        t_end = time.time()
        
        best_tpot_pipe, best_tpot_cv = u.get_best(self.pipes)
        
        self.vprint.v1(f"\n{u.YELLOW}* best pipe found by TPOT:{u.OFF}")
        self.vprint.v1(f"{best_tpot_pipe}")
        self.vprint.v1(f"{u.GREEN} * score:{u.OFF} {best_tpot_cv}")
        self.vprint.v1(f"\nTotal time elapsed: {round(t_end-t_start,2)} sec\n")
        
        
        # if out_path exists then write pipes to file
        if out_path:
            print(out_path)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            fname_tpot_pipes = os.path.join(out_path,'TPOT-BASE.pipes')
            print(fname_tpot_pipes)
            # write all evaluated pipes
            with open(fname_tpot_pipes, 'w') as f:
                for k,v in self.pipes.items():
                    f.write(f"{k};{v['generation']};{v['internal_cv_score']}\n")
                    
        return "Successful"