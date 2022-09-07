#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 12:22:43 2022

@author: gus

- take best cv (min)
- number of HPs (min)
- ND on those

- allocate budget proportional to number of HPs

- recompute ND



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

class TPOT_BO_ND(object):
    pipes = {}
    
    def __init__(self,  
                 init_pipes,
                 seed=42,
                 pop_size=100,
                 n_bo_evals=2000,
                 discrete_mode=True,
                 restricted_hps=False,
                 optuna_timeout_trials=100,
                 config_dict=default_tpot_config_dict,
                 pipe_eval_timeout=5,
                 vprint=u.Vprint(1)):
        
        self.pipes={}
        
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
        
        # set tpot verbosity to vprint.verbosity + 1 to give more information
        self.tpot_verb = vprint.verbosity + 1 if vprint.verbosity > 0 else 0
        
        for k,v in self.tpot_pipes.items():
            v['source'] = f'TPOT-BASE({str(u.string_to_structure(k))})'
        
        # get unique structures
        u_grps = u.get_unique_groups(copy.deepcopy(self.tpot_pipes), config_dict=self.config_dict)
        
        self.nd_grps = u.get_nd_best(u_grps, ("-cv_best","n_bo_params"))
        
        # truncate to pop_size if more than pop_size
        if len(self.nd_grps) > pop_size:
            self.nd_grps = {k:self.nd_grps[k] for k in list(self.nd_grps.keys())[:pop_size]}
        
        vprint.v2(f"\n{u.CYAN}{len(self.nd_grps)} groups in Pareto set, transplanting best pipe from previous for each..{u.OFF}\n")
        
        # create TPOT object for each pareto pipe and fit for 0 generations
        self.tpots = {}
        self.handlers = {}
        for k,v in self.nd_grps.items():
            # add to pipes
            self.pipes.update(copy.deepcopy(v['matching']))
            
            self.tpots[str(v['structure'])] = TPOTRegressor(generations=0,
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
            self.tpots[str(v['structure'])]._fit_init()
            
            # remove generated pipeline and transplant saved from before
            self.tpots[str(v['structure'])]._pop = [creator.Individual.from_string(v['best_pipe'], self.tpots[str(v['structure'])]._pset)]       
            
            # replace evaluated individuals dict
            self.tpots[str(v['structure'])].evaluated_individuals_ = copy.deepcopy(v['matching'])

        self.starting_size = len(self.pipes)
        
    def optimize(self, X_train, y_train, out_path=None):
        if out_path:
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            fname_h_pipes = os.path.join(out_path,"TPOT-BO-ND.pipes")
            # wipe pipe existing pipe files if they exist
            with open(fname_h_pipes,'w') as f:
                for k,v in self.pipes.items():
                    f.write(f"{k};0;{len(self.nd_grps)};{v['source']};{v['internal_cv_score']}\n")    
            
        t_start = time.time()
        
        self.vprint.v2(f"{u.CYAN}\nfitting tpot models with 0" 
                + f" generations to initialise..\n{u.OFF}")
        
        for k,model in self.tpots.items():
            model.fit(X_train, y_train)
        
        self.vprint.v1("")
        gen = 1
        
        while len(self.pipes) < (self.starting_size + self.n_bo_evals):
            tot_params = np.sum([v['n_bo_params'] for k,v in self.nd_grps.items()])
            
            # compute how many trials
            self.vprint.v2(f"\n{u.CYAN}TPOT-BO-ND generation {gen}, {len(self.nd_grps)} groups in the Pareto set..{u.OFF}\n")
            for i,(k,v) in enumerate(self.nd_grps.items()):
                
                n_grp_trials = int((v['n_bo_params']/tot_params)*self.pop_size)
                
                seed_samples = [(u.string_to_params(k2), v2['internal_cv_score']) for k2,v2 in v['matching'].items()]
                
                self.vprint.v2(f"\n{u.CYAN}Generation {gen}, group {i+1} of {len(self.nd_grps)} - {len(seed_samples)} seed samples generated, optimizing for a further {n_grp_trials} evaluations..{u.OFF}\n")
                
                # initialise tpot bo handler
                handler = TPOT_BO_Handler(self.tpots[str(v['structure'])], vprint=self.vprint, discrete_mode=self.discrete_mode)
                
                # run bayesian optimisation with seed_dicts as initial samples
                handler.optimise(0, X_train, y_train, n_evals=n_grp_trials,
                                 seed_samples=seed_samples, 
                                 discrete_mode=self.discrete_mode,
                                 skip_params=[],
                                 timeout_trials=self.optuna_timeout_trials)
                
                if out_path:
                    f = open(fname_h_pipes,'a')
                
                # update matching and recorded
                for k2,v2 in self.tpots[str(v['structure'])].evaluated_individuals_.items():
                    v2['source'] = f'TPOT-BO-ND({k})'
                    if k2 not in v['matching']:
                        v['matching'][k2] = copy.deepcopy(v2)
                    
                    if k2 not in self.pipes:
                        self.pipes[k2] = copy.deepcopy(v2)
                        if out_path:
                            f.write(f"{k2};{gen};{len(self.nd_grps)};{v2['source']};{v2['internal_cv_score']}\n")
                            
                if out_path:
                    f.close()
                
                # update group statistics
                self.nd_grps[k] = u.update_group(v)
                
            # update pareto set
            if len(self.nd_grps) > 1:
                self.nd_grps = u.get_nd_best(self.nd_grps, ("-cv_best","n_bo_params"))
            
            gen = gen+1
                
        
        t_end = time.time()
        
        best_tpot_pipe, best_tpot_cv = u.get_best(self.pipes, source='TPOT-BASE')
        best_bo_pipe, best_bo_cv = u.get_best(self.pipes, source='TPOT-BO-ND')
        
        self.vprint.v1(f"\n{u.YELLOW}* best pipe found by tpot:{u.OFF}")
        self.vprint.v1(f"{best_tpot_pipe}")
        self.vprint.v1(f"{u.GREEN} * score:{u.OFF} {best_tpot_cv}")
        self.vprint.v1(f"\n{u.YELLOW}best pipe found by BO:{u.OFF}")
        self.vprint.v1(f"{best_bo_pipe}\n{u.GREEN} * score:{u.OFF} {best_bo_cv}")
        self.vprint.v1(f"\nTotal time elapsed: {round(t_end-t_start,2)} sec\n")
        
        # # if out_path exists then write pipes to file
        # if out_path:
        #     if not os.path.exists(out_path):
        #         os.makedirs(out_path)
        #     fname_bo_pipes = os.path.join(out_path,'TPOT-BO-H.pipes')
        #     # write all evaluated pipes
        #     # with open(fname_bo_pipes, 'a') as f:
        #     #     for k,v in self.pipes.items():
        #     #         if v['source'] == 'TPOT-BO-H':
        #     #             f.write(f"{k};{v['internal_cv_score']}\n")
                    
        return "Successful"
                    