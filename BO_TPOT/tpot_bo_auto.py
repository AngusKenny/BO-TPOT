#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 16:41:47 2022

@author: gus
"""

from config.tpot_config import default_tpot_config_dict
from tpot import TPOTRegressor
from deap import creator
import numpy as np
from BO_TPOT.tpot_bo_tools import TPOT_BO_Handler
from BO_TPOT.tpot_bo_s import TPOT_BO_S
import utils.tpot_utils as u
import copy
import os
import time
import pickle


class TPOT_BO_AUTO(object):    
    def __init__(self,
                 pop_size=100,
                 n_gens=100,
                 seed=42,
                 discrete_mode=True,
                 optuna_timeout_trials=100,
                 config_dict=default_tpot_config_dict,
                 n_jobs=-1,
                 pipe_eval_timeout=5,
                 vprint=u.Vprint(1)):
        
        self.pipes = {}
        self.pop_size = pop_size
        self.n_gens = n_gens
        self.seed=seed
        self.discrete_mode=discrete_mode
        self.optuna_timeout_trials=optuna_timeout_trials
        self.config_dict=copy.deepcopy(config_dict)
        self.n_jobs=n_jobs
        self.pipe_eval_timeout=pipe_eval_timeout
        self.vprint=vprint
        self.type_flag = "d" if discrete_mode else "c"
        
        # set tpot verbosity to vprint.verbosity + 1 to give more information
        self.tpot_verb = vprint.verbosity + 1 if vprint.verbosity > 0 else 0
                
        # create TPOT object
        self.tpot = TPOTRegressor(generations=0,
                                  population_size=self.pop_size, 
                                  mutation_rate=0.9, 
                                  crossover_rate=0.1, 
                                  cv=5,
                                  verbosity=self.tpot_verb, 
                                  config_dict=copy.deepcopy(self.config_dict),
                                  random_state=self.seed, 
                                  n_jobs=self.n_jobs,
                                  warm_start=True,
                                  max_eval_time_mins=self.pipe_eval_timeout)
        
        self.tpot._fit_init()
        
        # initialise tpot bo handler
        self.tpot_handler = TPOT_BO_Handler(self.tpot, vprint=self.vprint, discrete_mode=self.discrete_mode)
        
        
    def optimize(self, X_train, y_train, out_path=None):
        t_start = time.time() 
        
        start_gen = 1
        log_file = None
        
        # force tpot to be run once first, then bo
        grads = {0: {f'TPOT-BO-AUTO{self.type_flag}(TPOT)':1e20, f'TPOT-BO-AUTO{self.type_flag}(BO)':0}}

        if out_path:
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            fname_auto_pipes = os.path.join(out_path,f"TPOT-BO-AUTO{self.type_flag}.pipes")
            fname_pickle_pipes = os.path.join(out_path,f'TPOT-BO-AUTO{self.type_flag}_pipes.pickle')
            fname_pickle_grads = os.path.join(out_path,f'TPOT-BO-AUTO{self.type_flag}_grads.pickle')
            log_file = os.path.join(out_path,f'TPOT-BO-AUTO{self.type_flag}.log')
        
            if os.path.exists(fname_pickle_pipes) and os.path.exists(fname_pickle_grads):
                with open(fname_pickle_pipes, 'rb') as f:
                    self.pipes = pickle.load(f)
                with open(fname_pickle_grads, 'rb') as f:
                    grads = pickle.load(f)
                start_gen = max([v['auto_gen'] for v in self.pipes.values()]) + 1
                self.vprint.v1(f"{u.RED}Loaded {start_gen-1} generations from previous interrupted run.. continuing..{u.OFF}")
        
        else:
            self.vprint.v2(f"{u.CYAN}\nevaluating initial population..{u.OFF}")
            self.tpot.fit(X_train, y_train)
        
        self.tpot.log_file=log_file
        
        self.tpot.generations = 1
        
        best_cv = -1e20
        
        do_tpot = True
        chosen_method = f'TPOT-BO-AUTO{self.type_flag}(TPOT)'
        
        # skip 0 for init pop
        for g in range(start_gen, self.n_gens):
            
            best_pipe,best_cv = u.get_best(self.pipes)
            best_tpot_idx,best_tpot_cv = self.tpot_handler.get_best_pipe_idx()
            
            self.vprint.v1(f"\n{u.YELLOW}best pipe found at generation "
                      + f"{g-1} ({chosen_method}):{u.OFF}")
            self.vprint.v2(f"{best_pipe}")
            self.vprint.v1(f"{u.GREEN}* score: {u.OFF}{best_cv}")
            self.vprint.v1(f"{u.GREEN}* slopes: {u.OFF}{grads[g-1]}\n")                
            
            self.vprint.v1(f"{u.CYAN_U}({time.strftime('%d %b, %H:%M', time.localtime())}) Generation: {g}{u.OFF}")
            old_best_cv = best_cv
            
            # if gradient is the same then toggle, otherwise take best
            if grads[g-1][f'TPOT-BO-AUTO{self.type_flag}(BO)'] == grads[g-1][f'TPOT-BO-AUTO{self.type_flag}(TPOT)']:
                do_tpot = not do_tpot
            elif grads[g-1][f'TPOT-BO-AUTO{self.type_flag}(BO)'] < grads[g-1][f'TPOT-BO-AUTO{self.type_flag}(TPOT)']:
                do_tpot = True
            else:
                do_tpot = False
            
            # check previous gradient to determine which to use
            if do_tpot:
                chosen_method = f'TPOT-BO-AUTO{self.type_flag}(TPOT)'
                grads[g] = {f'TPOT-BO-AUTO{self.type_flag}(BO)':grads[g-1][f'TPOT-BO-AUTO{self.type_flag}(BO)']}
                self.vprint.v2(f"{u.YELLOW}Running TPOT..\nseed {self.seed} - see {log_file} for progress{u.OFF}")
                # copy pipe dictionary to evaluated individuals
                self.tpot.fit(X_train, y_train)
                self.vprint.v1("")
                new_idx,new_tpot_cv = self.tpot_handler.get_best_pipe_idx()
                
                # update pipes
                for k,v in self.tpot.evaluated_individuals_.items():
                    if k not in self.pipes:
                        self.pipes[k] = v
                        self.pipes[k]['auto_gen'] = g
                        self.pipes[k]['source'] = chosen_method
            else:
                chosen_method = f'TPOT-BO-AUTO{self.type_flag}(BO)'
                grads[g] = {f'TPOT-BO-AUTO{self.type_flag}(TPOT)':grads[g-1][f'TPOT-BO-AUTO{self.type_flag}(TPOT)']}
                self.vprint.v2(f"{u.YELLOW}Running BO..{u.OFF}")

                # get all pipelines that match the structure of best pipe
                best_pipe_set = u.get_matching_set(best_pipe, self.pipes)
                
                tpot_bo_s = TPOT_BO_S(best_pipe_set,
                                  seed=self.seed,
                                  n_bo_evals=self.pop_size,
                                  discrete_mode=self.discrete_mode,
                                  config_dict=self.config_dict,
                                  pipe_eval_timeout=self.pipe_eval_timeout,
                                  source_method=f'TPOT-BO-AUTO{self.type_flag}(BO)',
                                  vprint=self.vprint)
            
                tpot_bo_s.optimize(X_train,y_train)
                
                best_bo_pipe = ""
                
                for k,v in tpot_bo_s.pipes.items():
                    if k not in self.pipes:
                        if v['internal_cv_score'] > new_cv:
                            new_cv = v['internal_cv_score']
                            best_bo_pipe = k
                        self.pipes[k] = v
                        self.pipes[k]['auto_gen'] = g
                        self.pipes[k]['source'] = chosen_method
                    
                    
            self.tpot.evaluated_individuals_ = copy.deepcopy(self.pipes)    
            
            new_pipe,new_cv = u.get_best(self.pipes)
                            
            if new_cv > best_cv:                        
                self.vprint.v1(f"{u.GREEN}improvement attempt successful!{u.OFF}")
                # update main pset with new individual
                if "(BO)" in chosen_method:
                    self.vprint.v2("updating main pset..")
                    
                    new_params = u.string_to_params(best_bo_pipe)
                    for (p,v) in new_params:
                        self.tpot_handler.add_param_to_pset(p, v)
                    
                    # create new pipe object with best params
                    new_pipe = creator.Individual.from_string(best_bo_pipe, self.tpot._pset)
                
                    self.vprint.v2(f"replacing pipe {best_tpot_idx} with best BO pipe and re-evaluating..\n")
                
                    self.tpot._pop[best_tpot_idx] = new_pipe
                
                    # evaluate new pipe
                    self.tpot_handler.evaluate(best_tpot_idx, X_train, y_train)
                    
                    self.vprint.v1(f"CV of new evaluated tpot pipe: {self.tpot._pop[best_tpot_idx].fitness.values[1]}")
                
            else:
                self.vprint.v1(f"{u.RED}improvement attempt unsuccessful - reverting to original"
                            + f" TPOT population..{u.OFF}")

            grads[g][chosen_method] = new_cv - best_cv
            
            # write current pipe list to file
            if out_path:
                with open(fname_pickle_pipes, 'wb') as f:
                    # Pickle the 'data' dictionary using the highest protocol available.
                    pickle.dump(self.pipes, f, pickle.HIGHEST_PROTOCOL)
                with open(fname_pickle_grads, 'wb') as f:
                    # Pickle the 'data' dictionary using the highest protocol available.
                    pickle.dump(grads, f, pickle.HIGHEST_PROTOCOL)
                    
                with open(fname_auto_pipes,'w') as f:
                    for k,v in self.pipes.items():
                        f.write(f"{k};{v['auto_gen']};{v['source']};"
                                    + f"{v['internal_cv_score']}\n")
            
        t_end = time.time() 
        
        best_pipe, best_cv = u.get_best(self.pipes)
        
        self.vprint.v1(f"\n{u.YELLOW}* best pipe found:{u.OFF}")
        self.vprint.v1(f"{best_pipe}")
        self.vprint.v1(f"{u.GREEN} * score:{u.OFF} {best_cv}")
        self.vprint.v1(f"\nTotal time elapsed: {round(t_end-t_start,2)} sec\n")
        
        if out_path:
            if os.path.exists(fname_pickle_pipes):
                os.remove(fname_pickle_pipes)
            if os.path.exists(fname_pickle_grads):
                os.remove(fname_pickle_grads)
            if os.path.exists(log_file):
                try:
                    self.tpot.log_file_.close()
                    os.remove(log_file)
                except:
                    pass
        
        return "Successful"