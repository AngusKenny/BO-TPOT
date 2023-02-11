#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 13:54:47 2022

@author: gus
"""

from config.tpot_config import default_tpot_config_dict
from tpot import TPOTRegressor
from deap import creator
from BO_TPOT.tpot_bo_tools import TPOT_BO_Handler
from BO_TPOT.tpot_bo_s import TPOT_BO_S
import utils.tpot_utils as u
import copy
import os
import time


class TPOT_BO_ALT(object):   
    def __init__(self,
                 n_iters=10,
                 pop_size=100,
                 n_tpot_gens=8,
                 n_total_gens=100,
                 n_bo_evals=200,
                 seed=42,
                 discrete_mode=True,
                 optuna_timeout_trials=100,
                 config_dict=default_tpot_config_dict,
                 n_jobs=-1,
                 pipe_eval_timeout=5,
                 vprint=u.Vprint(1)):
        
        self.pipes = {}
        self.n_iters = n_iters
        self.pop_size = pop_size
        self.seed=seed
        self.discrete_mode=discrete_mode
        self.optuna_timeout_trials=optuna_timeout_trials
        self.config_dict=copy.deepcopy(config_dict)
        self.n_jobs=-1
        self.n_tpot_gens = n_tpot_gens
        self.n_total_gens = n_total_gens
        # self.n_bo_evals = n_bo_evals
        self.pipe_eval_timeout=pipe_eval_timeout
        self.vprint=vprint
        self.d_flag = "d" if discrete_mode else "c"
        
        self.n_bo_evals = int((self.n_total_gens - self.n_iters * self.n_tpot_gens) * (self.pop_size/self.n_iters))
        
        # set tpot verbosity to vprint.verbosity + 1 to give more information
        self.tpot_verb = vprint.verbosity + 1 if vprint.verbosity > 0 else 0
        
        # create TPOT object
        self.tpot = TPOTRegressor(generations=self.n_tpot_gens-1,
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

        # add TPOT BO handler object to let us change the pset
        self.tpot_handler = TPOT_BO_Handler(self.tpot, vprint=self.vprint, discrete_mode=self.discrete_mode)
        
    def optimize(self, X_train, y_train, out_path=None):
        if out_path:
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            fname_alt_pipes = os.path.join(out_path,f"TPOT-BO-ALT{self.d_flag}.pipes")
            # wipe pipe existing pipe files if they exist
            f = open(fname_alt_pipes,'w')
            f.close()

        t_start = time.time()
        
        for i in range(self.n_iters):
            self.vprint.v1(f"{u.CYAN_U}Iteration: {i}{u.OFF}")
            
            self.vprint.v2(f"{u.CYAN}\nfitting tpot model with "
                      + f"{self.tpot.generations} generations..\n{u.OFF}")
            
            # fit tpot object
            self.tpot.fit(X_train, y_train)
            
            # if first iteration change number of generations
            if i == 0:
                self.tpot.generations = self.n_tpot_gens
            
            if out_path:
                f = open(fname_alt_pipes,'a')
            
            for k,v in self.tpot.evaluated_individuals_.items():
                if k not in self.pipes:
                    self.pipes[k] = v
                    self.pipes[k]['iteration'] = i
                    self.pipes[k]['source'] = f'TPOT-BO-ALT{self.d_flag}(TPOT)'
                    
                    if out_path:
                        f.write(f"{k};{self.pipes[k]['iteration']};{self.pipes[k]['generation']};"
                                + f"{self.pipes[k]['source']};{self.pipes[k]['internal_cv_score']}\n")
            
            if out_path:
                f.close()
            
            # get best pipe from tpot population and its 
            # corresponding string and params
            best_iter_idx,best_iter_cv = self.tpot_handler.get_best_pipe_idx()
            best_iter_pipe = str(self.tpot._pop[best_iter_idx])
            best_pipe_set = u.get_matching_set(best_iter_pipe, self.tpot.evaluated_individuals_)
            
            self.vprint.v1(f"\n{u.YELLOW}best pipe found by tpot for iteration "
                      + f"{i}:{u.OFF}")
            self.vprint.v2(f"{best_iter_pipe}")
            self.vprint.v1(f"{u.GREEN}* score: {u.OFF}{best_iter_cv}\n")
                               
            tpot_bo_s = TPOT_BO_S(best_pipe_set,
                                  seed=self.seed,
                                  n_bo_evals=self.n_bo_evals,
                                  discrete_mode=self.discrete_mode,
                                  config_dict=self.config_dict,
                                  pipe_eval_timeout=self.pipe_eval_timeout,
                                  source_method=f'TPOT-BO-ALT{self.d_flag}(BO)',
                                  vprint=self.vprint)
            
            tpot_bo_s.optimize(X_train,y_train)
            
            
            
            
            
            # # create BO TPOT object 
            # bo_tpot = TPOTRegressor(generations=0,
            #                         population_size=1, 
            #                         mutation_rate=0.9, 
            #                         crossover_rate=0.1, 
            #                         cv=5,
            #                         verbosity=self.tpot_verb, 
            #                         config_dict=copy.deepcopy(self.config_dict), 
            #                         random_state=self.seed, 
            #                         n_jobs=self.n_jobs,
            #                         warm_start=True,
            #                         max_eval_time_mins=self.pipe_eval_timeout)
            
            # # initialise bo tpot object to generate pset
            # bo_tpot._fit_init()
            
            # bo_tpot.evaluated_individuals_ = u.get_matching_set(best_iter_pipe, self.tpot.evaluated_individuals_)
            
            # seed_samples = [(u.string_to_params(k), v['internal_cv_score']) for k,v in bo_tpot.evaluated_individuals_.items()]
            
            # # initialise bo pipe optimiser object
            # bo_po = TPOT_BO_Handler(bo_tpot, vprint=self.vprint, discrete_mode=self.discrete_mode)
        
            # # update pset of BO tpot object
            # for (p,v) in best_iter_params:
            #     bo_po.add_param_to_pset(p, v)
            
            # # remove generated pipeline and transplant saved from before
            # bo_tpot._pop = [creator.Individual.from_string(best_iter_pipe, bo_tpot._pset)]
            
            # # fit for 0 gens to load into model
            # self.vprint.v2(f"{u.CYAN}\n"
            #           + f"({time.strftime('%d %b, %H:%M', time.localtime())})" 
            #           + " fitting temporary bo tpot model with "  
            #           + f"{bo_tpot.generations} generations..\n{u.WHITE}")
            
            # bo_tpot.fit(X_train, y_train)
            
            # self.vprint.v1("")
            
            # self.vprint.v2("Transplanting best pipe and optimising for " 
            #           + f"{self.n_bo_evals} evaluations..\n")
            
            # # initialise bo pipe optimiser object
            # bo_po = TPOT_BO_Handler(bo_tpot, vprint=self.vprint, discrete_mode=self.discrete_mode)
            
            # # run bayesian optimisation with seed_dicts as initial samples
            # bo_po.optimise(0, X_train, y_train, n_evals=self.n_bo_evals,
            #                    seed_samples=seed_samples,discrete_mode=self.discrete_mode, 
            #                    timeout_trials=self.optuna_timeout_trials)
            
            # # best_bo_pipe, best_bo_cv = u.get_best(bo_tpot.evaluated_individuals_)
            
            best_bo_pipe = ""
            best_bo_cv = -1e40
            
            if out_path:
                f = open(fname_alt_pipes,'a')
                        
            for k,v in tpot_bo_s.pipes.items():
                if k not in self.pipes:
                    if v['internal_cv_score'] > best_bo_cv:
                        best_bo_cv = v['internal_cv_score']
                        best_bo_pipe = k
                    self.pipes[k] = v
                    self.pipes[k]['iteration'] = i
                    self.pipes[k]['source'] = f'TPOT-BO-ALT{self.d_flag}(BO)'
                    self.pipes[k]['generation'] = -1
                if k not in self.tpot.evaluated_individuals_:
                    self.tpot.evaluated_individuals_[k] = v
                    self.tpot.evaluated_individuals_[k]['iteration'] = i
                    self.tpot.evaluated_individuals_[k]['source'] = f'TPOT-BO-ALT{self.d_flag}(BO)'
                    self.tpot.evaluated_individuals_[k]['generation'] = -1
                    
                    if out_path:
                        f.write(f"{k};{self.pipes[k]['iteration']};{self.pipes[k]['generation']};"
                                + f"{self.pipes[k]['source']};{self.pipes[k]['internal_cv_score']}\n")
            if out_path:
                f.close()
            
            self.vprint.v1(f"{u.YELLOW}* best pipe found by BO:{u.OFF}")
            self.vprint.v2(f"{best_bo_pipe}")
            self.vprint.v1(f"{u.GREEN}* score: {u.OFF}{best_bo_cv}\n")
            
            self.vprint.v1(f"{u.YELLOW}* best pipe found by tpot for iteration "
                           + f"{i}:{u.OFF}")
            self.vprint.v2(f"{best_iter_pipe}")
            self.vprint.v1(f"{u.GREEN}* score: {u.OFF}{best_iter_cv}\n")
            
            bo_success = best_bo_cv > best_iter_cv
        
            if bo_success:
                self.vprint.v1(f"{u.GREEN}BO successful!{u.OFF}")
                # update main pset with new individual
                self.vprint.v2("updating main pset..")
                new_params = u.string_to_params(best_bo_pipe)
                for (p,v) in new_params:
                    self.tpot_handler.add_param_to_pset(p, v)
            
                # create new pipe object with best params
                new_pipe = creator.Individual.from_string(best_bo_pipe, self.tpot._pset)
            
                self.vprint.v2(f"replacing pipe {best_iter_idx} with best BO pipe and re-evaluating..\n")
            
                self.tpot._pop[best_iter_idx] = new_pipe
                
                # # add to evaluated individuals
                # self.tpot.evaluated_individuals_[best_bo_pipe] = bo_tpot.evaluated_individuals_[best_bo_pipe]
                
                # evaluate new pipe
                self.tpot_handler.evaluate(best_iter_idx, X_train, y_train)
                
                self.vprint.v1(f"CV of new evaluated tpot pipe: {self.tpot._pop[best_iter_idx].fitness.values[1]}")
                                
            else:
                self.vprint.v1(f"{u.RED}BO unsuccessful - reverting to original"
                               + f" TPOT population..{u.OFF}")
        
        t_end = time.time() 
        
        best_tpot_pipe, best_tpot_cv = u.get_best(self.pipes,source=f'TPOT-BO-ALT{self.d_flag}(TPOT)')
        best_bo_pipe, best_bo_cv = u.get_best(self.pipes,source=f'TPOT-BO-ALT{self.d_flag}(BO)')
        
        self.vprint.v1(f"\n{u.YELLOW}* best pipe found by tpot:{u.OFF}")
        self.vprint.v1(f"{best_tpot_pipe}")
        self.vprint.v1(f"{u.GREEN} * score:{u.OFF} {best_tpot_cv}")
        self.vprint.v1(f"\n{u.YELLOW}best pipe found by BO:{u.OFF}")
        self.vprint.v1(f"{best_bo_pipe}\n{u.GREEN} * score:{u.OFF} {best_bo_cv}")
        self.vprint.v1(f"\nTotal time elapsed: {round(t_end-t_start,2)} sec\n")
                    
        return "Successful"