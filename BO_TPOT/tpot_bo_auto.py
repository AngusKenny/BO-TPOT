#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 16:41:47 2022

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


class TPOT_BO_AUTO(object):
    pipes = {}
    
    def __init__(self,
                 init_pipes={},
                 pop_size=100,
                 n_gens=100,
                 seed=42,
                 discrete_mode=True,
                 optuna_timeout_trials=100,
                 config_dict=default_tpot_config_dict,
                 n_jobs=-1,
                 pipe_eval_timeout=5,
                 vprint=u.Vprint(1)):
        
        self.pop_size = pop_size
        self.n_gens = n_gens
        self.seed=seed
        self.discrete_mode=discrete_mode
        self.optuna_timeout_trials=optuna_timeout_trials
        self.config_dict=copy.deepcopy(config_dict)
        self.n_jobs=-1
        self.pipe_eval_timeout=pipe_eval_timeout
        self.vprint=vprint
        
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
        
        for k,v in init_pipes.items():
            v['source'] = 'TPOT-BO-AUTO(TPOT)'
            v['auto_gen'] = 0
        
        if len(init_pipes) > 0:
            init_pop = u.truncate_pop(init_pipes, 0)
            
            if len(init_pop) != self.pop_size:
                self.vprint.vwarn(f"Initial population from init_pipes different size to pop_size ({len(init_pop)} != {self.pop_size}) - using what is there..")
            
            self.vprint.v2(f"{u.CYAN}\nimporting initial population and initializing..{u.OFF}")
            # initialise loaded population
            self.tpot._pop = []       
            for k,v in init_pop.items():
                self.tpot._pop.append(creator.Individual.from_string(k, self.tpot._pset))
            
        self.tpot.evaluated_individuals_ = copy.deepcopy(init_pipes)
        
        # initialise tpot bo handler
        self.tpot_handler = TPOT_BO_Handler(self.tpot, vprint=self.vprint, discrete_mode=self.discrete_mode)
        
        
    def optimize(self, X_train, y_train, out_path=None):
        t_start = time.time() 
        
        self.vprint.v2(f"{u.CYAN}\nevaluating initial population..{u.OFF}")
        self.tpot.fit(X_train, y_train)
        
        if out_path:
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            fname_auto_pipes = os.path.join(out_path,"TPOT-BO-AUTO.pipes")
            # wipe pipe existing pipe files if they exist
            f = open(fname_auto_pipes,'w')
            
        
        for k,v in self.tpot.evaluated_individuals_.items():
            self.pipes[k] = v
            if out_path:
                f.write(f"{k};{v['auto_gen']};{v['source']};"
                        + f"{v['internal_cv_score']}\n")
            
        if out_path:
            f.close()
        
        self.tpot.generations = 1
        
        # force tpot to be run once first, then bo
        grads = {0: {'TPOT-BO-AUTO(TPOT)':1e20, 'TPOT-BO-AUTO(BO)':1e10}}
        
        old_best_cv = -1e20
        
        do_tpot = True
        chosen_method = 'TPOT-BO-AUTO(TPOT)'
        
        # skip 0 for init pop
        for g in range(1, self.n_gens):
            
            best_pipe,best_cv = u.get_best(self.tpot.evaluated_individuals_)
            
            if out_path:
                f = open(fname_auto_pipes,'a')
            
            for k,v in self.tpot.evaluated_individuals_.items():
                if k not in self.pipes:
                    self.pipes[k] = v
                    self.pipes[k]['auto_gen'] = g
                    self.pipes[k]['source'] = chosen_method
                    
                    if out_path:
                        f.write(f"{k};{self.pipes[k]['auto_gen']};{self.pipes[k]['source']};"
                                + f"{self.pipes[k]['internal_cv_score']}\n")
            
            if out_path:
                f.close()
            
            grads[g-1][chosen_method] = best_cv - old_best_cv
            self.vprint.v1(f"\n{u.YELLOW}best pipe found at generation "
                      + f"{g-1} ({chosen_method}):{u.OFF}")
            self.vprint.v2(f"{best_pipe}")
            self.vprint.v1(f"{u.GREEN}* score: {u.OFF}{best_cv}")
            self.vprint.v1(f"{u.GREEN}* slopes: {u.OFF}{grads[g-1]}\n")                
            
            self.vprint.v1(f"{u.CYAN_U}({time.strftime('%d %b, %H:%M', time.localtime())}) Generation: {g}{u.OFF}")
            old_best_cv = best_cv
            
            # if gradient is the same then toggle, otherwise take best
            if grads[g-1]['TPOT-BO-AUTO(BO)'] == grads[g-1]['TPOT-BO-AUTO(TPOT)']:
                do_tpot = not do_tpot
            elif grads[g-1]['TPOT-BO-AUTO(BO)'] < grads[g-1]['TPOT-BO-AUTO(TPOT)']:
                do_tpot = True
            else:
                do_tpot = False
                
            improve_success = False
            
            # check previous gradient to determine which to use
            if do_tpot:
                chosen_method = 'TPOT-BO-AUTO(TPOT)'
                grads[g] = {'TPOT-BO-AUTO(BO)':grads[g-1]['TPOT-BO-AUTO(BO)']}
                self.vprint.v2(f"{u.YELLOW}Running TPOT..{u.OFF}")
                self.tpot.fit(X_train, y_train)
                self.vprint.v1("")
                new_idx,new_cv = self.tpot_handler.get_best_pipe_idx()
                improve_success = new_cv > best_cv
            else:
                chosen_method = 'TPOT-BO-AUTO(BO)'
                grads[g] = {'TPOT-BO-AUTO(TPOT)':grads[g-1]['TPOT-BO-AUTO(TPOT)']}
                self.vprint.v2(f"{u.YELLOW}Running BO..{u.OFF}")

                # get all pipelines that match the structure of best pipe
                matching = u.get_matching_set(best_pipe, self.tpot.evaluated_individuals_)
                
                # create BO TPOT object 
                bo_tpot = TPOTRegressor(generations=0,
                                      population_size=1, 
                                      mutation_rate=0.9, 
                                      crossover_rate=0.1, 
                                      cv=5,
                                      verbosity=self.tpot_verb, 
                                      config_dict=copy.deepcopy(self.config_dict), 
                                      random_state=self.seed, 
                                      n_jobs=self.n_jobs,
                                      warm_start=True,
                                      max_eval_time_mins=self.pipe_eval_timeout)
                
                # initialise bo tpot object to generate pset
                bo_tpot._fit_init()
                
                # share evaluated individuals dict across methods
                bo_tpot.evaluated_individuals_ = copy.deepcopy(self.tpot.evaluated_individuals_)
                
                seed_samples = [(u.string_to_params(k), v['internal_cv_score']) for k,v in matching.items()]
                
                # initialise bo pipe optimiser object
                bo_po = TPOT_BO_Handler(bo_tpot, vprint=self.vprint, discrete_mode=self.discrete_mode)
                
                best_params = u.string_to_params(best_pipe)
                
                # update pset of BO tpot object
                for (p,v) in best_params:
                    bo_po.add_param_to_pset(p, v)
                
                # remove generated pipeline and transplant saved from before
                bo_tpot._pop = [creator.Individual.from_string(best_pipe, bo_tpot._pset)]
            
                # fit for 0 gens to load into model
                self.vprint.v2(f"{u.CYAN}\n"
                          + "fitting temporary bo tpot model with "  
                          + f"0 generations..\n{u.WHITE}")
            
                bo_tpot.fit(X_train, y_train)
            
                self.vprint.v1("")
            
                self.vprint.v2("Transplanting best pipe and optimising for " 
                          + f"{self.pop_size} evaluations..\n")
            
                # re-initialise bo pipe optimiser object
                bo_po = TPOT_BO_Handler(bo_tpot, vprint=self.vprint, discrete_mode=self.discrete_mode)
            
                # run bayesian optimisation with seed_dicts as initial samples
                bo_po.optimise(0, X_train, y_train, n_evals=self.pop_size,
                                   seed_samples=seed_samples,discrete_mode=self.discrete_mode, 
                                   timeout_trials=self.optuna_timeout_trials)
                
                bo_best_cv = bo_po.best_score
                bo_best_pipe = bo_po.best_pipe
                
                improve_success = bo_best_cv > best_cv
                            
                if improve_success:
                    best_iter_idx = None
                    for i in range(len(self.tpot._pop)):
                        if str(self.tpot._pop[i]) == best_pipe:
                            best_iter_idx = i
                            break
                    if not best_iter_idx:
                        self.vprint.verr("Unable to replace pipeline!")
                        
                    self.vprint.v1(f"{u.GREEN}BO successful!{u.OFF}")
                    # update main pset with new individual
                    self.vprint.v2("updating main pset..")
                    
                    
                    new_params = u.string_to_params(bo_best_pipe)
                    for (p,v) in new_params:
                        self.tpot_handler.add_param_to_pset(p, v)
                
                    # create new pipe object with best params
                    new_pipe = creator.Individual.from_string(bo_best_pipe, self.tpot._pset)
                
                    self.vprint.v2(f"replacing pipe {best_iter_idx} with best BO pipe and re-evaluating..\n")
                
                    self.tpot._pop[best_iter_idx] = new_pipe
                    
                    # swap evaluated dict back
                    self.tpot.evaluated_individuals_= copy.deepcopy(bo_tpot.evaluated_individuals_)
                    
                    # evaluate new pipe
                    self.tpot_handler.evaluate(best_iter_idx, X_train, y_train)
                    
                    self.vprint.v1(f"CV of new evaluated tpot pipe: {self.tpot._pop[best_iter_idx].fitness.values[1]}")
                    
                    best_pipe = bo_best_pipe
                    best_cv = bo_best_cv
                    
                else:
                    self.vprint.v1(f"{u.RED}BO unsuccessful - reverting to original"
                              + f" TPOT population..{u.OFF}")
        
        if out_path:
            f = open(fname_auto_pipes,'a')

        for k,v in self.tpot.evaluated_individuals_.items():
            if k not in self.pipes:
                self.pipes[k] = v
                self.pipes[k]['auto_gen'] = g
                self.pipes[k]['source'] = chosen_method
                
                if out_path:
                    f.write(f"{k};{self.pipes[k]['auto_gen']};{self.pipes[k]['source']};"
                            + f"{self.pipes[k]['internal_cv_score']}\n")
        if out_path:
            f.close()
            
        t_end = time.time() 
        
        best_pipe, best_cv = u.get_best(self.pipes)
        
        self.vprint.v1(f"\n{u.YELLOW}* best pipe found:{u.OFF}")
        self.vprint.v1(f"{best_pipe}")
        self.vprint.v1(f"{u.GREEN} * score:{u.OFF} {best_cv}")
        self.vprint.v1(f"\nTotal time elapsed: {round(t_end-t_start,2)} sec\n")
        
        if out_path:
            fname_auto_grads = os.path.join(out_path,"TPOT-BO-AUTO.grads")
            with open(fname_auto_grads,'w') as f:
                for g,v in grads.items():
                    if 'TPOT-BO-AUTO(TPOT)' in v and 'TPOT-BO-AUTO(BO)' in v:
                        f.write(f"{g},{v['TPOT-BO-AUTO(TPOT)']},{v['TPOT-BO-AUTO(BO)']}\n")
                        
        return "Successful"