#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 12:22:43 2022

@author: gus

- selelct top k=100 ND (cv,n_hps) structures at TPOT80
- perform BO evals to ensure at least 2 solutions from each structure
- R = 2000 remaining budget
- while R > 0:
    - compute statistics
    - compute generation budget of G = R/(ceil(log2(k)) + 1)
    - m = ceil(k/2)
    - select top m of k structures with OCBA-m
    - allocate G among top m structures with info from OCBA-m
    - perform BO evaluations per allocations
    - R = R - G
    - k = m

* still not 100% sure on the ND selection. it could mean that 
  we dont choose the right initial population if there are two 
  structures with 6 hps each with similar cvs, but very different
  tree structure, and only one will be selected.. maybe best to 
  do the clustering?
"""
from config.tpot_config import default_tpot_config_dict
from tpot import TPOTRegressor
from deap import creator
from BO_TPOT.tpot_bo_tools import TPOT_BO_Handler
import utils.tpot_utils as u
import utils.ocba_m as ocba_m
import copy
import os
import time
import numpy as np
import utils.ocba as o
import pickle

EPS = 1e-10
MAX_SIGMA = 1e10

class TPOT_BO_O(object):    
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
                 n_0=10,
                 Delta=50,
                 m_strucs=5,
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
        self.d_flag = 'd' if discrete_mode else 'c'
        self.n_0 = n_0
        self.Delta = Delta
        self.m_strucs=m_strucs
        
        # set tpot verbosity to vprint.verbosity + 1 to give more information
        self.tpot_verb = vprint.verbosity + 1 if vprint.verbosity > 0 else 0
        
        for k,v in self.tpot_pipes.items():
            v['source'] = f'TPOT-BASE'
        
        # get unique structures
        self.strucs = u.get_structures(self.tpot_pipes, config_dict=self.config_dict)
                
        # get keys by CV ranking
        cvs = np.array([-self.strucs[k].cv for k in self.strucs.keys()])
        cv_idxs = np.argsort(cvs)        
        key_list = list(self.strucs.keys())
        
        remove_ids = []                
        # check key_list for any structures that cannot be used
        for i in range(cv_idxs.shape[0]):
            if self.strucs[key_list[cv_idxs[i]]].n_bo_params < 1:
                remove_ids.append(i)

        cv_idxs = np.delete(cv_idxs, remove_ids)
                        
        self.bo_struc_keys = [key_list[i] for i in cv_idxs[:int(pop_size*bo_pop_factor)]]
        
        for k in self.bo_struc_keys:
            self.pipes.update(self.strucs[k].pipes)
        
        self.starting_size = len(self.pipes)
        
        vprint.v2(f"\n{u.CYAN}{len(self.bo_struc_keys)} structures in BO set..{u.OFF}\n")
        
    def optimize(self, X_train, y_train, out_path=None):            
        t_start = time.time()
        self.vprint.v1("")
        early_finish = ""
        n_extra_bo = 0
        gen = 0
        tracking = []

        tpots = []
        handlers = []
        
        fname_pickle = os.path.join(out_path,f'TPOT-BO-O{self.d_flag}.pickle')
        
        start_delta = self.pop_size
        
        Deltas = [start_delta]
        
        for p,v in self.pipes.items():
            v['delta'] = self.pop_size
            v['structure'] = u.string_to_bracket(p)
        
        if out_path and os.path.exists(fname_pickle):
            # unpickle previous
            with open(fname_pickle, 'rb') as f:
                self.pipes = pickle.load(f)
            # update structures
            self.strucs.update(self.pipes)
            # find starting delta
            start_delta = min([v['delta'] for v in self.pipes.values() if v['source'] != "TPOT-BASE"])
            gen = max([v['generation'] for v in self.pipes.values() if v['source'] != "TPOT-BASE"])
            Deltas = list({v['delta'] for v in self.pipes.values()})
            Deltas.sort(reverse=True)
            self.vprint.v2(f"{u.RED}Loaded {gen} generations from previous interrupted run.. continuing from Delta = {np.ceil(start_delta/2)}..{u.OFF}\n")
            
        # perform extra evaluations to initialise
        for i,k in enumerate(self.bo_struc_keys):
            # if len(self.strucs[k]) < self.n_0:

            tracking = [[0 for k in self.bo_struc_keys]]
            
            tpots.append(TPOTRegressor(generations=0,
                                    population_size=1, 
                                    mutation_rate=0.9, 
                                    crossover_rate=0.1, 
                                    cv=5,
                                    verbosity=self.tpot_verb, 
                                    config_dict=copy.deepcopy(self.config_dict),
                                    random_state=self.seed, 
                                    n_jobs=1,
                                    warm_start=True,
                                    max_eval_time_mins=self.pipe_eval_timeout))
                
            # initialise tpot object to generate pset
            tpots[i]._fit_init()
            
            tpots[i].evaluated_individuals_ = self.strucs[k].pipes
            
            # initialise tpot bo handler
            handlers.append(TPOT_BO_Handler(tpots[i], vprint=self.vprint, discrete_mode=self.discrete_mode))
            new_params = self.strucs[k].bo_params
            
            # update pset of BO tpot object
            for (p,val) in new_params:
                handlers[i].add_param_to_pset(p, val)
            
            # remove generated pipeline and transplant saved from before
            tpots[i]._pop = [creator.Individual.from_string(self.strucs[k].best, tpots[i]._pset)]
            
            tpots[i].fit(X_train, y_train)
            
            # re-initialise tpot bo handler
            handlers[i] = TPOT_BO_Handler(tpots[i], vprint=self.vprint, discrete_mode=self.discrete_mode)
            
            extra_bo = max(self.n_0 - len(self.strucs[k].get_valid()),0)
            
            stagnate_cnt = 0
            
            if extra_bo > 0:
                self.vprint.v2(f"{u.CYAN}[Seed {self.seed}] Performing {extra_bo} initial evaluations on structure ({i}/{len(self.bo_struc_keys)}): {k}..{u.OFF}\n")
            
            old_invalid_cnt = len(self.strucs[k]) - len(self.strucs[k].get_valid())
            
            while len(self.strucs[k].get_valid()) < self.n_0 and stagnate_cnt < 1000:
                old_size = len(tpots[i].evaluated_individuals_)
                extra_bo = max(self.n_0 - len(self.strucs[k].get_valid()),0)
                # run bayesian optimisation with seed_dicts as initial samples
                handlers[i].optimise(0, X_train, y_train, n_evals=extra_bo,
                                seed_samples=self.strucs[k].get_seed_samples(), 
                                discrete_mode=self.discrete_mode,
                                skip_params=[],
                                timeout_trials=self.optuna_timeout_trials)
                
                stagnate_cnt += 1 if len(tpots[i].evaluated_individuals_) <= old_size else 0
                                
                invalid_cnt = 0
                
                # add new pipes to list
                for p,v in tpots[i].evaluated_individuals_.items():
                    if v['internal_cv_score'] == -np.inf: 
                        invalid_cnt += 1        
                    if 'source' not in v:
                        v['source'] = f'TPOT-BO-O{self.d_flag}'
                        v['structure'] = u.string_to_bracket(p)
                        v['generation'] = gen
                        v['delta'] = self.pop_size
                        self.strucs.add(p,v)
                        self.pipes[p] = v
                        n_extra_bo += 1
                        # tracking[gen][i] += 1

                if invalid_cnt > old_invalid_cnt:
                    self.vprint.v2(f"{u.RED}{invalid_cnt - old_invalid_cnt} invalid pipelines evaluated, continuing with same structure..\n{u.OFF}")
                
                old_invalid_cnt = invalid_cnt
                
        if out_path:
            with open(fname_pickle, 'wb') as f:
                # Pickle the 'data' dictionary using the highest protocol available.
                pickle.dump(self.pipes, f, pickle.HIGHEST_PROTOCOL)

            if not os.path.exists(out_path):
                os.makedirs(out_path)
            fname_h_pipes = os.path.join(out_path,f"TPOT-BO-O{self.d_flag}.pipes")
            # wipe existing pipe files if they exist
            with open(fname_h_pipes,'w') as f:
                for k,v in self.pipes.items():
                    f.write(f"{k};{v['structure']};{v['generation']};{v['delta']};{v['source']};{v['internal_cv_score']}\n")    
        
        self.vprint.v2(f"\n{u.CYAN}{n_extra_bo} initial evaluations performed so that all structures start with {self.n_0} samples..{u.OFF}\n")
        
        tracking = [[0 for k in self.bo_struc_keys] for _ in range(gen+1)]
        
        for p,v in self.pipes.items():
            if v['source'] != "TPOT-BASE":
                tracking[v['generation']][self.bo_struc_keys.index(v['structure'])] += 1
        
        
        # adjust number of bo evals to include extras
        self.n_bo_evals -= n_extra_bo
        
        # initialise m_strucs as 2*k as we halve at the start of each generation
        # m_strucs = [len(self.bo_struc_keys)]
        # m_strucs = [self.m_strucs]
        
        stagnate_cnt = 0
        
        max_evals = (self.starting_size + self.n_bo_evals)
        
        allocs = np.zeros(len(self.bo_struc_keys)) #!!
        
        while len(self.pipes) < max_evals:
            # m_strucs.append(int(np.ceil(m_strucs[-1]/2)))
            if Deltas[-1] > 1:
                tracking.append([0 for _ in self.bo_struc_keys])
                # m_strucs.append(self.m_strucs)
                Deltas.append(int(np.ceil(Deltas[-1]/2)))
                gen += 1
            B_r = max(max_evals - len(self.pipes),0)
            rem_halvings = int(np.ceil(np.log2(Deltas[-1])) + 1)
            B_g = int(((B_r / rem_halvings)//Deltas[-1]) * Deltas[-1]) if Deltas[-1] > 1 else 1
            
            if B_g <= 0:
                break
            
            n_evals = 0
            old_size = len(self.pipes)
            stagnate_cnt_gen = 0
            
            while n_evals < B_g:
                
                # # allocate self.Delta to top m_strucs
                # if Deltas[-1] > 1:
                    # compute statistics
                '''
                # means = np.array([self.strucs[k].mu for k in self.bo_struc_keys])
                # ses = np.array([self.strucs[k].stderr for k in self.bo_struc_keys])
                '''
                
                # get mu sigma and max allocs
                mu = -1 * np.array([self.strucs[s].mu for s in self.bo_struc_keys])
                sigma = np.array([self.strucs[s].std for s in self.bo_struc_keys])
                
                # deal with zero sigma issues
                # min_sigma = min(sigma[sigma > EPS]) - EPS if np.sum(sigma) > 0 else EPS
                # zero_ids = sigma < EPS
                # sigma[zero_ids] = min_sigma
                sigma[sigma < EPS] = EPS
                sigma[sigma > MAX_SIGMA] = MAX_SIGMA
                print(f"{u.CYAN}[{time.asctime()}]{u.OFF} - seed: {self.seed}, generation: {gen}")
                
                t_start_alloc = time.time()
                
                # get allocations
                # allocs = o.get_allocations(mu,sigma,Deltas[-1])!!

                # n_allocs = np.sum(allocs > 0)!!
                                
                old_allocs = allocs                #!!
                
                delta = min(Deltas[-1], B_g-n_evals) #!!
                
                # get allocations
                allocs = o.get_allocations(mu,sigma,delta,min_allocs=old_allocs)#!!
                
                new_allocs = allocs - old_allocs#!!
                
                # print(old_allocs)
                # print(allocs)
                print(new_allocs)
                
                n_allocs = np.sum(new_allocs > 0)#!!
                
                old_size_gen = len(self.pipes)
                
                # perform BO evaluations as per allocations
                # for i,alloc in enumerate(allocs): !!
                for i,alloc in enumerate(new_allocs):
                    # if no allocation, continue
                    if alloc <= 0:
                        continue
                    
                    struc = self.strucs[self.bo_struc_keys[i]]
                    
                    self.vprint.v2(f"\n{u.CYAN}[Seed {self.seed}] {len(self.pipes)-self.starting_size+n_extra_bo} total evaluations of {self.n_bo_evals+n_extra_bo} performed, {B_r-n_evals} remaining{u.OFF}")
                    self.vprint.v2(f"{u.CYAN}{n_evals} evaluations of {B_g} performed, with {B_g-n_evals} remaning in generation {gen} (Delta = {Deltas[-1]})\nPerforming {alloc} evaluations on structure ({n_allocs} allocs):\n{struc}..{u.OFF}\n")
                    self.vprint.v2(f"{u.CYAN}generation stagnate count: {stagnate_cnt_gen}{u.OFF}\n")
                                        
                    # run bayesian optimisation with seed_dicts as initial samples
                    handlers[i].optimise(0, X_train, y_train, n_evals=alloc,
                                    seed_samples=struc.get_seed_samples(), 
                                    discrete_mode=self.discrete_mode,
                                    skip_params=[],
                                    timeout_trials=self.optuna_timeout_trials)
                    
                    if out_path:
                        f = open(fname_h_pipes,'a')
                    
                    # add new pipes to list
                    for p,v in tpots[i].evaluated_individuals_.items():
                        if 'source' not in v:
                            v['structure'] = u.string_to_bracket(p)
                            v['source'] = f'TPOT-BO-O{self.d_flag}'
                            v['generation'] = gen
                            v['delta'] = Deltas[-1]
                            self.strucs.add(p,v)
                            self.pipes[p] = v
                            if out_path:
                                f.write(f"{p};{v['structure']};{v['generation']};{v['delta']};{v['source']};{v['internal_cv_score']}\n")
                            n_evals += 1 
                            tracking[gen][i] += 1
            
                    if out_path:
                        f.close()              
                
                if out_path:
                    fname_o_track = os.path.join(out_path,f"TPOT-BO-O{self.d_flag}.tracking")
                    with open(fname_o_track,'w') as f:
                        f.write(u.disp_ocba_tracking(tracking,Deltas,colours=False))
                
                stagnate_cnt_gen = stagnate_cnt_gen + 1 if len(self.pipes) == old_size_gen else 0
                if stagnate_cnt_gen >= 50:
                    print("50 OCBA iterations without change, exiting generation..")
                    break                
                
            stagnate_cnt = stagnate_cnt + 1 if len(self.pipes) == old_size else 0
            
            if stagnate_cnt >= 10:
                print("10 OCBA iterations without change, exiting iteration..")
                early_finish = f" - early finish (gen {gen})"
                break   
            
            if out_path:
                with open(fname_pickle, 'wb') as f:
                    # Pickle the 'data' dictionary using the highest protocol available.
                    pickle.dump(self.pipes, f, pickle.HIGHEST_PROTOCOL)             
        
        t_end = time.time()
        
        # delete pickle and log files if they exist
        if out_path:
            if os.path.exists(fname_pickle):
                os.remove(fname_pickle)
        
        best_tpot_pipe, best_tpot_cv = u.get_best(self.pipes, source='TPOT-BASE')
        best_bo_pipe, best_bo_cv = u.get_best(self.pipes, source=f'TPOT-BO-O{self.d_flag}')
        
        self.vprint.v1(f"\n{u.YELLOW}* best pipe found by tpot:{u.OFF}")
        self.vprint.v1(f"{best_tpot_pipe}")
        self.vprint.v1(f"{u.GREEN} * score:{u.OFF} {best_tpot_cv}")
        self.vprint.v1(f"\n{u.YELLOW}best pipe found by BO:{u.OFF}")
        self.vprint.v1(f"{best_bo_pipe}\n{u.GREEN} * score:{u.OFF} {best_bo_cv}")
        self.vprint.v1(f"\nTotal time elapsed: {round(t_end-t_start,2)} sec\n")
                    
        return f"Successful{early_finish}"
                    