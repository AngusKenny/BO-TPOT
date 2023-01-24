#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 18:14:11 2022

@author: gus
"""

from config.tpot_config import default_tpot_config_dict
from tpot import TPOTRegressor
from deap import creator
import utils.tpot_utils as u
import utils.ocba as o
from utils.data_structures import StructureCollection
import copy
import os
import time
import numpy as np

EPS = 1e-10

class oTPOT_Base(object):    
    def __init__(self,
                 start_gen=2,
                 n_gens=100,
                 pop_size=100,
                 seed=42,
                 config_dict=default_tpot_config_dict,
                 n_jobs=-1,
                 pipe_eval_timeout=5,
                 vprint=u.Vprint(1)):
        
        self.start_gen=start_gen
        self.pipes = {}
        self.n_gens=n_gens
        self.pop_size=pop_size
        self.vprint=vprint
        self.seed=seed
        self.config_dict = copy.deepcopy(config_dict)
        self.n_jobs = n_jobs
                
        # set tpot verbosity to vprint.verbosity + 1 to give more information
        tpot_verb = vprint.verbosity + 1 if vprint.verbosity > 0 else 0
        
        # create TPOT object and fit for tot_gens generations
        self.tpot = TPOTRegressor(generations=1,
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

        log_file = os.path.join(out_path,'oTPOT-BASE.log')       
        time_file = os.path.join(out_path,'oTPOT-BASE.times')       
        
        if out_path:
            self.tpot.log_file = log_file
            f = open(time_file,'w')
            f.close()
        
        fname_tracker = os.path.join(out_path,'oTPOT-BASE.tracker')
        
        self.vprint.v2(f"{u.CYAN}fitting tpot model with {self.tpot.generations}" 
               + " generations (-1 to account for initial evaluations)" 
               + f"..\n{u.WHITE}")
        
        t_tpot_start = time.time()
        # fit TPOT model
        self.tpot.fit(X_train, y_train)
        t_tpot_end = time.time()
        
        # instantiate structure collection object
        strucs = StructureCollection(config_dict=self.config_dict)
        
        print(f"len strucs before update: {len(strucs)}")
        
        # import structures from tpot dictionary
        strucs.update(self.tpot.evaluated_individuals_)
        
        pop_tracker = {0: {}, 1: {}}
        
        for p,v in self.tpot.evaluated_individuals_.items():
            if v['internal_cv_score'] == -np.inf: continue
            if v['structure'] not in pop_tracker[v['generation']]:
                pop_tracker[v['generation']][v['structure']] = 1
            else:
                pop_tracker[v['generation']][v['structure']] = pop_tracker[v['generation']][v['structure']] + 1
        
        if (out_path):
            with open(fname_tracker, 'w') as f:
                for g in pop_tracker:
                    for s in pop_tracker[g]:
                        f.write(f"{g};{s};{pop_tracker[g][s]};{len(strucs[s].operators)}\n")
            with open(time_file,'w') as f:
                f.write(f"{0};{0}\n")
                f.write(f"{1};{t_tpot_end-t_tpot_start}\n")
        
        
        
        print(f"len strucs after update: {len(strucs)}")
        
        # copy evaluated individuals dictionary
        self.pipes = copy.deepcopy(self.tpot.evaluated_individuals_)
               
        for gen in range(self.start_gen,self.n_gens):
            # get mu sigma and max allocs
            mu = -1 * np.array([strucs[s].mu for s in strucs.keys()])
            max_allocs = np.array([len(strucs[s]) for s in strucs.keys()])
            sigma = np.array([strucs[s].std for s in strucs.keys()])
            
            # deal with zero sigma issues
            min_sigma = min(sigma[sigma > EPS]) - EPS if np.sum(sigma) > 0 else EPS
            zero_ids = sigma < EPS
            sigma[zero_ids] = min_sigma
            sigma[sigma > 1e10] = 1e10
            print(f"{u.CYAN}[{time.asctime()}]{u.OFF} - generation: {gen}")
            # print(f"min_sigma: {min_sigma}")
            # print(f"mu: {mu}")
            # print(f"sigma: {sigma}")
            # print(f"max: {max_allocs}")
            
            t_start_alloc = time.time()
            
            # get allocations
            allocs = o.get_allocations(mu,sigma,self.pop_size,max_allocs=max_allocs)   
            # allocs = o.get_allocations(mu,sigma,self.pop_size)           
            
            t_end_alloc = time.time()
            
            # print(f"allocs: {allocs}")
            
            print(f"allocs completed in {t_end_alloc - t_start_alloc} seconds")
            
            # create TPOT population for next generation
            self.tpot._pop = []
            
            pop_tracker[gen] = {}
            
            print("\nCONSTRUCTING POPULATION:\n")
            
            for i,n in enumerate(allocs):
                if n == 0: continue
                struc = strucs.get_by_index(i)
                print(f"{u.RED}{i}{u.OFF}: {u.CYAN}{struc.structure}{u.OFF}: {n} pipes, {len(struc.operators)} operators")
                add_pipes = struc.get_best(n)
                
                pop_tracker[gen][struc.structure] = n
                    
                for j,p in enumerate(add_pipes):
                    print(f"{u.YELLOW}{j}{u.OFF}: {p}, cv: {self.tpot.evaluated_individuals_[p]['internal_cv_score']}")
                    self.tpot._pop.append(creator.Individual.from_string(p, self.tpot._pset))
                print("")
            
            print(f"\n{u.RED}Population of {len(self.tpot._pop)} individuals created{u.OFF}\n")
            
            if (out_path):
                with open(fname_tracker, 'a') as f:
                    for s in pop_tracker[gen]:
                        f.write(f"{gen};{s};{pop_tracker[gen][s]};{len(strucs[s].operators)}\n")
                print(f"\n{u.CYAN}[{time.asctime()}]{u.OFF} - {u.YELLOW}Fitting TPOT model for gen {gen}, seed {self.seed} - see {log_file} for progress{u.OFF}")
        
            # pop_tracker[gen] = {}
            
            # for p in self.tpot._pop:                
            #     s = u.string_to_bracket(str(p))
            #     if s not in pop_tracker[gen]:
            #         pop_tracker[gen][s] = 1
            #     else:
            #         pop_tracker[gen][s] = pop_tracker[gen][s] + 1
            
            t_tpot_start = time.time()
            # fit TPOT model
            self.tpot.fit(X_train, y_train)
            t_tpot_end = time.time()
            
            if out_path:
                with open(time_file,'a') as f:
                    f.write(f"{gen};{t_tpot_end-t_tpot_start}\n")
            
            print(f"{u.RED}TPOT took {t_tpot_end-t_tpot_start} seconds{u.OFF}")
            
            for k,v in self.tpot.evaluated_individuals_.items():
                if k not in self.pipes:
                    self.pipes[k] = v
                    self.pipes[k]['generation'] = gen
                    self.pipes[k]['source'] = 'oTPOT-BASE'
                    
            # update structures
            strucs.update(self.tpot.evaluated_individuals_)
        
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
            fname_tpot_pipes = os.path.join(out_path,'oTPOT-BASE.pipes')
            fname_tracker = os.path.join(out_path,'oTPOT-BASE.tracker')
            print(fname_tpot_pipes)
            # write all evaluated pipes
            with open(fname_tpot_pipes, 'w') as f:
                for k,v in self.pipes.items():
                    f.write(f"{k};{v['generation']};{v['internal_cv_score']}\n")
            # with open(fname_tracker, 'w') as f:
            #     for g in pop_tracker:
            #         for s in pop_tracker[g]:
            #             f.write(f"{g};{s};{pop_tracker[g][s]}\n")
                    
        return "Successful"