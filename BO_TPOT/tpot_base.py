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
from utils.data_structures import StructureCollection
import numpy as np
import pickle

class TPOT_Base(object):    
    def __init__(self,
                 n_gens=100,
                 pop_size=100,
                 seed=42,
                 config_dict=default_tpot_config_dict,
                 n_jobs=-1,
                 pipe_eval_timeout=5,
                 allow_restart=True,
                 vprint=u.Vprint(1)):
        
        self.allow_restart = allow_restart
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
        
        self.tpot._fit_init()
        
    def optimize(self, X_train, y_train, out_path=None):
        t_start = time.time()
        
        log_file = None
        
        if out_path:
            log_file = os.path.join(out_path,'TPOT-BASE.log')
            # time_file = os.path.join(out_path,'TPOT-BASE.times')       
            fname_pickle = os.path.join(out_path,'TPOT-BASE.pickle')
            fname_tracker = os.path.join(out_path,'TPOT-BASE.tracker')
            self.tpot.log_file = log_file
        
        self.start_gen = 2
        
        if out_path and self.allow_restart and os.path.exists(fname_pickle):
            with open(fname_pickle, 'rb') as f:
                self.tpot.evaluated_individuals_ = pickle.load(f)
            self.start_gen = max([v['generation'] for v in self.tpot.evaluated_individuals_.values()]) + 1
            print(f"{u.RED}Loaded {self.start_gen-1} generations from previous interrupted run.. continuing..{u.OFF}")
        else:
            self.vprint.v2(f"{u.CYAN}fitting tpot model with {self.tpot.generations}" 
               + " generations (-1 to account for initial evaluations)" 
               + f"..\n{u.WHITE}")
            t_tpot_start = time.time()
            # fit TPOT model
            self.tpot.fit(X_train, y_train)
            t_tpot_end = time.time()
            # if out_path:
                # with open(time_file,'w') as f:
                #     f.write(f"{0};{0}\n")
                #     f.write(f"{1};{t_tpot_end-t_tpot_start}\n")
            if out_path and self.allow_restart:
                with open(fname_pickle, 'wb') as f:
                    # Pickle the 'data' dictionary using the highest protocol available.
                    pickle.dump(self.tpot.evaluated_individuals_, f, pickle.HIGHEST_PROTOCOL)
        
        # instantiate structure collection object
        strucs = StructureCollection(config_dict=self.config_dict)
        
        print(f"len strucs before update: {len(strucs)}")
        
        # import structures from tpot dictionary
        strucs.update(self.tpot.evaluated_individuals_)
                
        pop_tracker = {0: {}, 1: {}}
        
        for p,v in self.tpot.evaluated_individuals_.items():
            # if v['internal_cv_score'] == -np.inf: continue
            if v['generation'] not in pop_tracker:
                pop_tracker[v['generation']] = {}
            if v['structure'] not in pop_tracker[v['generation']]:
                pop_tracker[v['generation']][v['structure']] = 1
            else:
                pop_tracker[v['generation']][v['structure']] = pop_tracker[v['generation']][v['structure']] + 1
        
        if (out_path):
            with open(fname_tracker, 'w') as f:
                for g in pop_tracker:
                    for s in pop_tracker[g]:
                        f.write(f"{g};{s};{pop_tracker[g][s]};{strucs[s].cv}\n")
            # with open(time_file,'w') as f:
            #     f.write(f"{0};{0}\n")
            #     f.write(f"{1};{t_tpot_end-t_tpot_start}\n")    
        
        print(f"len strucs after update: {len(strucs)}")
        
        # copy evaluated individuals dictionary
        self.pipes = copy.deepcopy(self.tpot.evaluated_individuals_)

        for gen in range(self.start_gen,self.n_gens):            
            # create TPOT population for next generation           
            pop_tracker[gen] = {}
                        
            for i,p in enumerate(self.tpot._pop):
                struc = u.string_to_bracket(str(p))
                if struc not in pop_tracker[gen]:
                    pop_tracker[gen][struc] = 0
                else:
                    pop_tracker[gen][struc] = pop_tracker[gen][struc] + 1
                
            if (out_path):
                with open(fname_tracker, 'a') as f:
                    for s in pop_tracker[gen]:
                        f.write(f"{gen};{s};{pop_tracker[gen][s]};{strucs[s].cv}\n")
                        
            print(f"\n{u.CYAN}[{time.asctime()}]{u.OFF} - {u.YELLOW}Fitting TPOT model for gen {gen}, seed {self.seed}{u.OFF}")
            
            t_tpot_start = time.time()
            # fit TPOT model
            self.tpot.fit(X_train, y_train)
            t_tpot_end = time.time()
            
            # if out_path:
            #     with open(time_file,'a') as f:
            #         f.write(f"{gen};{t_tpot_end-t_tpot_start}\n")
            
            print(f"{u.RED}TPOT took {t_tpot_end-t_tpot_start} seconds{u.OFF}")
            
            for k,v in self.tpot.evaluated_individuals_.items():
                if k not in self.pipes:
                    self.pipes[k] = v
                    self.pipes[k]['generation'] = gen
                    self.pipes[k]['source'] = 'TPOT-BASE'
                    
            # update structures
            strucs.update(self.tpot.evaluated_individuals_)
            
            if out_path and self.allow_restart:
                with open(fname_pickle, 'wb') as f:
                    # Pickle the 'data' dictionary using the highest protocol available.
                    pickle.dump(self.tpot.evaluated_individuals_, f, pickle.HIGHEST_PROTOCOL)
               
        t_end = time.time()
        
        best_tpot_pipe, best_tpot_cv = u.get_best(self.pipes)
        
        self.vprint.v1(f"\n{u.YELLOW}* best pipe found by TPOT:{u.OFF}")
        self.vprint.v1(f"{best_tpot_pipe}")
        self.vprint.v1(f"{u.GREEN} * score:{u.OFF} {best_tpot_cv}")
        self.vprint.v1(f"\nTotal time elapsed: {round(t_end-t_start,2)} sec\n")
                
        # if out_path exists then write pipes to file
        if out_path:
            # delete pickle and log files if they exist
            if os.path.exists(fname_pickle):
                os.remove(fname_pickle)
            if os.path.exists(log_file):
                self.tpot.log_file_.close()
                os.remove(log_file)
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