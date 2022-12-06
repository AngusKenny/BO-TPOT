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

class TPOT_BO_H(object):    
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
        
        # set tpot verbosity to vprint.verbosity + 1 to give more information
        self.tpot_verb = vprint.verbosity + 1 if vprint.verbosity > 0 else 0
        
        for k,v in self.tpot_pipes.items():
            v['source'] = f'TPOT-BASE'
        
        # get unique structures
        self.strucs = u.get_structures(self.tpot_pipes, config_dict=self.config_dict)
        # u_grps = u.get_unique_groups(copy.deepcopy(self.tpot_pipes), config_dict=self.config_dict)
                        
        self.bo_struc_keys = u.get_best_structures(self.strucs, size=int(pop_size*bo_pop_factor))
        
        vprint.v2(f"\n{u.CYAN}{len(self.bo_struc_keys)} groups in BO set, populating TPOT evaluated dictionary..{u.OFF}\n")

        # self.tpot = TPOTRegressor(generations=0,
        #                           population_size=1, 
        #                           mutation_rate=0.9, 
        #                           crossover_rate=0.1, 
        #                           cv=5,
        #                           verbosity=self.tpot_verb, 
        #                           config_dict=copy.deepcopy(self.config_dict),
        #                           random_state=self.seed, 
        #                           n_jobs=1,
        #                           warm_start=True,
        #                           max_eval_time_mins=self.pipe_eval_timeout)            
            
        #     # initialise tpot object to generate pset
        # self.tpot._fit_init()
        
        # self.tpot.evaluated_individuals_ = {}
        
        # create TPOT object for each pipe in set and fit for 0 generations
        for i,k in enumerate(self.bo_struc_keys):
            # print(f"{i}:{k}")
            # for p,v in self.strucs[k].pipes.items():
                # print(f"\t{i}:{len(self.strucs[k].pipes)}:{u.string_to_bracket(p)}:{v['structure']}")
            #     if u.string_to_bracket(p) != k:
                    # print(f"{u.string_to_bracket(p)} not equal to {k}")
            # add to pipes and TPOT evaluated dictionary
            self.pipes.update(self.strucs[k].pipes)
            # self.tpot.evaluated_individuals_.update(copy.deepcopy(v['matching']))

        self.starting_size = len(self.pipes)
        
    def optimize(self, X_train, y_train, out_path=None):
        if out_path:
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            fname_h_pipes = os.path.join(out_path,"TPOT-BO-H.pipes")
            # wipe existing pipe files if they exist
            with open(fname_h_pipes,'w') as f:
                for k,v in self.pipes.items():
                    f.write(f"{k};{v['structure']};0;{len(self.bo_struc_keys)};{v['source']};{v['internal_cv_score']}\n")    
            
        t_start = time.time()
        
        self.vprint.v1("")
        gen = 1
        
        early_finish = ""

        stagnate_cnt = 0
        
        while len(self.pipes) < (self.starting_size + self.n_bo_evals):
            old_size = len(self.pipes)
            n_strucs = len(self.bo_struc_keys)
            rem_evals = self.starting_size + self.n_bo_evals - len(self.pipes)
            rem_halvings = int(np.ceil(np.log2(n_strucs)) + 1)
            gen_evals = rem_evals / rem_halvings
            
            tot_params = np.sum([len(self.strucs[k].bo_params) for k in self.bo_struc_keys])
            
            # compute how many trials
            self.vprint.v2(f"\n{u.CYAN}TPOT-BO-H generation {gen}, {len(self.bo_struc_keys)} structures in the BO set..{u.OFF}\n")
            
            for i,k in enumerate(self.bo_struc_keys):
                n_hp = len(self.strucs[k].bo_params)
                struc_data = self.strucs[k]                
                n_grp_trials = max(1,int((n_hp/tot_params)*gen_evals))
                
                seed_samples = [(u.string_to_params(k2), v2['internal_cv_score']) for k2,v2 in struc_data.pipes.items()]
                
                self.vprint.v2(f"\n{u.CYAN}Generation {gen}, structure group {i+1} of {n_strucs} - {len(seed_samples)} seed samples generated with {n_hp} BO hyper-parameters, optimizing for a further {n_grp_trials} evaluations..{u.OFF}\n")
                
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
                handler.optimise(0, X_train, y_train, n_evals=n_grp_trials,
                                 seed_samples=seed_samples, 
                                 discrete_mode=self.discrete_mode,
                                 skip_params=[],
                                 timeout_trials=self.optuna_timeout_trials)
                
                if out_path:
                    f = open(fname_h_pipes,'a')
                
                # update matching and recorded
                for k2,v2 in tpot.evaluated_individuals_.items():
                    if 'source' not in v2:
                        v2['source'] = f'TPOT-BO-H{self.d_flag}'
                    if k2 not in self.strucs:
                        self.strucs.add(k2,v2)
                        # v['matching'][k2] = copy.deepcopy(v2)
                    
                    if k2 not in self.pipes:
                        # self.pipes[k2] = copy.deepcopy(v2)
                        self.pipes[k2] = copy.deepcopy(v2)
                        if out_path:
                            f.write(f"{k2};{v2['structure']};{gen};{n_strucs};{v2['source']};{v2['internal_cv_score']}\n")    
                            # f.write(f"{k2};{gen};{len(self.bo_set)};{v2['source']};{v2['internal_cv_score']}\n")
                            
                if out_path:
                    f.close()
                
                # # update group statistics
                # self.bo_set[k] = u.update_group(v)
                
            # do halving and update BO set
            if len(self.bo_struc_keys) > 1:
                # self.bo_struc_keys = u.get_best(self.bo_set, size=int(np.ceil(len(self.bo_set)/2)))
                self.bo_struc_keys = u.get_best_structures(self.strucs, size=int(np.ceil(n_strucs/2)))
                
            stagnate_cnt = stagnate_cnt + 1 if len(self.pipes) == old_size else 0
            
            if stagnate_cnt > 9:
                print("10 generations without change, exiting..")
                early_finish = f" - early finish (gen {gen})"
                break
            
            gen = gen+1
                
        
        t_end = time.time()
        
        best_tpot_pipe, best_tpot_cv = u.get_best(self.pipes, source='TPOT-BASE')
        best_bo_pipe, best_bo_cv = u.get_best(self.pipes, source=f'TPOT-BO-H{self.d_flag}')
        
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
                    
        return f"Successful{early_finish}"
                    
    
class TPOT_BO_Hs(object):    
    def __init__(self,  
                 tbh_pipes,
                 seed=42,
                 pop_size=100,
                 bo_pop_factor=0.5,
                 n_bo_evals=2000,
                 optuna_timeout_trials=100,
                 config_dict=default_tpot_config_dict,
                 pipe_eval_timeout=5,
                 vprint=u.Vprint(1)):
        
        self.pipes = {}
        self.pop_size = pop_size
        self.n_bo_evals=n_bo_evals
        self.tbh_pipes=copy.deepcopy(tbh_pipes)
        self.config_dict=copy.deepcopy(config_dict)
        self.optuna_timeout_trials=optuna_timeout_trials
        self.seed=seed
        self.pipe_eval_timeout=pipe_eval_timeout
        self.vprint=vprint
        
        # set tpot verbosity to vprint.verbosity + 1 to give more information
        self.tpot_verb = vprint.verbosity + 1 if vprint.verbosity > 0 else 0
        
        # get unique structures
        # u_grps = u.get_unique_groups(copy.deepcopy(self.tbh_pipes), config_dict=self.config_dict)
        self.strucs = u.get_structures(self.tbh_pipes, config_dict=self.config_dict)
        
        self.bo_struc_keys = u.get_best_structures(self.strucs, size=1)
        
        # self.bo_set = u.get_best(u_grps, size=1)
                
        struct_str = list(self.bo_struc_keys)[0]
        
        vprint.v2(f"\n{u.CYAN}best structure from TPOT-BO-Hd:{u.OFF}\n{struct_str}\n")

        # self.tpot = TPOTRegressor(generations=0,
        #                           population_size=1, 
        #                           mutation_rate=0.9, 
        #                           crossover_rate=0.1, 
        #                           cv=5,
        #                           verbosity=self.tpot_verb, 
        #                           config_dict=copy.deepcopy(self.config_dict),
        #                           random_state=self.seed, 
        #                           n_jobs=1,
        #                           warm_start=True,
        #                           max_eval_time_mins=self.pipe_eval_timeout)            
            
        #     # initialise tpot object to generate pset
        # self.tpot._fit_init()
        
        self.pipes = self.strucs[struct_str].pipes
        
        # self.tpot.evaluated_individuals_ = copy.deepcopy(self.pipes)
        
        # # create TPOT object for each pipe in set and fit for 0 generations
        # for k,v in self.bo_set.items():
        #     # add to pipes and TPOT evaluated dictionary
        #     self.pipes.update(copy.deepcopy(v['matching']))
        #     # self.tpot.evaluated_individuals_.update(copy.deepcopy(v['matching']))

        self.starting_size = len(self.pipes)
        self.starting_gen = np.max([v['generation'] for v in self.tbh_pipes.values()]) + 1
        
    def optimize(self, X_train, y_train, out_path=None):
        if out_path:
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            fname_hs_pipes = os.path.join(out_path,"TPOT-BO-Hs.pipes")
            # wipe pipe existing pipe files if they exist
            with open(fname_hs_pipes,'w') as f:
                for k,v in self.tbh_pipes.items():
                    f.write(f"{k};{v['structure']};{v['generation']};{v['n_bo_pop']};{v['source']};{v['internal_cv_score']}\n")    
            
        t_start = time.time()
        
        self.vprint.v1("")
        
        early_finish = ""
        
            
        for i,k in enumerate(self.bo_struc_keys):
            n_hp = len(self.strucs[k].bo_params)
            struc_data = self.strucs[k]                
            seed_samples = [(u.string_to_params(k2), v2['internal_cv_score']) for k2,v2 in struc_data.pipes.items()]
            
            self.vprint.v2(f"\n{u.CYAN}Generation {self.starting_gen} - {len(seed_samples)} seed samples generated with {n_hp} BO hyper-parameters, optimizing for a further {self.n_bo_evals} evaluations..{u.OFF}\n")
            
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
            
            # initialise tpot bo handler
            handler = TPOT_BO_Handler(tpot, vprint=self.vprint, discrete_mode=False)
            new_params = u.string_to_params(struc_data.best)
            
            # update pset of BO tpot object
            for (p,val) in new_params:
                handler.add_param_to_pset(p, val)
            
            # remove generated pipeline and transplant saved from before
            tpot._pop = [creator.Individual.from_string(struc_data.best, tpot._pset)]
            
            tpot.fit(X_train, y_train)
            
            # initialise tpot bo handler
            handler = TPOT_BO_Handler(tpot, vprint=self.vprint, discrete_mode=False)
            
            # run bayesian optimisation with seed_dicts as initial samples
            handler.optimise(0, X_train, y_train, n_evals=self.n_bo_evals,
                             seed_samples=seed_samples, 
                             discrete_mode=False,
                             skip_params=[],
                             timeout_trials=self.optuna_timeout_trials)
            
            if out_path:
                f = open(fname_hs_pipes,'a')
            
            # update matching and recorded
            for k2,v2 in tpot.evaluated_individuals_.items():
                if 'source' not in v2:
                    v2['source'] = f'TPOT-BO-Hs'
                if k2 not in self.strucs:
                        self.strucs.add(k2,v2)
                # if k2 not in v['matching']:
                #     v['matching'][k2] = copy.deepcopy(v2)
                
                if k2 not in self.pipes:
                    v2['source'] = f'TPOT-BO-Hs({k})'
                    self.pipes[k2] = copy.deepcopy(v2)
                    if out_path:
                        f.write(f"{k2};{v2['structure']};{self.starting_gen};1;{v2['source']};{v2['internal_cv_score']}\n")
                        
            if out_path:
                f.close()
            
            # update group statistics
            # self.bo_set[k] = u.update_group(v)
                
            # # do halving and update BO set
            # if len(self.bo_set) > 1:
            #     self.bo_set = u.get_best(self.bo_set, size=int(np.ceil(len(self.bo_set)/2)))
            
            # stagnate_cnt = stagnate_cnt + 1 if len(self.pipes) == old_size else 0
            
            # if stagnate_cnt > 9:
            #     print("10 generations without change, exiting..")
            #     early_finish = f" - early finish (gen {gen})"
            #     break
            
            # gen = gen+1
                
        
        t_end = time.time()
                
        best_tpot_pipe, best_tpot_cv = u.get_best(self.pipes, source='TPOT-BASE')
        best_tbh_pipe, best_tbh_cv = u.get_best(self.pipes, source='TPOT-BO-Hd')
        best_tbhs_pipe, best_tbhs_cv = u.get_best(self.pipes, source='TPOT-BO-Hs')
        
        self.vprint.v1(f"\n{u.YELLOW}* best pipe found by tpot:{u.OFF}")
        self.vprint.v1(f"{best_tpot_pipe}")
        self.vprint.v1(f"{u.GREEN} * score:{u.OFF} {best_tpot_cv}")
        self.vprint.v1(f"\n{u.YELLOW}best pipe found by BO-Hd at {2000-self.n_bo_evals} evals:{u.OFF}")
        self.vprint.v1(f"{best_tbh_pipe}\n{u.GREEN} * score:{u.OFF} {best_tbh_cv}")
        self.vprint.v1(f"\n{u.YELLOW}best pipe found by BO-Hs:{u.OFF}")
        self.vprint.v1(f"{best_tbhs_pipe}\n{u.GREEN} * score:{u.OFF} {best_tbhs_cv}")
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
                    
        return f"Successful{early_finish}"