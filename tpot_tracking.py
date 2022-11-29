#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tracking a run of TPOT:

file - <problem>_<seed>.tracking:
---------------------------------
generation;n_archive;n_structures;n_nd_params;n_nd_unique_params;n_nd_unique_structures_params;n_nd_ops;n_nd_unique_ops;n_nd_unique_structures_ops

file - <problem>_<seed>.gen_<generation>
----------------------------------------
pipe_string;structure_string;cv;n_ops;n_hps;parent/child

NOTES:
add unique ND
add ND n_ops as well
parent = 0, child = 1
"""
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting, find_non_dominated
from config.tpot_config import default_tpot_config_dict
from tpot import TPOTRegressor
from utils.data_structures import StructureCollection
import utils.tpot_utils as u
import copy
import os
import time
import numpy as np

POP_SIZE = 100
nGENS = 100
OUT_PATH = "Results_test"
PROBLEMS = ["quake"]
DATA_DIR = "Data"
SEED = 45
nRUNS = 1

cwd = os.getcwd()

for problem in PROBLEMS:

    fname = problem + ".data"
    fpath = os.path.join(cwd,DATA_DIR, fname)
    X_train, X_test, y_train, y_test = u.load_data(fpath)
    
    for seed_inc in range(nRUNS):
        seed = SEED + seed_inc
        track_path = os.path.join(cwd,OUT_PATH,"Tracking3",f"{problem}_{seed}_pop_{POP_SIZE}")

        if not os.path.exists(track_path):
            os.makedirs(track_path)

        f_track = os.path.join(track_path,f"{problem}_{seed}_pop_{POP_SIZE}.tracking")

        with open(f_track, 'w') as f:
            f.write("generation;n_archive;n_structures;n_nd_params;n_nd_unique_params;n_nd_unique_structures_params;n_nd_ops;n_nd_unique_ops;n_nd_unique_structures_ops\n")

        # initialise tpot object
        tpot = TPOTRegressor(generations=0,
                            population_size=POP_SIZE, 
                            mutation_rate=0.9, 
                            crossover_rate=0.1, 
                            cv=5,
                            verbosity=3,
                            config_dict = default_tpot_config_dict, 
                            random_state=seed, 
                            n_jobs=-1,
                            warm_start=True,
                            max_eval_time_mins=5)

        tpot._fit_init()

        grps = {}
        g_keys = []

        strucs = StructureCollection(config_dict=default_tpot_config_dict)

        for gen in range(nGENS):
            print(f"\n{u.CYAN}({time.strftime('%d %b, %H:%M', time.localtime())}) Generation {gen} of {nGENS}, archive size: {len(tpot.evaluated_individuals_)}{u.OFF}\n")
            gen_str = f"0{gen}" if gen < 10 else f"{gen}"
            f_gen = os.path.join(track_path,f"{problem}_{seed}_pop_{POP_SIZE}.gen_{gen_str}")
            with open(f_gen, 'w') as f:
                # store parent population
                for ind in tpot._pop:
                    ind_str = str(ind)
                    grp_str = u.string_to_bracket(ind_str)
                    n_hps = len(u.string_to_params(ind_str,config_dict=default_tpot_config_dict))
                    f.write(f"{list(tpot.evaluated_individuals_.keys()).index(ind_str)};{strucs.index(grp_str)};{tpot.evaluated_individuals_[ind_str]['internal_cv_score']};{tpot.evaluated_individuals_[ind_str]['operator_count']};{n_hps};0\n")
            # fit tpot object
            tpot.fit(X_train,y_train)
            
            tpot.generations=1
            
            n_new_pipes = 0
            n_new_groups = 0
            
            # with open(f_gen, 'a') as f:    
            for i,(p,v) in enumerate(tpot.evaluated_individuals_.items()):
                if strucs.has_pipe(p):
                    continue
                
                n_new_pipes = n_new_pipes + 1
                
                
                # v['structure'] = u.string_to_bracket(p)
                v['generation'] = gen
                n_new_groups = n_new_groups + strucs.add(p,v)
                n_hps = len(u.string_to_params(p,config_dict=default_tpot_config_dict))
                
                # # if group already exists, add to existing group
                # if v['structure'] in grps:
                #     grps[v['structure']]['matching'][p] = copy.deepcopy(v)
                #     grps[v['structure']] = u.update_group(grps[v['structure']])
                # else:
                #     n_new_groups = n_new_groups + 1
                #     # create new group
                #     grps[v['structure']] = u.make_new_group(p,v,default_tpot_config_dict)
                                        
                #     # add to g_keys
                #     g_keys.append(v['structure'])
                
                # g_id = g_keys.index(v['structure'])
                # # # write children
                # # f.write(f"{i};{g_id};{v['internal_cv_score']};{v['operator_count']};{n_hps};1\n")
            
            with open(f_gen, 'a') as f:
                # store parent population
                for ind in tpot._pop:
                    ind_str = str(ind)
                    grp_str = u.string_to_bracket(ind_str)
                    n_hps = len(u.string_to_params(ind_str,config_dict=default_tpot_config_dict))
                    f.write(f"{list(tpot.evaluated_individuals_.keys()).index(ind_str)};{strucs.index(grp_str)};{tpot.evaluated_individuals_[ind_str]['internal_cv_score']};{tpot.evaluated_individuals_[ind_str]['operator_count']};{n_hps};1\n")
            
            # get points for ND sort by cv/params
            p_params = np.hstack((np.array([-v['internal_cv_score'] for v in tpot.evaluated_individuals_.values()]).reshape(-1,1), 
                                        np.array([len(strucs[v['structure']].bo_params) for v in tpot.evaluated_individuals_.values()]).reshape(-1,1)))
            
            # get points for ND sort by cv/ops
            p_ops = np.hstack((np.array([-v['internal_cv_score'] for v in tpot.evaluated_individuals_.values()]).reshape(-1,1), 
                                        np.array([v['operator_count'] for v in tpot.evaluated_individuals_.values()]).reshape(-1,1)))
            
            # get unique points
            u_params,u_params_ids = np.unique(p_params,return_index=True,axis=0)
            u_ops,u_ops_ids = np.unique(p_ops,return_index=True,axis=0)
            
            # do non-dominated sorts
            ndf_params = find_non_dominated(p_params)
            u_ndf_params = find_non_dominated(u_params)
            ndf_ops = find_non_dominated(p_ops)
            u_ndf_ops = find_non_dominated(u_ops)
            
            print(f"\n{n_new_pipes} new pipelines added, with {n_new_groups} new structures")
            
            print(f"archive contains {len(ndf_params)} non-dominated solutions, {len(u_ndf_params)} of which are unique\n")
            
            p_keys = list(tpot.evaluated_individuals_.keys())
            
            u_grp_params = {}
            u_grp_ops = {}
            
            for idx in ndf_params:
                u_grp_params[tpot.evaluated_individuals_[p_keys[idx]]['structure']] = 0
            
            for idx in ndf_params:
                u_grp_ops[tpot.evaluated_individuals_[p_keys[idx]]['structure']] = 0
            
            with open(f_track, 'a') as f:
                f.write(f"{gen};{len(tpot.evaluated_individuals_)};{len(strucs)};{len(ndf_params)};{len(u_ndf_params)};{len(u_grp_params)};{len(ndf_ops)};{len(u_ndf_ops)};{len(u_grp_ops)}\n")

        f_pipe = os.path.join(track_path,f"{problem}_{seed}_pop_{POP_SIZE}.pipes")
        f_struct = os.path.join(track_path,f"{problem}_{seed}_pop_{POP_SIZE}.structures")

        with open(f_pipe, 'w') as f:
            for i,(p,v) in enumerate(tpot.evaluated_individuals_.items()):
                f.write(f"{i};{p};{v['structure']};{v['internal_cv_score']};{v['operator_count']};{len(u.string_to_params(p,config_dict=default_tpot_config_dict))}\n")
                
        with open(f_struct, 'w') as f:
            for i,k in enumerate(strucs.keys()):
                f.write(f"{i};{k}\n")