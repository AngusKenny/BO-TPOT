#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 10:37:55 2022

@author: gus


find ND set of cv best and n_root
allocate pop_size evaluations over all of those
recompute ND set
if ND set has changed, re-allocate
do for 20 gens

"""

import sys
import os
import copy
from utils import tpot_utils as u
from utils.samplers import CustomTPESampler
from config.tpot_config import default_tpot_config_dict
import numpy as np
import pygmo as pg
import optuna
from utils.bo_utils import (make_hp_space_cont, 
                              make_hp_space_discrete, 
                              make_optuna_trial_cont,
                              make_optuna_trial_discrete)


P_COL = 22
RESULTS_PATH = "Results (copy)"
PROBLEM = "socmob"
RUN = "Run_09"
STOP_GEN = 80
ND_SORT = ('cv_best','n_operators')
VERBOSITY = 2
nTPE_CANDIDATES = 1000
DISCRETE_MODE = True

cwd = os.getcwd()
f_pipes = os.path.join(cwd, RESULTS_PATH, PROBLEM, RUN, 'TPOT-BASE', 'TPOT-BASE.pipes')
# f_out = os.path.join(cwd, RESULTS_PATH, 'extracted',f'{PROBLEM}_{RUN}_unique_data.csv')

pipes = u.get_progress_pop(f_pipes,STOP_GEN-1)
u_pipes = u.get_unique_groups(pipes, STOP_GEN-1, config_dict=default_tpot_config_dict)

u_pipes2 = u.load_unique_pop(f_pipes, STOP_GEN-1, config_dict=default_tpot_config_dict)

print(len(u_pipes))

print(len(u_pipes2))


u_pipes = {k:v for k,v in u_pipes.items() if (len(v['bo_params']) != 0 and len(v['matching']) != 1)}
u_pipes2 = {k:v for k,v in u_pipes2.items() if (len(v['bo_params']) != 0 and len(v['matching']) != 1)}

for k,v in u_pipes.items():
    v['n_root'] = np.power(len(v['matching']),1/len(v['bo_params']))

for k,v in u_pipes2.items():
    v['n_root'] = np.power(len(v['matching']),1/len(v['bo_params']))


points = np.hstack((np.array([-v[ND_SORT[0]] for k,v in u_pipes.items()]).reshape(-1,1), np.array([v[ND_SORT[1]] for k,v in u_pipes.items()]).reshape(-1,1)))

points2 = np.hstack((np.array([-v[ND_SORT[0]] for k,v in u_pipes2.items()]).reshape(-1,1), np.array([v[ND_SORT[1]] for k,v in u_pipes2.items()]).reshape(-1,1)))


ks = list(u_pipes.keys())

ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points=points)
ndf2, dl2, dc2, ndr2 = pg.fast_non_dominated_sorting(points=points2)

print(ndf[0],ndf2[0])

for idx in ndf[0]:
    print(u_pipes[ks[idx]]['n_operators'],u_pipes[ks[idx]]['cv_best'],end="")
    if '202.713849' in str(u_pipes[ks[idx]]['cv_best']):
        print("*")
    else:
        print("")

# nd_idxs = np.concatenate(ndf)

# print(nd_idxs)

# for front,front_ids in enumerate(ndf):
#     for i in front_ids:
#         u_pipes[ks[i]]['nd_front'] = front

# nd_pipes = {}

# for i in ndf[0]:
#     nd_pipes[ks[i]] = u_pipes[ks[i]]

# suggested = {}

# for k1,v1 in nd_pipes.items():
#     suggested[k1] = {}
#     seed_samples = [(u.string_to_params(k2,config_dict=default_tpot_config_dict), v2['internal_cv_score'])
#                     for k2,v2 in v1['matching'].items()]
        
#     optuna.logging.set_verbosity(verbosity=VERBOSITY)
#     sampler = CustomTPESampler(n_startup_trials=0, n_ei_candidates=nTPE_CANDIDATES)
#     study = optuna.create_study(sampler=sampler, direction="maximize")

#     for seed_sample in seed_samples:
#         if DISCRETE_MODE:
#             trial = make_optuna_trial_discrete(seed_sample[0], seed_sample[1])
#         else:
#             trial = make_optuna_trial_cont(seed_sample[0], seed_sample[1])
        
#         study.add_trial(trial)
    
#     distributions = copy.deepcopy(trial.distributions)
    
#     for param in v1['bo_params']:
#         trial = study.ask(distributions)
#         suggested[k1][param[0]] = study.sampler.sample_independent_override(study, trial, param[0], distributions[param[0]])
        

# for k1,v1 in nd_pipes.items():
#     print([(k2,v2['best_param_val']) for k2,v2 in suggested[k1].items()])
#     v1['avg_tpe'] = np.mean([v2['best_score'] for k2,v2 in suggested[k1].items()])


# with open(f_out, 'w') as f:
#     f.write(f"{str('mean_cv'):<{P_COL}};{str('std_cv'):<{P_COL}};{str('best_cv'):<{P_COL}};{str('n_hps'):<{P_COL}};{str('n_matching'):<{P_COL}};{str('n_root'):<{P_COL}};{str('avg_tpe'):<{P_COL}}\n")
#     for k,v in nd_pipes.items():
#         f.write(f"{-v['cv_mu']:<{P_COL}};{v['cv_sigma']:<{P_COL}};{-v['cv_best']:<{P_COL}};{len(v['bo_params']):<{P_COL}};{len(v['matching']):<{P_COL}};{v['n_root']:<{P_COL}};{v['avg_tpe']:<{P_COL}}\n")

# with open(f_out, 'w') as f:
#     nd_string = f"nd_front{ND_SORT}"
#     f.write(f"{str('mean_cv'):<{P_COL}};{str('std_cv'):<{P_COL}};{str('best_cv'):<{P_COL}};{str('n_hps'):<{P_COL}};{str('n_matching'):<{P_COL}};{str('n_root'):<{P_COL}};{nd_string:<{P_COL}}\n")
#     for i in nd_idxs:
#         f.write(f"{-u_pipes[ks[i]]['cv_mu']:<{P_COL}};{u_pipes[ks[i]]['cv_sigma']:<{P_COL}};{-u_pipes[ks[i]]['cv_best']:<{P_COL}};{len(u_pipes[ks[i]]['bo_params']):<{P_COL}};{len(u_pipes[ks[i]]['matching']):<{P_COL}};{u_pipes[ks[i]]['n_root']:<{P_COL}};{u_pipes[ks[i]]['nd_front']:<{P_COL}}\n")
        
        
