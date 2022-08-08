#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 10:57:03 2022

@author: gus
"""

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from matplotlib.lines import Line2D
from tpot_config import default_tpot_config_dict
import os
import sys
import utils as u
import numpy as np
import time
PRINT_COL = 12

def convert_str_param(p,config_dict):
    try:
        return float(p[1])
    except ValueError:
        if p[1] in 'TrueFalse':
            return int(bool(p[1]))
        
        p_s = p[0].split("__")
        for k,v in config_dict.items():
            if p_s[0] in k:
                return v[p_s[1]].index(p[1])

params = {
    'RESULTS_DIR'   : 'house_16h_Results_C',
    'PROBLEM'       :  'house_16h',
    'RUN'           : 0,
    }


cwd = os.getcwd()
results_path = os.path.join(cwd,params['RESULTS_DIR'])
if not os.path.exists(results_path):
    sys.exit(f"Cannot find results directory {results_path}")

prob_dir = os.path.join(results_path, params['PROBLEM'])

run_str = str(params['RUN'])    
if params['RUN'] < 10:
    run_str = "0" + str(params['RUN'])
    
run_dir = os.path.join(prob_dir,"Run_" + run_str)

tpot_dir = os.path.join(run_dir, 'tpot')

fname_tpot_prog = os.path.join(tpot_dir, "tpot_progress.out")
fname_matching_pipes = os.path.join(tpot_dir, "matching_pipes.out") 

best_pipe = None
best_cv = None

with open(fname_tpot_prog, 'r') as f:
    for line in f:
        if 'Best initial TPOT CV' in line:
            best_cv = float(line.split(":")[-1])
        if 'Best initial TPOT pipeline' in line:
            best_pipe = next(f).strip()
            break

pipes = {best_pipe:best_cv}

with open(fname_matching_pipes, 'r') as f:
    for line in f:
        pipe = line.split(";")[0]
        cv = float(line.split(";")[-1])
        if cv == -np.inf:
            continue
        pipes[pipe] = cv

best_params = u.string_to_params(best_pipe)

param_list = [v[0] for v in best_params]

rem_p = []

for p in param_list:
    p_s = p.split("__")
    for k,v in default_tpot_config_dict.items():
        if p_s[0] in k:
            if len(v[p_s[1]]) == 1:
                rem_p.append(p)

[param_list.remove(v) for v in rem_p]

hp_x = np.empty((0,len(param_list)))        
        
for pipe in pipes:
    pipe_params = u.string_to_params(pipe)
    hp_x = np.vstack((hp_x,np.array([convert_str_param(v,default_tpot_config_dict) for v in pipe_params if v[0] in param_list])))

scores = []
x_vals = []
freeze_params = []

hp_y = np.array(list(pipes.values())).reshape(-1,1)

print(param_list)

while hp_x.shape[1] > 0:
    x_vals.append(hp_x.shape[1])
    regressor = LinearRegression(fit_intercept=False)
    regressor.fit(hp_x, hp_y)
    scores.append(regressor.score(hp_x, hp_y))
    worst_idx = np.argmin(np.abs(regressor.coef_))
    freeze_params.append(param_list.pop(worst_idx))
    print(f"\ncoeffs: {regressor.coef_}")
    print(f"worst: {worst_idx}")
    print(f"new param list: {param_list}")
    hp_x = np.delete(hp_x, worst_idx, axis=1)
    
print(f"\nscores: {scores}\n")

# x_vals = list(range(len(scores),0,-1))

plt.plot(x_vals,scores)

freeze_idx = np.argmin(np.abs(scores-0.95*np.max(scores)))

print(f"parameters to freeze: {freeze_params[:freeze_idx]}")



# # print(x_vals)

# angles = []

# for i in range(1,len(scores)-1):
#     a1 = np.arctan(scores[i] - scores[i+1])
#     a2 = np.arctan(scores[i-1] - scores[i])
#     angles.append(a1 - a2)

# print(angles)

# knee_idx = np.argmax(np.array(angles))

# print(knee_idx)

