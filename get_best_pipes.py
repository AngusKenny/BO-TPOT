#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 18:47:05 2022

@author: gus
"""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import sys
import utils as u
import numpy as np
import time
PRINT_COL = 12


params = {
    'RESULTS_DIR'   : 'Results_discrete',
    'PROBLEMS'      : {'quake': range(21),
                       'abalone': range(21),
                       'socmob': range(21),
                       'brazilian_houses': [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 22, 24, 25],
                       'elevators': range(21),
                       'house_16h': range(21)},
    'SAVE_PLOTS'    : True,
    'SAVE_STATS'    : True,
    'PLOT_ALT'      : True,
    'PLOT_AUTO'      : True,
    'PLOT_MIN_MAX'  : False,
    'SKIP_PLOT_INIT': 10,
    'ADD_TITLE_TEXT': '(continuous)',
    'COLOURMAP'     : plt.cm.bwr
    }


cwd = os.getcwd()
results_path = os.path.join(cwd,params['RESULTS_DIR'])
if not os.path.exists(results_path):
    sys.exit(f"Cannot find results directory {results_path}")

prob_list = params['PROBLEMS']
# if problem list is empty, search problem directory for problems
if len(prob_list) == 0:
    prob_list = [os.path.basename(d) 
                 for d in os.scandir(results_path) if d.is_dir()]


pipe_data = {k: {} for k in params['PROBLEMS'].keys()}

for problem, run_list in params['PROBLEMS'].items():
    runs = list(run_list)

    prob_dir = os.path.join(results_path, problem)

    print(f"Processing results from {prob_dir}")
    
    # validate and collect data from specified runs
    for run in runs:
        run_str = str(run)    
        if run < 10:
            run_str = "0" + str(run)
            
        run_dir = os.path.join(prob_dir,"Run_" + run_str)
        
        tpot_dir = os.path.join(run_dir, 'tpot')
        
        fname_tpot_prog = os.path.join(tpot_dir, "tpot_progress.out")
        fname_matching_pipes = os.path.join(tpot_dir, "matching_pipes.out") 
        
        with open(fname_tpot_prog, 'r') as f:
            for line in f:
                if 'Best initial TPOT pipeline' in line:
                    n_matching = int(line.split("(")[-1].split(" ")[0])
                    pipe_data[problem][next(f).strip()] = n_matching
                    break
                    
param_counts = np.hstack([np.array([len(u.string_to_params(pipe)) for pipe in pipes]).reshape(-1,1) for prob,pipes in pipe_data.items()])

match_counts = np.hstack([np.array([val+1 for val in pipes.values()]).reshape(-1,1) for prob,pipes in pipe_data.items()])


print(f"\nparam_counts:\n{param_counts}\n")
print(f"match_counts:\n{match_counts}\n")

fig1, ax_param = plt.subplots()
ax_param.grid()
boxplot_data = ax_param.boxplot(param_counts,patch_artist=True)
ax_param.set_xticks(range(1,len(params['PROBLEMS'])+1), params['PROBLEMS'].keys(),
       rotation=20)  # Set text labels and properties.
# ax_param.set_xlim([0.25,max(matching_idx)+.75])
ax_param.set_xlabel("Problem")
ax_param.set_ylabel("Number of hyperparameters")
ax_param.set_title(f"Number of hyperparameters for best TPOT pipeline @ 80 gens")

        
fig2, ax_match = plt.subplots()
ax_match.grid()
boxplot_data = ax_match.boxplot(match_counts,patch_artist=True)
ax_match.set_xticks(range(1,len(params['PROBLEMS'])+1), params['PROBLEMS'].keys(),
       rotation=20)  # Set text labels and properties.
# ax_match.set_xlim([0.25,max(matching_idx)+.75])
ax_match.set_xlabel("Problem")
ax_match.set_ylabel("Number of matching pipes")
ax_match.set_title(f"Number of matching pipes for best TPOT pipeline @ 80 gens")

fig3, ax_match2 = plt.subplots()
ax_match2.grid()
boxplot_data = ax_match2.boxplot(match_counts,showfliers=False,patch_artist=True)
ax_match2.set_xticks(range(1,len(params['PROBLEMS'])+1), params['PROBLEMS'].keys(),
       rotation=20)  # Set text labels and properties.
# ax_match2.set_xlim([0.25,max(matching_idx)+.75])
ax_match2.set_xlabel("Problem")
ax_match2.set_ylabel("Number of matching pipes")
ax_match2.set_title(f"Number of matching pipes for best TPOT pipeline @ 80 gens")

plot_dir = os.path.join(results_path, "Stats")

if params['SAVE_PLOTS']:
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    fname_params = os.path.join(
        plot_dir, "hp_per_prob.png")
    fname_match1 = os.path.join(
        plot_dir, "matching_w_outliers.png")
    fname_match2 = os.path.join(
        plot_dir, "matching_no_outliers.png")
    fig1.savefig(fname_params,bbox_inches='tight')
    fig2.savefig(fname_match1,bbox_inches='tight')
    fig3.savefig(fname_match2,bbox_inches='tight')
    


# for problem in prob_list:
#     print(f"\n{u.CYAN}{problem} statistics:{u.OFF}")
#     print(f"{str(''):>{PRINT_COL}}{str('TPOT only'):>{PRINT_COL}}{str('TPOT + BO'):>{PRINT_COL}}{str('TPOT + BO (alt)'):>{PRINT_COL}}{str('TPOT + BO (auto)'):>{PRINT_COL}}")
#     for _ in range(5*PRINT_COL):
#         print("=",end='')
#     print()
    
#     for stat,methods in stats[problem].items():
#         if stat == 'runs':
#             continue
#         print(f"{str(stat):>{PRINT_COL}}",end="")
#         for method,val in methods.items():
#             print(f"{round(val,12):>{PRINT_COL}}",end="")
#         print()


for i in range(len(pipe_data)):
    prob = list(pipe_data.keys())[i]
    print(f"{prob}:")
    for _ in range(len(prob)+1):
        print("-",end="")
    print(f"\nmean: {np.mean(param_counts[:,i])}, median: {np.median(param_counts[:,i])}\n")


print(np.hstack((np.array(range(param_counts.shape[0])).reshape(-1,1),param_counts[:,1].reshape(-1,1),match_counts[:,1].reshape(-1,1),(match_counts[:,1]-param_counts[:,1]).reshape(-1,1))))
      
