#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 12:30:39 2022

@author: gus
"""

import os
import sys
import utils.tpot_utils as u
import numpy as np
import time
PRINT_COL = 20
'''
***** Parameter constants *****
PROBLEM:        String with problem name defined by its filename, and also
                where the run data is stored. e.g., /Data/abalone.data 
                would be 'abalone'    
RUN_LIST:           List of runs to plot. Set to [] to plot using all data.
SAVE_PLOTS:     Save generated plots to file in ./<RESULTS_DIR>/Plots/
'''

params = {
    'RESULTS_DIR'       : 'Results',
    'PROBLEMS'          : [
                            # 'abalone',
                            # 'socmob',
                            # 'abalone',
                            'brazilian_houses',
                            # 'house_16h',
                            # 'elevators'
                          ],
    'METHODS'           : ['TPOT-BO-H'],
    'RUN_LIST'          : [],
    'SAVE_STATS'        : False,
    'MODE'              : ['discrete','continuous'],
    # 'MODE'              : ['discrete'],
    # 'MODE'              : ['continuous'],
    'THRESHOLD'         : 1e-8
    }

cwd = os.getcwd()
results_path = os.path.join(cwd,params['RESULTS_DIR'])
if not os.path.exists(results_path):
    sys.exit(f"Cannot find results directory {results_path}")

params['RUN_LIST'] = list(params['RUN_LIST'])

prob_list = params['PROBLEMS']
# if problem list is empty, search problem directory for problems
if len(prob_list) == 0:
    prob_list = [os.path.basename(d) 
                 for d in os.scandir(results_path) if d.is_dir()]

stats = {}
 
data = {}

results = {mode: {problem: {method: {'win':0, 'loss':0, 'draw':0} for method in params['METHODS']} for problem in prob_list} for mode in params['MODE']} 

for mode in params['MODE']:
    
    data[mode] = {}
    stats[mode] = {}
    
    # iterate over problem list
    for problem in prob_list:
        
        prob_path = os.path.join(results_path, problem)
    
        print(f"Processing results from {prob_path}")
        
        if len(params['RUN_LIST']) == 0:
            run_idxs = [int(d.path.split("_")[-1]) 
                        for d in os.scandir(prob_path) 
                        if d.is_dir() and "Plots" not in d.path]
            run_idxs.sort()
        else:
            run_idxs = params['RUN_LIST']
        
        # pop_size = stop_gen = tot_gens = bo_trials = None
        # discrete_mode = restricted_hps = None
        # alt_tpot_gens = alt_bo_trials = n_iters = None
        # auto_pop_size = auto_gen = None
        
        data[mode][problem] = {}
        skipped_runs = []
        
        # validate and collect data from specified runs
        for run in run_idxs:
            run_str = str(run)    
            if run < 10:
                run_str = "0" + str(run)
                
            run_path = os.path.join(prob_path,"Run_" + run_str)
            
            files = {method: os.path.join(run_path, method, mode,f'{method}.progress') for method in params['METHODS']}
            files['TPOT-BASE'] = os.path.join(run_path, 'TPOT-BASE', 'TPOT-BASE.progress')
            
            # check if correct files exist if not then skip run
            for fpath in files.values():
                if not os.path.exists(fpath):
                    print(f"{u.RED}Missing file error:{u.OFF} Run {run} of " 
                          + f"{problem} is missing file {os.path.basename(fpath)}\n{fpath}"
                          + " - skipping run..")
                    skipped_runs.append(run)
                    break
            
            if run in skipped_runs:
                continue
            
            data[mode][problem][run] = {}
            
            for method in files:
                with open(files[method], 'r') as f:
                    if method == 'TPOT-BASE':
                        for line in f:
                            if 'Best full TPOT CV' in line:
                                data[mode][problem][run][method] = -float(line.split(":")[-1])
                
                    if method == 'TPOT-BO-S' or method == 'TPOT-BO-H':
                        read_data = False
                        n_evals = 0
                        for line in f:
                            if 'BAYESIAN OPTIMISATION EVALS:' in line:
                                n_evals = int(line.split(':')[-1])
                            if f'{n_evals} BAYESIAN OPTIMISATION' in line:
                                read_data = True
                            if 'Best CV' in line and read_data:
                                data[mode][problem][run][method] = -float(line.split(":")[-1])
                                
                    if method == 'TPOT-BO-ALT':
                        read_data = False
                        final_iter = 0
                        for line in f:
                            if 'nITERS' in line:
                                final_iter = int(line.split(':')[-1])
                            if f'ITERATION {final_iter-1}' in line:
                                read_data = True
                            if 'Best BO CV' in line and read_data:
                                data[mode][problem][run][method] = -float(line.split(":")[-1])
                                
                    if method == 'TPOT-BO-AUTO':
                        read_data = False
                        final_gen = 100
                        for line in f:
                            if 'nGENS' in line:
                                final_iter = int(line.split(':')[-1])
                            if f'GENERATION {final_gen-1}' in line:
                                read_data = True
                            if 'Best CV' in line and read_data:
                                data[mode][problem][run][method] = -float(line.split(":")[-1])
                        
                if method not in data[mode][problem][run]:
                    print(f"{u.RED}Data read error:{u.OFF} {files[method]} does not contain valid data for {method} for run {run}")
    
        for method in params['METHODS']:
            for run in data[mode][problem]:
                if data[mode][problem][run]['TPOT-BASE'] - data[mode][problem][run][method] > params['THRESHOLD']:
                    results[mode][problem][method]['win'] = results[mode][problem][method]['win'] + 1
                elif (data[mode][problem][run][method] - data[mode][problem][run]['TPOT-BASE']) > params['THRESHOLD']:
                    results[mode][problem][method]['loss'] = results[mode][problem][method]['loss'] + 1
                else:
                    results[mode][problem][method]['draw'] = results[mode][problem][method]['draw'] + 1
    
print()
print(f"{str(''):>{PRINT_COL}}{str('Win'):>{PRINT_COL}}{str('Draw'):>{PRINT_COL}}{str('Loss'):>{PRINT_COL}}{str('Total'):>{PRINT_COL}}")
for problem in prob_list:
    for mode in params['MODE']:
        print(f"{problem} ({mode}):")
        for method in params['METHODS']:
            print(f"{method:>{PRINT_COL}}",end="")
            for stat in results[mode][problem][method].values():
                print(f"{stat:>{PRINT_COL}}",end="")
            print(f"{np.sum([stat for stat in results[mode][problem][method].values()]):>{PRINT_COL}}")
        print()
        