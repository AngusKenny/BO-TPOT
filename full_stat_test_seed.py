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
from scipy.stats import ranksums
PRINT_COL = 15
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
                            'quake',
                            # 'socmob',
                            # 'abalone',
                            # 'brazilian_houses',
                            # 'house_16h',
                            # 'elevators'
                          ],
    'METHODS'           : ['TPOT-BASE',
                        #    'TPOT-BO-Sd','TPOT-BO-Sc',
                            # 'TPOT-BO-ALTd','TPOT-BO-ALTc',
                        #    'TPOT-BO-AUTOd','TPOT-BO-AUTOc',
                        #    'TPOT-BO-Hd','TPOT-BO-Hc',
                        #    'TPOT-BO-Hs',
                           'oTPOT-BASE',
                           ],
    'SEED_LIST'          : [],
    'SAVE_STATS'        : False,
    # 'MODE'              : ['discrete'],
    # 'MODE'              : ['continuous'],
    'CONFIDENCE_LEVEL'  : 0.1
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

results = {problem: {base_method: {method: {'win':0, 'draw':0, 'loss':0} for method in params['METHODS']} for base_method in params['METHODS']} for problem in params['PROBLEMS']} 

for problem in prob_list:
    data[problem] = {}
    stats[problem] = {}
    
    prob_path = os.path.join(results_path, problem)
    
    print(f"Processing results from {prob_path}")
    
    for method in params['METHODS']:
        mode = '' if method == 'TPOT-BASE' else 'discrete' if 'd' in method else 'continuous' if 'c' in method else 'sequential'
        raw_method = method.strip('dcs')
    
        method_path = os.path.join(prob_path,raw_method,mode)
    
        data[problem][method] = {}
        
        if len(params['SEED_LIST']) == 0:
            seed_idxs = [int(d.path.split("_")[-1]) 
                            for d in os.scandir(method_path) 
                            if d.is_dir() and "Plots" not in d.path]
            seed_idxs.sort()
        else:
            seed_idxs = params['SEED_LIST']
        
        skipped_seeds = []
    
        # validate and collect data from specified runs
        for seed in seed_idxs:
            seed_path = os.path.join(prob_path,f"Seed_{seed}")
        
        
        
        
    
        for method in params['METHODS']:
            mode = '' if method == 'TPOT-BASE' else 'discrete' if 'd' in method else 'continuous' if 'c' in method else 'sequential'
            raw_method = method.strip('dcs')
            
            f_prog = os.path.join(run_path,raw_method,mode,f'{method}.progress') if 's' in method else os.path.join(run_path,raw_method,mode,f'{raw_method}.progress')
            
            if not os.path.exists(f_prog):
                print(f"{u.RED}Missing file error:{u.OFF} Run {run} of " 
                      + f"{problem} is missing file {os.path.basename(f_prog)}\n{f_prog}"
                      + " - skipping run..")
                skipped_runs.append(run)
                break
            
            with open(f_prog, 'r') as f:
                if raw_method == 'TPOT-BASE':
                    for line in f:
                        if 'Best full TPOT CV' in line:
                            data[problem][run][method] = -float(line.split(":")[-1])
            
                if raw_method == 'TPOT-BO-S' or (raw_method == 'TPOT-BO-H' and method != 'TPOT-BO-Hs'):
                    read_data = False
                    n_evals = 0
                    for line in f:
                        if 'BAYESIAN OPTIMISATION EVALS:' in line:
                            n_evals = int(line.split(':')[-1])
                        if f'{n_evals} BAYESIAN OPTIMISATION' in line:
                            read_data = True
                        if 'Best CV' in line and read_data:
                            data[problem][run][method] = -float(line.split(":")[-1])
                            
                if method == 'TPOT-BO-Hs':
                    for line in f:
                        if 'AFTER' in line and 'TPOT-BO-Hs' in line:
                            next(f)
                            cv_line = next(f)
                            data[problem][run][method] = -float(cv_line.split(":")[-1])
                            
                if raw_method == 'TPOT-BO-ALT':
                    read_data = False
                    final_iter = 0
                    for line in f:
                        if 'nITERS' in line:
                            final_iter = int(line.split(':')[-1])
                        if f'ITERATION {final_iter-1}' in line:
                            read_data = True
                        if 'Best BO CV' in line and read_data:
                            data[problem][run][method] = -float(line.split(":")[-1])
                            
                if raw_method == 'TPOT-BO-AUTO':
                    read_data = False
                    final_gen = 100
                    for line in f:
                        if 'nGENS' in line:
                            final_iter = int(line.split(':')[-1])
                        if f'GENERATION {final_gen-1}' in line:
                            read_data = True
                        if 'Best CV' in line and read_data:
                            data[problem][run][method] = -float(line.split(":")[-1])
                    
            if method not in data[problem][run]:
                print(f"{u.RED}Data read error:{u.OFF} {f_prog} does not contain valid data for {method} for run {run}")
            
            
        if run in skipped_runs:
            continue            

    # for base_method in params['METHODS']:
    #     for tgt_method in params['METHODS']:
    #         for run in data[problem]:
    #             threshold = np.power(10, np.floor(np.log10(data[problem][run][base_method]))-params['THRESHOLD_DEG'])
    #             if data[problem][run][base_method] - data[problem][run][tgt_method] > threshold:
    #                 results[problem][base_method][tgt_method]['win'] = results[problem][base_method][tgt_method]['win'] + 1
    #             elif (data[problem][run][tgt_method] - data[problem][run][base_method]) > threshold:
    #                 results[problem][base_method][tgt_method]['loss'] = results[problem][base_method][tgt_method]['loss'] + 1
    #             else:
    #                 results[problem][base_method][tgt_method]['draw'] = results[problem][base_method][tgt_method]['draw'] + 1

    

# for prob,d in results.items():
#     print(f'{prob}:')
#     print('='*(len(prob)+1))
#     print(f"{str(''):>{PRINT_COL}}",end="")
#     for method in d.keys():
#         print(f"{method:>{PRINT_COL}}",end="")
#     print()
#     for method,vals in d.items():
#         print(f"{method:>{PRINT_COL}}",end="")
#         for v in vals.values():
#             wdl = f"{v['win']}/{v['draw']}/{v['loss']}"
#             print(f"{wdl:>{PRINT_COL}}",end="")
#         print()
#     print()
    
    
stat_data = {problem: {method: np.array([data[problem][run][method] for run in data[problem]]) for method in params['METHODS']} for problem in params['PROBLEMS']}

med_data = {problem: {method: np.median(stat_data[problem][method]) for method in params['METHODS']} for problem in params['PROBLEMS']}

print("\n\n")

wtl_overall = {tgt: {src:{'W':0, 'T': 0, 'L':0} for src in params['METHODS']} for tgt in params['METHODS']}

for prob,d in stat_data.items():
    print(f'{prob}:')
    print('='*(len(prob)+1))
    print(f"{str('.'):>{PRINT_COL}}",end="")
    for src_method in params['METHODS']:
        print(f"{src_method:>{PRINT_COL}}",end="")
    print()
    for tgt_method in params['METHODS']:
        print(f"{tgt_method:>{PRINT_COL}}",end="")
        for src_method in params['METHODS']:
            res = ranksums(stat_data[prob][src_method],stat_data[prob][tgt_method])
            wtl_res = 'T'
            if res.pvalue <= params['CONFIDENCE_LEVEL']:
                wtl_res = 'W' if med_data[prob][src_method] < med_data[prob][tgt_method] else 'L'
            
            wtl_overall[tgt_method][src_method][wtl_res] = wtl_overall[tgt_method][src_method][wtl_res] + 1
            
            res_txt = f"{wtl_res}(p={res.pvalue:.4f})"
            
            print(f"{res_txt:>{PRINT_COL}}",end="")
        print()
    print()
        
        
print(f"\n\nover all (confidence level {int(params['CONFIDENCE_LEVEL']*100)}%):")
print(f"{str('.'):>{PRINT_COL}}",end="")
for src_method in params['METHODS']:
    print(f"{src_method:>{PRINT_COL}}",end="")
print()
for tgt_method in params['METHODS']:
    print(f"{tgt_method:>{PRINT_COL}}",end="")
    for src_method in params['METHODS']:
        wtl_txt = f"{wtl_overall[tgt_method][src_method]['W']}/{wtl_overall[tgt_method][src_method]['T']}/{wtl_overall[tgt_method][src_method]['L']}"
        print(f"{wtl_txt:>{PRINT_COL}}",end="")
    print()
print()

print("x = {",end="")
for i,prob in enumerate(stat_data):
    print(f"{stat_data[prob]['TPOT-BASE']},",end="")
print("}")

print("z = {",end="")
for i,prob in enumerate(stat_data):
    print(f"{stat_data[prob]['TPOT-BO-Hs']},",end="")
print("}")


# for mode in params['MODE']:
    
#     data[mode] = {}
#     stats[mode] = {}
    
#     # iterate over problem list
#     for problem in prob_list:
        
#         prob_path = os.path.join(results_path, problem)
    
#         print(f"Processing results from {prob_path}")
        
#         if len(params['RUN_LIST']) == 0:
#             run_idxs = [int(d.path.split("_")[-1]) 
#                         for d in os.scandir(prob_path) 
#                         if d.is_dir() and "Plots" not in d.path]
#             run_idxs.sort()
#         else:
#             run_idxs = params['RUN_LIST']
        
#         # pop_size = stop_gen = tot_gens = bo_trials = None
#         # discrete_mode = restricted_hps = None
#         # alt_tpot_gens = alt_bo_trials = n_iters = None
#         # auto_pop_size = auto_gen = None
        
#         data[mode][problem] = {}
#         skipped_runs = []
        
#         # validate and collect data from specified runs
#         for run in run_idxs:
#             run_str = str(run)    
#             if run < 10:
#                 run_str = "0" + str(run)
                
#             run_path = os.path.join(prob_path,"Run_" + run_str)
            
#             files = {method: os.path.join(run_path, method, mode,f'{method}.progress') for method in params['METHODS']}
#             files['TPOT-BASE'] = os.path.join(run_path, 'TPOT-BASE', 'TPOT-BASE.progress')
            
#             # check if correct files exist if not then skip run
#             for fpath in files.values():
#                 if not os.path.exists(fpath):
#                     print(f"{u.RED}Missing file error:{u.OFF} Run {run} of " 
#                           + f"{problem} is missing file {os.path.basename(fpath)}\n{fpath}"
#                           + " - skipping run..")
#                     skipped_runs.append(run)
#                     break
            
#             if run in skipped_runs:
#                 continue
            
#             data[mode][problem][run] = {}
            
#             for method in files:
#                 with open(files[method], 'r') as f:
#                     if method == 'TPOT-BASE':
#                         for line in f:
#                             if 'Best full TPOT CV' in line:
#                                 data[mode][problem][run][method] = -float(line.split(":")[-1])
                
#                     if method == 'TPOT-BO-S' or method == 'TPOT-BO-H':
#                         read_data = False
#                         n_evals = 0
#                         for line in f:
#                             if 'BAYESIAN OPTIMISATION EVALS:' in line:
#                                 n_evals = int(line.split(':')[-1])
#                             if f'{n_evals} BAYESIAN OPTIMISATION' in line:
#                                 read_data = True
#                             if 'Best CV' in line and read_data:
#                                 data[mode][problem][run][method] = -float(line.split(":")[-1])
                                
#                     if method == 'TPOT-BO-ALT':
#                         read_data = False
#                         final_iter = 0
#                         for line in f:
#                             if 'nITERS' in line:
#                                 final_iter = int(line.split(':')[-1])
#                             if f'ITERATION {final_iter-1}' in line:
#                                 read_data = True
#                             if 'Best BO CV' in line and read_data:
#                                 data[mode][problem][run][method] = -float(line.split(":")[-1])
                                
#                     if method == 'TPOT-BO-AUTO':
#                         read_data = False
#                         final_gen = 100
#                         for line in f:
#                             if 'nGENS' in line:
#                                 final_iter = int(line.split(':')[-1])
#                             if f'GENERATION {final_gen-1}' in line:
#                                 read_data = True
#                             if 'Best CV' in line and read_data:
#                                 data[mode][problem][run][method] = -float(line.split(":")[-1])
                        
#                 if method not in data[mode][problem][run]:
#                     print(f"{u.RED}Data read error:{u.OFF} {files[method]} does not contain valid data for {method} for run {run}")
        
#         for method in params['METHODS']:
#             for run in data[mode][problem]:
#                 threshold = np.power(10, np.floor(np.log10(data[mode][problem][run]['TPOT-BASE']))-params['THRESHOLD_DEG'])
#                 if data[mode][problem][run]['TPOT-BASE'] - data[mode][problem][run][method] > threshold:
#                     results[mode][problem][method]['win'] = results[mode][problem][method]['win'] + 1
#                 elif (data[mode][problem][run][method] - data[mode][problem][run]['TPOT-BASE']) > threshold:
#                     results[mode][problem][method]['loss'] = results[mode][problem][method]['loss'] + 1
#                 else:
#                     results[mode][problem][method]['draw'] = results[mode][problem][method]['draw'] + 1
    
# print()
# print(f"{str(''):>{PRINT_COL}}{str('Win'):>{PRINT_COL}}{str('Draw'):>{PRINT_COL}}{str('Loss'):>{PRINT_COL}}{str('Total'):>{PRINT_COL}}")
# for problem in prob_list:
#     for mode in params['MODE']:
#         print(f"{problem} ({mode}):")
#         for method in params['METHODS']:
#             print(f"{method:>{PRINT_COL}}",end="")
#             for stat in results[mode][problem][method].values():
#                 print(f"{stat:>{PRINT_COL}}",end="")
#             print(f"{np.sum([stat for stat in results[mode][problem][method].values()]):>{PRINT_COL}}")
#         print()
        