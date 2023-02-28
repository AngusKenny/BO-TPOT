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
PROBLEMS:        String with problem name defined by its filename, and also
                where the run data is stored. e.g., /Data/abalone.data 
                would be 'abalone'    
RUN_LIST:           List of runs to plot. Set to [] to plot using all data.
'''

### suffix 2 indicates old OCBA without checking for outliers

params = {
    'RESULTS_DIR'       : 'Results',
    'PROBLEMS'          : 
        [
            'quake',
            # 'socmob',
            # 'abalone',
            # 'brazilian_houses',
            # 'house_16h',
            # 'elevators'
        ],
    'METHODS'           : 
        [
            'TPOT-BASE',
            'TPOT-BO-Sd',
            'TPOT-BO-Sc',
            'TPOT-BO-ALTd',
            'TPOT-BO-ALTc',
            'TPOT-BO-AUTOd',
            'TPOT-BO-AUTOc',
        ],
    'STOP_GEN'       : 80,
    'SEED_LIST'          : [],
    'CONFIDENCE_LEVEL'  : 0.05
    }

cwd = os.getcwd()
results_path = os.path.join(cwd,params['RESULTS_DIR'])
if not os.path.exists(results_path):
    sys.exit(f"Cannot find results directory {results_path}")

params['SEED_LIST'] = list(params['SEED_LIST'])

prob_list = params['PROBLEMS']
# if problem list is empty, search problem directory for problems
if len(prob_list) == 0:
    prob_list = [os.path.basename(d) 
                 for d in os.scandir(results_path) if d.is_dir()]

stats = {}
 
data = {}

method_list = [f"TPOT-BASE{params['STOP_GEN']}"] + params['METHODS'] if params['STOP_GEN'] else params['METHODS']

results = {problem: {base_method: {method: {'win':0, 'draw':0, 'loss':0} for method in method_list} for base_method in method_list} for problem in params['PROBLEMS']} 

for problem in prob_list:
    data[problem] = {}
    stats[problem] = {}
    
    prob_path = os.path.join(results_path, problem)
    
    print(f"Processing results from {prob_path}")

    if len(params['SEED_LIST']) == 0:
        tb_path = os.path.join(prob_path,'TPOT-BASE')


        seed_idxs = [int(d.path.split("_")[-1]) 
                        for d in os.scandir(tb_path) 
                        if d.is_dir() and "Seed" in d.path]
        seed_idxs.sort()
    else:
        seed_idxs = params['SEED_LIST']
    
    skipped_seeds = []

    # validate and collect data from specified runs
    for seed in seed_idxs:
        seed_str = str(seed)    
        if seed < 10:
            run_str = "0" + str(seed)
    
        data[problem][seed] = {}
    
        for method in params['METHODS']:
            base_method = method.strip('dc')

            method_path = os.path.join(prob_path,method)

            if not os.path.exists(method_path):
                print(f"{u.RED}Path error:{u.OFF} Method {method} of " 
                      + f"{problem} is missing"
                      + " - skipping method..")
                continue

            seed_path = os.path.join(method_path,"Seed_" + seed_str)
            if not os.path.exists(seed_path):
                print(f"{u.RED}Path error:{u.OFF} Seed {seed} of " 
                      + f"{method} in {problem} is missing"
                      + " - skipping seed..")
                skipped_seeds.append(seed)
                continue

            f_prog = os.path.join(seed_path,f'{method}.progress')

            if not os.path.exists(f_prog):
                print(f"{u.RED}Missing file error:{u.OFF} Seed {seed} of " 
                      + f"{problem} is missing file {os.path.basename(f_prog)}\n{f_prog}"
                      + " - skipping seed..")
                skipped_seeds.append(seed)
                break
            
            with open(f_prog, 'r') as f:
                if method == 'TPOT-BASE':
                    for line in f:
                        if 'Best full TPOT CV' in line:
                            data[problem][seed]['TPOT-BASE'] = -float(line.split(":")[-1])
                        if params['STOP_GEN']:
                            f_pipes = os.path.join(seed_path,f'TPOT-BASE.pipes')
                            t_pipes = u.get_progress_pop(f_pipes,stop_gen=params['STOP_GEN'])
                            min_cv = np.min(np.array([-v['internal_cv_score'] for v in t_pipes.values()]))
                            data[problem][seed][f"TPOT-BASE{params['STOP_GEN']}"] = min_cv
                        
                if base_method == 'TPOT-BO-S':
                    read_data = False
                    n_evals = 0
                    for line in f:
                        if 'BAYESIAN OPTIMISATION EVALS:' in line:
                            n_evals = int(line.split(':')[-1])
                        if f'{n_evals} BAYESIAN OPTIMISATION' in line:
                            read_data = True
                        if 'Best CV' in line and read_data:
                            data[problem][seed][method] = -float(line.split(":")[-1])
                            
                if base_method == 'TPOT-BO-ALT':
                    read_data = False
                    final_iter = 0
                    for line in f:
                        if 'nITERS' in line:
                            final_iter = int(line.split(':')[-1])
                        if f'ITERATION {final_iter-1}' in line:
                            read_data = True
                        if 'Best BO CV' in line and read_data:
                            data[problem][seed][method] = -float(line.split(":")[-1])
                            
                if base_method == 'TPOT-BO-AUTO':
                    read_data = False
                    final_gen = 100
                    for line in f:
                        if 'nGENS' in line:
                            final_iter = int(line.split(':')[-1])
                        if f'GENERATION {final_gen-1}' in line:
                            read_data = True
                        if 'Best CV' in line and read_data:
                            data[problem][seed][method] = -float(line.split(":")[-1])
                    
            if method not in data[problem][seed]:
                print(f"{u.RED}Data read error:{u.OFF} {f_prog} does not contain valid data for {method} for seed {seed}")
            
            
        if seed in skipped_seeds:
            if seed in data[problem]:
                data[problem].pop(seed)
            continue            

stats = {problem: {'best':{},'worst':{},'median':{},'mean':{},'std_dev':{}} for problem in params['PROBLEMS']}

for problem in params['PROBLEMS']:
    for method in method_list:
        stats[problem]['best'][method] = np.min([data[problem][seed][method] for seed in data[problem]])
        stats[problem]['worst'][method] = np.max([data[problem][seed][method] for seed in data[problem]])
        stats[problem]['median'][method] = np.median([data[problem][seed][method] for seed in data[problem]])
        stats[problem]['mean'][method] = np.mean([data[problem][seed][method] for seed in data[problem]])
        stats[problem]['std_dev'][method] = np.std([data[problem][seed][method] for seed in data[problem]])

best_markers = {prob : {'best':None,'worst':None,'median':None,'mean':None,'std_dev':None} for prob in params['PROBLEMS']}

for prob,p_stats in stats.items():
    means = [p_stats['mean'][m] for m in p_stats['mean']]
    bests = [p_stats['best'][m] for m in p_stats['best']]
    medians = [p_stats['median'][m] for m in p_stats['median']]
    m_keys = np.array(list(p_stats['median'].keys()))
    best_markers[prob]['best'] = m_keys[np.flatnonzero(bests == np.min(bests))]
    best_markers[prob]['mean'] = m_keys[np.flatnonzero(means == np.min(means))]
    best_markers[prob]['median'] = m_keys[np.flatnonzero(medians == np.min(medians))]
    
stat_data = {problem: {method: np.array([data[problem][seed][method] for seed in data[problem]]) for method in method_list} for problem in params['PROBLEMS']}

med_data = {problem: {method: np.median(stat_data[problem][method]) for method in method_list} for problem in params['PROBLEMS']}

print("\n\n")

wtl_overall = {tgt: {src:{'W':0, 'T': 0, 'L':0} for src in method_list} for tgt in method_list}

for prob,d in stat_data.items():
    print(f'{prob}:')

    print(f"{str(''):>{PRINT_COL}}",end='')
    for method in method_list:
        print(f"{method:>{PRINT_COL}}",end='')
    print("")
            
    print("="*((len(method_list)+1)*PRINT_COL + 2))
    
    for stat,methods in stats[prob].items():
        print(f"{str(stat):>{PRINT_COL}}",end="")
        for method,val in methods.items():
            col_txt = f"{u.CYAN}" if np.any(best_markers[prob][stat] == method) else ""
            print(f"{col_txt}{val:>{PRINT_COL}.6e}{u.OFF}",end="")
        print()


    print("\n")

    if params['CONFIDENCE_LEVEL']> 0:

        print(f"{str('.'):>{PRINT_COL}}",end="")
        for src_method in method_list:
            print(f"{src_method:>{PRINT_COL}}",end="")
        print()
        for tgt_method in method_list:
            print(f"{tgt_method:>{PRINT_COL}}",end="")
            for src_method in method_list:
                res = ranksums(stat_data[prob][src_method],stat_data[prob][tgt_method])
                wtl_res = 'T'
                if res.pvalue <= params['CONFIDENCE_LEVEL']:
                    wtl_res = 'W' if med_data[prob][src_method] < med_data[prob][tgt_method] else 'L'
                
                wtl_overall[tgt_method][src_method][wtl_res] = wtl_overall[tgt_method][src_method][wtl_res] + 1
                
                res_txt = f"{wtl_res}(p={res.pvalue:.4f})"
                
                print(f"{res_txt:>{PRINT_COL}}",end="")
            print()
        print()
        
if params['CONFIDENCE_LEVEL']> 0:    
    print(f"\n\nover all (confidence level {int(params['CONFIDENCE_LEVEL']*100)}%):")
    print(f"{str('.'):>{PRINT_COL}}",end="")
    for src_method in method_list:
        print(f"{src_method:>{PRINT_COL}}",end="")
    print()
    for tgt_method in method_list:
        print(f"{tgt_method:>{PRINT_COL}}",end="")
        for src_method in method_list:
            wtl_txt = f"{wtl_overall[tgt_method][src_method]['W']}/{wtl_overall[tgt_method][src_method]['T']}/{wtl_overall[tgt_method][src_method]['L']}"
            print(f"{wtl_txt:>{PRINT_COL}}",end="")
        print()
    print()
