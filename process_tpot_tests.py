#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 11:37:39 2022

@author: gus
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
                            'quake',
                            # 'abalone',
                            # 'socmob',
                            # 'brazilian_houses',
                            # 'house_16h',
                            # 'elevators'
                          ],
    'RUN_LIST'          : [],
    'SAVE_PLOTS'        : True,
    'SAVE_STATS'        : True,
    'PLOT_TPOT-BO-Sr'   : True,
    'PLOT_TPOT-BO-ALT'  : True,
    'PLOT_TPOT-BO-AUTO' : True,
    'DISCRETE_MODE'     : False,
    'SKIP_PLOT_INIT'    : 200
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

disc_txt = "discrete" if params['DISCRETE_MODE'] else "continuous"

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
    
    pop_size = stop_gen = tot_gens = bo_trials = None
    discrete_mode = restricted_hps = None
    alt_tpot_gens = alt_bo_trials = n_iters = None
    auto_pop_size = auto_gen = None
    
    data = {}
    skipped_runs = []
    
    # validate and collect data from specified runs
    for run in run_idxs:
        run_str = str(run)    
        if run < 10:
            run_str = "0" + str(run)
            
        run_path = os.path.join(prob_path,"Run_" + run_str)
        
        tpot_path = os.path.join(run_path, 'TPOT-BASE')
        bo_path = os.path.join(run_path, 'TPOT-BO-S',disc_txt)
        bo_r_path = os.path.join(run_path, 'TPOT-BO-Sr',disc_txt)
        alt_path = os.path.join(run_path, 'TPOT-BO-ALT',disc_txt)
        auto_path = os.path.join(run_path, 'TPOT-BO-AUTO',disc_txt)
        
        fname_tpot_prog = os.path.join(tpot_path, "TPOT-BASE.progress")
        fname_tpot_pipes = os.path.join(tpot_path, "TPOT-BASE.pipes")
        
        fname_bo_prog = os.path.join(bo_path, "TPOT-BO-S.progress")
        fname_bo_pipes = os.path.join(bo_path, "TPOT-BO-S.pipes")
        fname_bo_r_prog = os.path.join(bo_r_path, "TPOT-BO-Sr.progress")
        fname_bo_r_pipes = os.path.join(bo_r_path, "TPOT-BO-Sr.pipes")
        
        fname_alt_prog = os.path.join(alt_path, "TPOT-BO-ALT.progress")
        fname_alt_pipes = os.path.join(alt_path, "TPOT-BO-ALT.pipes")
        # fname_alt_bo_pipes = os.path.join(alt_path, "alt_bo_pipes.out")
        
        fname_auto_prog = os.path.join(auto_path, "TPOT-BO-AUTO.progress")
        fname_auto_pipes = os.path.join(auto_path, "TPOT-BO-AUTO.pipes")
        fname_auto_grads = os.path.join(auto_path, "TPOT-BO-AUTO.grads")
        
        
        check_files = [fname_tpot_prog, fname_tpot_pipes, fname_bo_pipes]
        
        
        
        if params['PLOT_TPOT-BO-ALT']:
            check_files = check_files + [fname_alt_prog, 
                                         fname_alt_pipes]#, 
                                         # fname_alt_bo_pipes]
            
        if params['PLOT_TPOT-BO-AUTO']:
            check_files = check_files + [fname_auto_prog, 
                                         fname_auto_pipes, 
                                         fname_auto_grads]
        
        if params['PLOT_TPOT-BO-Sr']:
            check_files = check_files + [fname_bo_r_pipes, fname_bo_r_prog]
        
        # check if correct files exist if not then skip run
        for fpath in check_files:
            if not os.path.exists(fpath):
                print(f"{u.RED}Missing file error:{u.OFF} Run {run} of " 
                      + f"{problem} is missing file {os.path.basename(fpath)}\n{fpath}"
                      + " - skipping run..")
                skipped_runs.append(run)
                break
        
        if run in skipped_runs:
            continue

        r_seed = r_stop_gen = r_tot_gens = r_bo_trials = None
        r_alt_tpot_gens = r_alt_bo_trials = r_n_iters = None
        r_discrete_mode = r_restricted_hps = None
        
        # read data from progress file and verify consistency with first run
        with open(fname_tpot_prog, 'r') as f:
            for line in f:
                if "*" in line:
                    continue
                if "SEED" in line:
                    r_seed = int(line.split(":")[-1])
                if "POP SIZE" in line:
                    r_pop_size = int(line.split(":")[-1])
                    if pop_size == None:
                        pop_size = r_pop_size            
                    else:
                        if r_pop_size != pop_size:
                            print(f"{u.RED}Pop size error:{u.OFF} Run {run} "
                                  + "has different pop size to run " 
                                  f"{run_idxs[0]} ({r_pop_size} != {pop_size})" 
                                  + " - skipping run..")
                            skipped_runs.append(run)
                            break
                if "TOTAL" in line:
                    r_tot_gens = int(line.split(":")[-1])
                    if tot_gens == None:
                        tot_gens = r_tot_gens
                    else:
                        if r_tot_gens != tot_gens:
                            print(f"{u.RED}Total gens error:{u.OFF} Run {run} " 
                                  + "has different total TPOT gens "
                                  + f"to run {run_idxs[0]} ({r_tot_gens} != " 
                                  + f"{tot_gens}) - skipping run..")
                            skipped_runs.append(run)
                            break        
        
        # read data from progress file and verify consistency with first run
        with open(fname_bo_prog, 'r') as f:
            for line in f:
                if "*" in line:
                    continue
                if "SEED" in line:
                    b_seed = int(line.split(":")[-1])
                    if b_seed != r_seed:
                        print(f"{u.RED}Seed error:{u.OFF} Run " 
                              + f"{run} has different BO seed to TPOT-BASE seed "
                              + f"({b_seed} != " 
                              + f"{r_seed}) - skipping run..")
                        skipped_runs.append(run)
                        break
                    
                if "POP SIZE" in line:
                    b_pop_size = int(line.split(":")[-1])
                    if b_pop_size != r_pop_size:
                        print(f"{u.RED}Pop size error:{u.OFF} Run {run} "
                              + "has different BO pop size to TPOT-BASE pop size " 
                              f"({b_pop_size} != {r_pop_size})" 
                              + " - skipping run..")
                        skipped_runs.append(run)
                        break
                if "STOP" in line:
                    r_stop_gen = int(line.split(":")[-1])
                    if stop_gen == None:
                        stop_gen = r_stop_gen
                    else:
                        if r_stop_gen != stop_gen:
                            print(f"{u.RED}Initial gens error:{u.OFF} Run " 
                                  + f"{run} has different initial TPOT gens "
                                  + f"to run {run_idxs[0]} ({r_stop_gen} != " 
                                  + f"{stop_gen}) - skipping run..")
                            skipped_runs.append(run)
                            break
                if "TOTAL" in line:
                    b_tot_gens = int(line.split(":")[-1])
                    if b_tot_gens != r_tot_gens:
                        print(f"{u.RED}Total gens error:{u.OFF} Run {run} " 
                              + "has different BO total TPOT gens to TPOT-BASE"
                              + f"to run {run_idxs[0]} ({r_tot_gens} != " 
                              + f"{tot_gens}) - skipping run..")
                        skipped_runs.append(run)
                        break
                if "BAYESIAN" in line:
                    r_bo_trials = int(line.split(":")[-1])
                    if bo_trials == None:
                        bo_trials = r_bo_trials
                    else:
                        if r_bo_trials != bo_trials:
                            print(f"{u.RED}BO trials error:{u.OFF} Run {run} "
                                  + "has different BO trials to run " 
                                  + f"{run_idxs[0]} ({r_bo_trials} != "
                                  + f"{bo_trials}) - skipping run..")
                            skipped_runs.append(run)
                            break
                if "DISCRETE" in line:
                    r_discrete_mode = True if line.split(":")[-1].strip() == 'True' else False
                    if discrete_mode == None:
                        discrete_mode = r_discrete_mode
                    else:
                        if r_discrete_mode != params['DISCRETE_MODE']:
                            print(f"{u.RED}Discrete mode error:{u.OFF} Run {run} "
                                  + "has different discrete mode flag to " 
                                  + f"requested ({r_discrete_mode} != "
                                  + f"{params['DISCRETE_MODE']}) - skipping run..")
                            skipped_runs.append(run)
                            break
                        
                        if r_discrete_mode != discrete_mode:
                            print(f"{u.RED}Discrete mode error:{u.OFF} Run {run} "
                                  + "has different discrete mode flag to run " 
                                  + f"{run_idxs[0]} ({r_discrete_mode} != "
                                  + f"{discrete_mode}) - skipping run..")
                            skipped_runs.append(run)
                            break
                if "RESTRICTED" in line:
                    r_restricted_hps = True if line.split(":")[-1].strip() == 'True' else False
                    if restricted_hps == None:
                        restricted_hps = r_restricted_hps
                    else:
                        if r_restricted_hps != restricted_hps:
                            print(f"{u.RED}Restricted HPs error:{u.OFF} Run {run} "
                                  + "has different restricted HPs flag to run " 
                                  + f"{run_idxs[0]} ({r_restricted_hps} != "
                                  + f"{restricted_hps}) - skipping run..")
                            skipped_runs.append(run)
                            break        
        
        if params['PLOT_TPOT-BO-ALT']:
            with open(fname_alt_prog, 'r') as f:
                for line in f:
                    if "nITERS" in line:
                        r_n_iters = int(line.split(":")[-1])
                        if n_iters == None:
                            n_iters = r_n_iters
                        else:
                            if r_n_iters != n_iters:
                                print(f"{u.RED}Number of iterations error:{u.OFF}"
                                      + f" Run {run} has different number of " 
                                      + f"iterations to run {run_idxs[0]} "
                                      + f"({r_n_iters} != {n_iters}) "
                                      + "- skipping run..")
                                skipped_runs.append(run)
                                break
                    if "TPOT GENS" in line:
                        r_alt_tpot_gens = int(line.split(":")[-1])
                        if alt_tpot_gens == None:
                            alt_tpot_gens = r_alt_tpot_gens
                        else:
                            if r_alt_tpot_gens != alt_tpot_gens:
                                print(f"{u.RED}TPOT gens error:{u.OFF} Run {run} "
                                      + "has different TPOT gens per iteration "
                                      + f"to run {run_idxs[0]} ({r_alt_tpot_gens} "
                                      + f"!= {alt_tpot_gens}) - skipping run..")
                                skipped_runs.append(run)
                                break
                    if "EVALS" in line:
                        r_alt_bo_trials = int(line.split(":")[-1])
                        if alt_bo_trials == None:
                            alt_bo_trials = r_alt_bo_trials
                        else:
                            if r_alt_bo_trials != alt_bo_trials:
                                print(f"{u.RED}BO trials error:{u.OFF} Run {run} "
                                      + "has different BO trials per iteration to " 
                                      + f"run {run_idxs[0]} ({r_alt_bo_trials} " 
                                      + f"!= {alt_bo_trials}) - skipping run..")
                                skipped_runs.append(run)
                                break
            
            if r_n_iters is None:
                print(f"{u.RED}Number of iterations error:{u.OFF}"
                      + f" Run {run} did not finish all {n_iters} ALT iterations "
                      + "- skipping run..")
        
        r_auto_gen = r_alt_bo_trials = r_auto_pop_size = None
        
        if params['PLOT_TPOT-BO-AUTO']:
            with open(fname_auto_prog, 'r') as f:
                for line in f:
                    if "GENERATION" in line:
                        r_auto_gen = int(line.split(" ")[-2])+1
                        if auto_gen == None:
                            auto_gen = r_auto_gen
                        else:
                            if r_auto_gen != auto_gen:
                                print(f"{u.RED}Number of generations error:{u.OFF}"
                                      + f" Run {run} has different number of " 
                                      + f"generations to run {run_idxs[0]} "
                                      + f"({r_auto_gen} != {auto_gen}) "
                                      + "- skipping run..")
                                skipped_runs.append(run)
                                break
                    if "POP SIZE" in line:
                        r_auto_pop_size = int(line.split(":")[-1])
                        if auto_pop_size == None:
                            auto_pop_size = r_auto_pop_size
                        else:
                            if r_auto_pop_size != auto_pop_size:
                                print(f"{u.RED}Pop size error:{u.OFF} Run {run} "
                                      + "has different pop size "
                                      + f"to run {run_idxs[0]} ({r_auto_pop_size} "
                                      + f"!= {auto_pop_size}) - skipping run..")
                                skipped_runs.append(run)
                                break
            
        if run in skipped_runs:
            continue

        add_text = "(discrete)" if discrete_mode else "(continuous)"

        # if data is consistent then store
        data[run] = {}
        data[run]['seed'] = r_seed            
            
        # get best TPOT CV values and best TPOT
        max_val = -1e40
        max_val_stop = -1e40
        init_tpot_y = np.array([])
        full_tpot_y = np.array([])
        with open(fname_tpot_pipes, 'r') as f:
            for line in f:
                split_line = line.split(";")
                val = float(split_line[2])
                if val > max_val:
                    max_val = val
                full_tpot_y = np.append(full_tpot_y, -max_val)
                if int(split_line[1]) < stop_gen:
                    if val > max_val_stop:
                        max_val_stop = val
                    init_tpot_y = np.append(init_tpot_y, -max_val)
        
        # interpolate initial tpot data between 0 and pop_size * stop_gen
        data[run]['init_tpot_y'] = np.interp(
            np.linspace(0, len(init_tpot_y), pop_size * stop_gen), 
            range(len(init_tpot_y)), init_tpot_y)
        
        # interpolate full tpot data between 0 and pop_size * tot_gens
        data[run]['full_tpot_y'] = np.interp(
            np.linspace(0, len(full_tpot_y), pop_size * tot_gens), 
            range(len(full_tpot_y)), full_tpot_y)
        
        # store max vals
        tpot_max_val = max_val
        tpot_max_val_stop = max_val_stop
        
        
        # get BO y values - this starts at the best tpot value at stop_gen 
        # because BO is initialised with matching individuals, some of which 
        # have worse cv values than the best so far
        bo_y = np.array([-max_val_stop])
        with open(fname_bo_pipes, 'r') as f:
            for line in f:
                split_line = line.split(";")
                val = float(split_line[-1])
                if val > max_val_stop:
                    max_val_stop = val
                bo_y = np.append(bo_y, -max_val_stop)
    
        # interpolate between 0 and bo_trials
        data[run]['bo_y'] = np.interp(
            np.linspace(0, len(bo_y), bo_trials), 
            range(len(bo_y)), bo_y)      
                    
        
        if params['PLOT_TPOT-BO-Sr']:
            # get BO y values - this starts at the best tpot value at stop_gen 
            # because BO is initialised with matching individuals, some of which 
            # have worse cv values than the best so far
            bo_r_y = np.array([])
            with open(fname_bo_r_pipes, 'r') as f:
                for line in f:
                    split_line = line.split(";")
                    val = float(split_line[-1])
                    if val > tpot_max_val_stop:
                        tpot_max_val_stop = val
                    bo_r_y = np.append(bo_r_y, -tpot_max_val_stop)
                
            if len(bo_r_y) == 0:
                bo_r_y = np.append(bo_r_y, -tpot_max_val_stop)
                
            # interpolate between 0 and bo_trials
            data[run]['bo_r_y'] = np.interp(
                np.linspace(0, len(bo_r_y), bo_trials), 
                range(len(bo_r_y)), bo_r_y)
    
        if params['PLOT_TPOT-BO-ALT']:
            # get best TPOT CV values and best TPOT from alt version
            best_alt_cv = -1e40
            alt_raw = {k : {i:np.array([]) for i in range(n_iters)} for k in ['TPOT-BO-ALT(TPOT)','TPOT-BO-ALT(BO)']}
            # alt_tpot_y = {i:np.array([]) for i in range(n_iters)}
            # alt_bo_y = {i:np.array([]) for i in range(n_iters)}
            with open(fname_alt_pipes, 'r') as f:
                for line in f:
                    split_line = line.split(";")
                    curr_iter = int(split_line[1])
                    gen = int(split_line[2])
                    method = split_line[3]
                    val = float(split_line[4])
                    
                    if val > best_alt_cv:
                        best_alt_cv = val
            
                    alt_raw[method][curr_iter] = np.append(alt_raw[method][curr_iter], -best_alt_cv)
            
            data[run]['alt_tpot_y'] = {}
            # interpolate initial tpot data between 0 and pop_size * stop_gen
            for k,v in alt_raw['TPOT-BO-ALT(TPOT)'].items():
                if len(v) > 0:
                    data[run]['alt_tpot_y'][k] = np.interp(
                        np.linspace(
                            0, len(v), pop_size * alt_tpot_gens), range(len(v)), v)
            
            data[run]['alt_bo_y'] = {}    
            # interpolate between 0 and bo_trials
            for k,v in alt_raw['TPOT-BO-ALT(BO)'].items():
                if len(v) > 0:
                    data[run]['alt_bo_y'][k] = np.interp(
                        np.linspace(0, len(v), alt_bo_trials), range(len(v)), v)  
    

        if params['PLOT_TPOT-BO-AUTO']:
            best_auto_cv = -1e40
            auto_y_raw = {i:np.array([]) for i in range(auto_gen)}
            types = [None for _ in range(auto_gen)]

            with open(fname_auto_pipes, 'r') as f:
                for line in f:
                    split_line = line.split(";")
                    curr_gen = int(split_line[1])
                    curr_type = split_line[2].strip()
                    val = float(split_line[3])
                    if val > best_auto_cv:
                        best_auto_cv = val

                    auto_y_raw[curr_gen] = np.append(auto_y_raw[curr_gen], -best_auto_cv)
                    types[curr_gen] = curr_type

            data[run]['auto_y'] = np.empty((0,2))
        
            # interpolate generational data between 0 and pop_size
            for k,v in auto_y_raw.items():
                if len(v) == 0:
                    print(f"generation {k} of run {run} has no pipelines recorded")
                    data[run]['auto_y'] = np.vstack((data[run]['auto_y'],np.ones((pop_size,2)) * data[run]['auto_y'][-1,:]))      
                    continue
                tmp = np.interp(np.linspace(0, len(v), pop_size), range(len(v)), v)
                
                data[run]['auto_y'] = np.vstack((data[run]['auto_y'],np.hstack((tmp.reshape(-1,1), (types[k] == 'TPOT-BO-AUTO(TPOT)') * np.ones((pop_size,1))))))      
                
    ###################
    
    # remove all skipped runs from run_idxs
    for run in skipped_runs:
        run_idxs.pop(run_idxs.index(run))
    
    # get tpot y data until stop gen
    init_tpot_y_mu = np.array([np.mean(
        [data[run]['init_tpot_y'][i] for run in run_idxs]) 
        for i in range(pop_size * stop_gen)])
    init_tpot_y_sigma = np.array(
        [np.std([data[run]['init_tpot_y'][i] for run in run_idxs]) 
         for i in range(pop_size * stop_gen)])
    init_tpot_y_best = np.min(
        [np.min([data[run]['init_tpot_y'][i] for run in run_idxs]) 
         for i in range(pop_size * stop_gen)])
    init_tpot_y_b = np.array(
        [np.min([data[run]['init_tpot_y'][i] for run in run_idxs]) 
         for i in range(pop_size * stop_gen)])
    init_tpot_y_w = np.array(
        [np.max([data[run]['init_tpot_y'][i] for run in run_idxs]) 
         for i in range(pop_size * stop_gen)])
    
    # get full tpot y data
    full_tpot_y_mu = np.array(
        [np.mean([data[run]['full_tpot_y'][i] for run in run_idxs]) 
         for i in range(pop_size * tot_gens)])
    full_tpot_y_sigma = np.array(
        [np.std([data[run]['full_tpot_y'][i] for run in run_idxs]) 
         for i in range(pop_size * tot_gens)])
    full_tpot_y_best = np.min(
        [np.min([data[run]['full_tpot_y'][i] for run in run_idxs]) 
         for i in range(pop_size * tot_gens)])
    full_tpot_y_b = np.array(
        [np.min([data[run]['full_tpot_y'][i] for run in run_idxs]) 
         for i in range(pop_size * tot_gens)])
    full_tpot_y_w = np.array(
        [np.max([data[run]['full_tpot_y'][i] for run in run_idxs]) 
         for i in range(pop_size * tot_gens)])
    
    # get BO y data
    bo_y_mu = np.array([np.mean([data[run]['bo_y'][i] for run in run_idxs]) 
                        for i in range(bo_trials)])
    bo_y_sigma = np.array([np.std([data[run]['bo_y'][i] for run in run_idxs]) 
                           for i in range(bo_trials)])
    bo_y_best = np.min([np.min([data[run]['bo_y'][i] for run in run_idxs]) 
                        for i in range(bo_trials)])
    bo_y_b = np.array([np.min([data[run]['bo_y'][i] for run in run_idxs]) 
                       for i in range(bo_trials)])
    bo_y_w = np.array([np.max([data[run]['bo_y'][i] for run in run_idxs]) 
                       for i in range(bo_trials)])
    
    if params['PLOT_TPOT-BO-Sr']:
        # get BO y data
        bo_r_y_mu = np.array([np.mean([data[run]['bo_r_y'][i] for run in run_idxs]) 
                            for i in range(bo_trials)])
        bo_r_y_sigma = np.array([np.std([data[run]['bo_r_y'][i] for run in run_idxs]) 
                               for i in range(bo_trials)])
        bo_r_y_best = np.min([np.min([data[run]['bo_r_y'][i] for run in run_idxs]) 
                            for i in range(bo_trials)])
        bo_r_y_b = np.array([np.min([data[run]['bo_r_y'][i] for run in run_idxs]) 
                           for i in range(bo_trials)])
        bo_r_y_w = np.array([np.max([data[run]['bo_r_y'][i] for run in run_idxs]) 
                           for i in range(bo_trials)])
    
    mean_text = " (mean)"
    if len(run_idxs) == 1:
        mean_text = ""
    
    
    if params['PLOT_TPOT-BO-ALT']:
    
        # get y data for alt version
        alt_tpot_y_mu = {k:np.array(
            [np.mean([data[run]['alt_tpot_y'][k][i] for run in run_idxs]) 
             for i in range(pop_size * alt_tpot_gens)]) 
            for k in range(len(data[run]['alt_tpot_y']))}
        alt_tpot_y_sigma = {k:np.array(
            [np.std([data[run]['alt_tpot_y'][k][i] for run in run_idxs]) 
             for i in range(pop_size * alt_tpot_gens)]) 
            for k in range(len(data[run]['alt_tpot_y']))}
        alt_tpot_y_b = {k:np.array(
            [np.min([data[run]['alt_tpot_y'][k][i] for run in run_idxs]) 
             for i in range(pop_size * alt_tpot_gens)]) 
            for k in range(len(data[run]['alt_tpot_y']))}
        alt_tpot_y_w = {k:np.array(
            [np.max([data[run]['alt_tpot_y'][k][i] for run in run_idxs]) 
             for i in range(pop_size * alt_tpot_gens)]) 
            for k in range(len(data[run]['alt_tpot_y']))}
        alt_tpot_y_best = {k:np.min(
            [np.min([data[run]['alt_tpot_y'][k][i] for run in run_idxs]) 
             for i in range(pop_size * alt_tpot_gens)]) 
            for k in range(len(data[run]['alt_tpot_y']))}
    
        
        # get BO y data
        alt_bo_y_mu = {k:np.array([np.mean(
            [data[run]['alt_bo_y'][k][i] for run in run_idxs]) 
            for i in range(alt_bo_trials)]) 
            for k in range(len(data[run]['alt_bo_y']))}
        alt_bo_y_sigma = {k:np.array([np.std(
            [data[run]['alt_bo_y'][k][i] for run in run_idxs]) 
            for i in range(alt_bo_trials)]) 
            for k in range(len(data[run]['alt_bo_y']))}
        alt_bo_y_b = {k:np.array([np.min(
            [data[run]['alt_bo_y'][k][i] for run in run_idxs]) 
            for i in range(alt_bo_trials)]) 
            for k in range(len(data[run]['alt_bo_y']))}
        alt_bo_y_w = {k:np.array([np.max(
            [data[run]['alt_bo_y'][k][i] for run in run_idxs]) 
            for i in range(alt_bo_trials)]) 
            for k in range(len(data[run]['alt_bo_y']))}
        alt_bo_y_best = {k:np.min([np.min(
            [data[run]['alt_bo_y'][k][i] for run in run_idxs]) 
            for i in range(alt_bo_trials)]) 
            for k in range(len(data[run]['alt_bo_y']))}
    
    
    if params['PLOT_TPOT-BO-AUTO']:
        # get mean data
        auto_y_mu = np.mean(np.array([data[run]['auto_y'] for run in run_idxs]),axis=0)
        auto_y_sigma = np.std(np.array([data[run]['auto_y'] for run in run_idxs]),axis=0)
    
    
    # max min mean median and std of TPOT and TPOT + BO
    stats[problem] = {'runs': run_idxs, 'best':{}, 'worst': {}, 
                      'median': {}, 'mean': {},'std dev': {}}    
    stats[problem]['best']['tpot'] = np.min(
        [data[run]['full_tpot_y'][-1] for run in run_idxs])
    stats[problem]['best']['bo'] = np.min(
        [data[run]['bo_y'][-1] for run in run_idxs])
    stats[problem]['worst']['tpot'] = np.max(
        [data[run]['full_tpot_y'][-1] for run in run_idxs])
    stats[problem]['worst']['bo'] = np.max(
        [data[run]['bo_y'][-1] for run in run_idxs])
    stats[problem]['median']['tpot'] = np.median(
        [data[run]['full_tpot_y'][-1] for run in run_idxs])
    stats[problem]['median']['bo'] = np.median(
        [data[run]['bo_y'][-1] for run in run_idxs])
    stats[problem]['mean']['tpot']= np.mean(
        [data[run]['full_tpot_y'][-1] for run in run_idxs])
    stats[problem]['mean']['bo'] = np.mean(
        [data[run]['bo_y'][-1] for run in run_idxs])
    stats[problem]['std dev']['tpot'] = full_tpot_y_sigma[-1]
    stats[problem]['std dev']['bo'] = bo_y_sigma[-1]
    
    if params['PLOT_TPOT-BO-Sr']:
        stats[problem]['best']['bo_r'] = np.min([data[run]['bo_r_y'][-1] for run in run_idxs])
        stats[problem]['worst']['bo_r'] = np.max([data[run]['bo_r_y'][-1] for run in run_idxs])
        stats[problem]['median']['bo_r'] = np.median([data[run]['bo_r_y'][-1] for run in run_idxs])
        stats[problem]['mean']['bo_r'] = np.mean([data[run]['bo_r_y'][-1] for run in run_idxs])
        stats[problem]['std dev']['bo_r'] = bo_r_y_sigma[-1]
        
    
    if params['PLOT_TPOT-BO-ALT']:
        stats[problem]['best']['alt'] = (
            alt_bo_y_b[len(data[run_idxs[-1]]['alt_bo_y'])-1][-1])
        stats[problem]['worst']['alt'] = (
            alt_bo_y_w[len(data[run_idxs[-1]]['alt_bo_y'])-1][-1])
        stats[problem]['mean']['alt'] = (
            alt_bo_y_mu[len(data[run_idxs[-1]]['alt_bo_y'])-1][-1])
        stats[problem]['std dev']['alt'] = (
            alt_bo_y_sigma[len(data[run_idxs[-1]]['alt_bo_y'])-1][-1])
        stats[problem]['median']['alt'] = np.median(
            [data[run]['alt_bo_y'][len(data[run_idxs[-1]])-1][-1] 
             for run in run_idxs])
    
    if params['PLOT_TPOT-BO-AUTO']:
        obj_vals = np.array([data[run]['auto_y'][-1,0] for run in run_idxs])
        best_auto_run_idx = np.argmin(obj_vals)
        best_auto_run = run_idxs[best_auto_run_idx]
        worst_auto_run_idx = np.argmax(obj_vals)
        worst_auto_run = run_idxs[worst_auto_run_idx]
        # find median run index (closest to median value)
        med_auto_run_idx = np.abs(obj_vals - np.median(obj_vals)).argmin()
        med_auto_run = run_idxs[med_auto_run_idx]
        
        stats[problem]['best']['auto'] = obj_vals[best_auto_run_idx]
        stats[problem]['worst']['auto'] = obj_vals[worst_auto_run_idx]
        stats[problem]['mean']['auto'] = auto_y_mu[-1,0]
        stats[problem]['std dev']['auto'] = auto_y_sigma[-1,0]
        stats[problem]['median']['auto'] = obj_vals[med_auto_run_idx]
    
    # compute plot limits
    y_mu_start = full_tpot_y_mu[params['SKIP_PLOT_INIT']]
    y_mu_end = bo_y_mu[-1]
    if len(run_idxs) > 1:
        ylim_min = y_mu_end - 1.5*bo_y_sigma[-1]
    else:
        ylim_min = y_mu_end - (y_mu_start-y_mu_end)
        
    ylim_max = y_mu_start
    
    if params['PLOT_TPOT-BO-ALT'] and stats[problem]['mean']['alt'] < y_mu_end:
        y_mu_end = stats[problem]['mean']['alt']      
        ylim_min = y_mu_end - 1.5*stats[problem]['std dev']['alt']
    
    # plot TPOT only data (mu/sigma)
    fig2, ax_end_y_s = plt.subplots()
    ax_end_y_s.fill_between(range(len(full_tpot_y_mu)), 
                            full_tpot_y_mu - full_tpot_y_sigma, 
                            full_tpot_y_mu + full_tpot_y_sigma, 
                            alpha=.5, linewidth=0)
    ax_end_y_s.plot(range(len(full_tpot_y_mu)), 
                    full_tpot_y_mu, linewidth=2,
                    label='TPOT evaluation'+mean_text)
    ax_end_y_s.legend()
    title_text = (f"{problem} - TPOT-BASE\n" 
                  + f"μ: {full_tpot_y_mu[-1]:.6e}, "
                  + f"σ: {full_tpot_y_sigma[-1]:.4e}")
    ax_end_y_s.set_title(title_text)
    ax_end_y_s.set_ylim([ylim_min,ylim_max])
    ax_end_y_s.set_xlabel("Evaluations")
    ax_end_y_s.set_ylabel("CV")
    plt.show()
    
    # plot TPOT and BO data (mu/sigma)
    fig4, ax_tpot_bo_y_s = plt.subplots()
    ax_tpot_bo_y_s.fill_between(range(len(init_tpot_y_mu)), 
                                init_tpot_y_mu - init_tpot_y_sigma, 
                                init_tpot_y_mu + init_tpot_y_sigma, 
                                alpha=.5, linewidth=0)
    ax_tpot_bo_y_s.plot(range(1,len(init_tpot_y_mu)+1), 
                        init_tpot_y_mu, linewidth=2,
                        label='TPOT evaluation'+mean_text)
    ax_tpot_bo_y_s.fill_between(range(len(init_tpot_y_mu),
                                      len(init_tpot_y_mu)+len(bo_y_mu)), 
                                bo_y_mu - bo_y_sigma, bo_y_mu + bo_y_sigma, 
                                alpha=.5, linewidth=0,color='red')
    ax_tpot_bo_y_s.plot(range(len(init_tpot_y_mu),
                              len(init_tpot_y_mu)+len(bo_y_mu)), 
                        bo_y_mu, linewidth=2,
                        label='BO evaluation'+mean_text,color='red')
    ax_tpot_bo_y_s.legend()
    title_text = (f"{problem} - TPOT-BO-S {add_text}\n"
                  + f"μ: {bo_y_mu[-1]:.6e}, "
                  + f"σ: {bo_y_sigma[-1]:.4e}")
    ax_tpot_bo_y_s.set_title(title_text)
    ax_tpot_bo_y_s.set_ylim([ylim_min,ylim_max])
    ax_tpot_bo_y_s.set_xlabel("Evaluations")
    ax_tpot_bo_y_s.set_ylabel("CV")
    plt.show()
    

    if params['PLOT_TPOT-BO-Sr']:
        # plot TPOT and BO data (mu/sigma)
        fig13, ax_tpot_bo_r_y_s = plt.subplots()
        ax_tpot_bo_r_y_s.fill_between(range(len(init_tpot_y_mu)), 
                                    init_tpot_y_mu - init_tpot_y_sigma, 
                                    init_tpot_y_mu + init_tpot_y_sigma, 
                                    alpha=.5, linewidth=0)
        ax_tpot_bo_r_y_s.plot(range(1,len(init_tpot_y_mu)+1), 
                            init_tpot_y_mu, linewidth=2,
                            label='TPOT evaluation'+mean_text)
        ax_tpot_bo_r_y_s.fill_between(range(len(init_tpot_y_mu),
                                          len(init_tpot_y_mu)+len(bo_r_y_mu)), 
                                    bo_r_y_mu - bo_r_y_sigma, bo_r_y_mu + bo_r_y_sigma, 
                                    alpha=.5, linewidth=0,color='red')
        ax_tpot_bo_r_y_s.plot(range(len(init_tpot_y_mu),
                                  len(init_tpot_y_mu)+len(bo_r_y_mu)), 
                            bo_r_y_mu, linewidth=2,
                            label='BO evaluation'+mean_text,color='red')
        ax_tpot_bo_r_y_s.legend()
        title_text = (f"{problem} - TPOT-BO-Sr {add_text}\n"
                      + f"μ: {bo_r_y_mu[-1]:.6e}, "
                      + f"σ: {bo_r_y_sigma[-1]:.4e}")
        ax_tpot_bo_r_y_s.set_title(title_text)
        ax_tpot_bo_r_y_s.set_ylim([ylim_min,ylim_max])
        ax_tpot_bo_r_y_s.set_xlabel("Evaluations")
        ax_tpot_bo_r_y_s.set_ylabel("CV")
        plt.show()

    # plot alt results
    # compute plot limits
    # take max from the worst of the first bo iteration (excludes initial tpot)
    
    if params['PLOT_TPOT-BO-ALT']:
        # plot alt results (mu/sigma)
        fig10, ax_alt_tpot_bo_s = plt.subplots()
        alt_tpot_lines_s = {}
        alt_bo_lines_s = {}
        for i in range(len(data[run_idxs[-1]]['alt_bo_y'])):
            alt_tpot_start = i * pop_size * alt_tpot_gens + i * alt_bo_trials
            alt_bo_start = ((i+1) * pop_size * alt_tpot_gens 
                            + ((i>0) * (i)) * alt_bo_trials)
            ax_alt_tpot_bo_s.fill_between(
                range(alt_tpot_start, alt_tpot_start + len(alt_tpot_y_mu[i])), 
                alt_tpot_y_mu[i] + alt_tpot_y_sigma[i], 
                alt_tpot_y_mu[i] - alt_tpot_y_sigma[i], alpha=.5, 
                linewidth=0,color='C0')
            alt_tpot_lines_s[i], = ax_alt_tpot_bo_s.plot(
                range(alt_tpot_start, alt_tpot_start + len(alt_tpot_y_mu[i])), 
                alt_tpot_y_mu[i], linewidth=2,
                label='TPOT evaluation'+mean_text,color='C0')
            ax_alt_tpot_bo_s.fill_between(
                range(alt_bo_start, alt_bo_start + len(alt_bo_y_mu[i])), 
                alt_bo_y_mu[i] + alt_bo_y_sigma[i], 
                alt_bo_y_mu[i] - alt_bo_y_sigma[i], 
                alpha=.5, linewidth=0,color='r')
            alt_bo_lines_s[i], = ax_alt_tpot_bo_s.plot(
                range(alt_bo_start, alt_bo_start + len(alt_bo_y_mu[i])), 
                alt_bo_y_mu[i], linewidth=2,
                label='BO evaluation'+mean_text,color='r')
        ax_alt_tpot_bo_s.legend(handles=[alt_tpot_lines_s[0], alt_bo_lines_s[0]])
            
        alt_title_text_s = (f"{problem} - TPOT-BO-ALT {add_text}\n"
                        + f"μ: {alt_bo_y_mu[len(data[run_idxs[-1]]['alt_bo_y'])-1][-1]:.6e}, "
                        + f"σ: {alt_bo_y_sigma[len(data[run_idxs[-1]]['alt_bo_y'])-1][-1]:.4e}")
        ax_alt_tpot_bo_s.set_title(alt_title_text_s)
        ax_alt_tpot_bo_s.set_ylim([ylim_min, ylim_max])
        ax_alt_tpot_bo_s.set_xlabel("Evaluations")
        ax_alt_tpot_bo_s.set_ylabel("CV")
    
    plt.show()
    
    if params['PLOT_TPOT-BO-AUTO']:                
        # plot alt results (median)
        fig12, ax_auto_m = plt.subplots()
        for i in range(2*pop_size, data[med_auto_run]['auto_y'].shape[0], pop_size):
            colour = "red" if data[med_auto_run]['auto_y'][i,1] == 1 else "C0"
            ax_auto_m.plot(range(i,i+pop_size),data[med_auto_run]['auto_y'][i:i+pop_size,0], c=colour,lw=4)
        tpot_line = Line2D([0], [0], label="TPOT evaluation", color='C0', linewidth=3)
        bo_line = Line2D([0], [0], label="BO evaluation", color='red', linewidth=3)
        labels = ['TPOT evaluation', 'BO evaluation']
        ax_auto_m.legend(handles=[tpot_line, bo_line])
        
        auto_title_text_m = (f"{problem} - TPOT-BO-AUTO {add_text}\n"
                        + f"median run: {med_auto_run}, "
                        +f"obj: {data[med_auto_run]['auto_y'][-1,0]:.6e}")
        ax_auto_m.set_title(auto_title_text_m)
        # ax_auto_m.set_ylim([ylim_min, ylim_max])
        ax_auto_m.set_xlabel("Evaluations")
        ax_auto_m.set_ylabel("CV")      
        
    plt.show()           
    
    
    res_txt = "_(res)" if params['PLOT_TPOT-BO-Sr'] else ""
    
    if params['SAVE_PLOTS']:
        d_flag = "D" if params['DISCRETE_MODE'] else "C"
        plot_path = os.path.join(prob_path, "Plots")
        
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        fname_tpot_plot_s = os.path.join(
            plot_path, problem + f"_TPOT-BASE_{d_flag}.png")
        fname_tpot_bo_plot_s = os.path.join(
            plot_path, problem + f"_TPOT-BO_S_{d_flag}.png")
        if params['PLOT_TPOT-BO-ALT']:
            fname_alt_plot_s = os.path.join(
                plot_path, problem + f"_TPOT-BO-ALT_{d_flag}.png")
        
        if params['PLOT_TPOT-BO-AUTO']:
            fname_auto_plot_s = os.path.join(
                plot_path, problem + f"_TPOT-BO-AUTO_{d_flag}.png")
                
        if params['PLOT_TPOT-BO-Sr']:
            fname_bo_r_plot_s = os.path.join(
                plot_path, problem + f"_TPOT-BO-Sr_{d_flag}.png")
        
        fig2.savefig(fname_tpot_plot_s,bbox_inches='tight')
        fig4.savefig(fname_tpot_bo_plot_s,bbox_inches='tight')

        if params['PLOT_TPOT-BO-ALT']:
            fig10.savefig(fname_alt_plot_s,bbox_inches='tight')
        
        if params['PLOT_TPOT-BO-AUTO']:
            fig12.savefig(fname_auto_plot_s,bbox_inches='tight')
            
        if params['PLOT_TPOT-BO-Sr']:
            fig13.savefig(fname_bo_r_plot_s,bbox_inches='tight')
            
for problem in prob_list:
    print(f"\n{u.CYAN}{problem} {add_text} statistics:{u.OFF}")
    print(f"{str(''):>{PRINT_COL}}{str('TPOT-BASE'):>{PRINT_COL}}{str('TPOT-BO-S'):>{PRINT_COL}}",end="")
    if params['PLOT_TPOT-BO-Sr']:
        print(f"{str('TPOT-BO-Sr'):>{PRINT_COL}}",end="")
    if params['PLOT_TPOT-BO-ALT']:
        print(f"{str('TPOT-BO-ALT'):>{PRINT_COL}}",end="")
    if params['PLOT_TPOT-BO-AUTO']:
        print(f"{str('TPOT-BO-AUTO'):>{PRINT_COL}}",end="")
        
    print()
    for _ in range((3 + params['PLOT_TPOT-BO-Sr'] + params['PLOT_TPOT-BO-ALT'] + params['PLOT_TPOT-BO-AUTO'])*PRINT_COL):
        print("=",end='')
    
    print()
    
    for stat,methods in stats[problem].items():
        if stat == 'runs':
            continue
        print(f"{str(stat):>{PRINT_COL}}",end="")
        for method,val in methods.items():
            print(f"{val:>{PRINT_COL}.6e}",end="")
        print()
            
if params['SAVE_STATS']:
    fname_stats = os.path.join(results_path, f"BO-TPOT.stats")
    with open(fname_stats, 'a') as f:
        f.write(f"\n%TIME: {time.asctime()}\n\n")
        # print statistics
        for problem in prob_list:
            f.write(f"***** {problem} {add_text} *****\n")
            f.write(f"!RUN LIST:{stats[problem]['runs']}\n")
            f.write(f"{str('#'):<{PRINT_COL}};{str('TPOT-BASE'):>{PRINT_COL}};{str('TPOT-BO-S'):>{PRINT_COL}};")
            if params['PLOT_TPOT-BO-Sr']:
                f.write(f"{str('TPOT-BO-Sr'):>{PRINT_COL}};")
            if params['PLOT_TPOT-BO-ALT']:
                f.write(f"{str('TPOT-BO-ALT'):>{PRINT_COL}};")
            if params['PLOT_TPOT-BO-AUTO']:
                f.write(f"{str('TPOT-BO-AUTO'):>{PRINT_COL}};")
            f.write("\n")
            
            for stat,methods in stats[problem].items():
                if stat == 'runs':
                    continue
                f.write(f"{str(stat):>{PRINT_COL}};")
                for method,val in methods.items():
                    f.write(f"{val:>{PRINT_COL}.6e};")
                f.write("\n")
            f.write("\n")
    
