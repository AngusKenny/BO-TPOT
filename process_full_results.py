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
import utils as u
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
    'RESULTS_DIR'   : 'Results_discrete',
    'PROBLEMS'      : ['brazilian_houses','elevators'],
    'RUN_LIST'      : range(21),
    'SAVE_PLOTS'    : False,
    'SAVE_STATS'    : False,
    'PLOT_ALT'      : True,
    'PLOT_AUTO'      :False,
    'PLOT_MIN_MAX'  : False,
    'SKIP_PLOT_INIT': 10,
    'ADD_TITLE_TEXT': '(discrete)',
    'COLOURMAP'     : plt.cm.bwr
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

# in case ADD_TITLE_TEXT is not defined
add_text = params['ADD_TITLE_TEXT'] if 'ADD_TITLE_TEXT' in params else ""

# iterate over problem list
for problem in prob_list:
    
    prob_dir = os.path.join(results_path, problem)

    print(f"Processing results from {prob_dir}")
    
    if len(params['RUN_LIST']) == 0:
        run_idxs = [int(d.path.split("_")[-1]) 
                    for d in os.scandir(prob_dir) 
                    if d.is_dir() and "Plots" not in d.path]
        run_idxs.sort()
    else:
        run_idxs = params['RUN_LIST']
    
    pop_size = stop_gen = tot_gens = bo_trials = None
    alt_tpot_gens = alt_bo_trials = n_iters = None
    auto_pop_size = None
    
    data = {}
    skipped_runs = []
    
    # validate and collect data from specified runs
    for run in run_idxs:
        run_str = str(run)    
        if run < 10:
            run_str = "0" + str(run)
            
        run_dir = os.path.join(prob_dir,"Run_" + run_str)
        
        tpot_dir = os.path.join(run_dir, 'tpot')
        bo_dir = os.path.join(run_dir, 'bo')
        alt_dir = os.path.join(run_dir, 'alt')
        auto_dir = os.path.join(run_dir, 'auto')
        
        fname_tpot_prog = os.path.join(tpot_dir, "tpot_progress.out")
        fname_tpot_pipes = os.path.join(tpot_dir, "tpot_pipes.out")
        fname_matching_pipes = os.path.join(tpot_dir, "matching_pipes.out") 
        
        fname_bo_pipes = os.path.join(bo_dir, "bo_pipes.out")
        
        fname_alt_prog = os.path.join(alt_dir, "alt_progress.out")
        fname_alt_tpot_pipes = os.path.join(alt_dir, "alt_tpot_pipes.out")
        fname_alt_bo_pipes = os.path.join(alt_dir, "alt_bo_pipes.out")
        
        fname_auto_prog = os.path.join(auto_dir, "auto_progress.out")
        fname_auto_pipes = os.path.join(auto_dir, "auto_pipes.out")
        fname_auto_grads = os.path.join(auto_dir, "auto_grads.out")
        
        
        check_files = [fname_tpot_prog, fname_tpot_pipes, 
                      fname_bo_pipes, fname_matching_pipes]
        
        if params['PLOT_ALT']:
            check_files = check_files + [fname_alt_prog, 
                                         fname_alt_tpot_pipes, 
                                         fname_alt_bo_pipes]
            
        if params['PLOT_AUTO']:
            check_files = check_files + [fname_auto_prog, 
                                         fname_auto_pipes, 
                                         fname_auto_grads]
        
        # check if correct files exist if not then skip run
        for fpath in check_files:
            if not os.path.exists(fpath):
                print(f"{u.RED}Missing file error:{u.OFF} Run {run} of " 
                      + f"{problem} is missing file {os.path.basename(fpath)}"
                      + " - skipping run..")
                skipped_runs.append(run)
                break
        
        if run in skipped_runs:
            continue
        
        r_seed = r_stop_gen = r_tot_gens = r_bo_trials = None
        r_alt_tpot_gens = r_alt_bo_trials = r_n_iters = None
        
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
        
        r_max_iter = -1
        
        if params['PLOT_ALT']:
            with open(fname_alt_prog, 'r') as f:
                for line in f:
                    if "ITERATION" in line:
                        r_iter = int(line.split(" ")[-2])
                        if  r_iter > r_max_iter:
                            r_max_iter = r_iter
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
                    if "BO TRIALS" in line or "EVALS" in line:
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
            
            if r_max_iter < n_iters-1:
                print(f"{u.RED}Iterations error:{u.OFF} Run {run} crashed after "
                      + f"{r_max_iter+1} of {n_iters} iterations - skipping run..")
                skipped_runs.append(run)
                continue
        
        r_gen = r_alt_bo_trials = r_auto_pop_size = None
        r_max_gen = -1
        
        if params['PLOT_AUTO']:
            with open(fname_auto_prog, 'r') as f:
                for line in f:
                    if "GENERATION" in line:
                        r_gen = int(line.split(" ")[-2])
                        if r_gen > r_max_gen:
                            r_max_gen = r_gen
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
            
            if r_max_gen < tot_gens-1:
                print(f"{u.RED}Generations error:{u.OFF} Run {run} crashed after "
                      + f"{r_max_gen+1} of {tot_gens} iterations - skipping run..")
                skipped_runs.append(run)
                continue
        
        
        if run in skipped_runs:
            continue
        
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
    
        # get BO y values - this starts at the best tpot value at stop_gen 
        # because BO is initialised with matching individuals, some of which 
        # have worse cv values than the best so far
        bo_y = np.array([])
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
    
        # get data for matching pipelines (up until stop_gen)
        # include best from tpot pipes (not in matching pipes file)
        data[run]['matching'] = np.array([-max_val_stop]) 
        with open(fname_matching_pipes, 'r') as f:
            for line in f:
                split_line = line.split(";")
                val = float(split_line[2])
                if int(split_line[1]) < stop_gen:
                    data[run]['matching'] = np.append(
                        data[run]['matching'], -val)        
    
        if params['PLOT_ALT']:
            # get best TPOT CV values and best TPOT from alt version
            best_alt_cv = -1e40
            # for BO starting positions
            best_alt_bo_cvs = [-1e40 for i in range(n_iters)] 
            alt_tpot_y = {i:np.array([]) for i in range(n_iters)}
            with open(fname_alt_tpot_pipes, 'r') as f:
                for line in f:
                    split_line = line.split(";")
                    curr_iter = int(split_line[1])
                    val = float(split_line[3])
                    if val > best_alt_cv:
                        best_alt_cv = val
                        best_alt_bo_cvs[curr_iter] = val
                    alt_tpot_y[curr_iter] = np.append(
                        alt_tpot_y[curr_iter], -best_alt_cv)
            
            data[run]['alt_tpot_y'] = {}
            # interpolate initial tpot data between 0 and pop_size * stop_gen
            for k,v in alt_tpot_y.items():
                if len(v) > 0:
                    data[run]['alt_tpot_y'][k] = np.interp(
                        np.linspace(
                            0, len(v), pop_size * alt_tpot_gens), range(len(v)), v)
            
            # get BO y values - this starts at the best tpot value at stop_gen because
            # BO is initialised with matching individuals, some of which have worse cv
            # values than the best so far
            alt_bo_y = {i:np.array([]) for i in range(n_iters)}
            with open(fname_alt_bo_pipes, 'r') as f:
                for line in f:
                    split_line = line.split(";")
                    curr_iter = int(split_line[1])
                    val = float(split_line[2])
                    if val > best_alt_bo_cvs[curr_iter]:
                        best_alt_bo_cvs[curr_iter] = val
                    alt_bo_y[curr_iter] = np.append(
                        alt_bo_y[curr_iter], -best_alt_bo_cvs[curr_iter])
        
            data[run]['alt_bo_y'] = {}    
            # interpolate between 0 and bo_trials
            for k,v in alt_bo_y.items():
                if len(v) > 0:
                    data[run]['alt_bo_y'][k] = np.interp(
                        np.linspace(0, len(v), alt_bo_trials), range(len(v)), v)   
    
    #############
        if params['PLOT_AUTO']:
            # get best TPOT CV values and best TPOT from alt version
            best_auto_cv = -1e40
            auto_y_raw = {i:np.array([]) for i in range(auto_pop_size)}
            types = [None for _ in range(auto_pop_size)]

            with open(fname_auto_pipes, 'r') as f:
                for line in f:
                    split_line = line.split(";")
                    curr_gen = int(split_line[1])
                    val = float(split_line[2])
                    curr_type = split_line[3].strip()

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
                
                data[run]['auto_y'] = np.vstack((data[run]['auto_y'],np.hstack((tmp.reshape(-1,1), (types[k] == 'tpot') * np.ones((pop_size,1))))))      
                
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
    
    # print([data[run]['full_tpot_y'][0] for run in run_idxs])
    
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
    
    mean_text = " (mean)"
    if len(run_idxs) == 1:
        mean_text = ""
    
    matching_data = np.empty((0,2))
    for run in run_idxs:
        for i in range(len(data[run]['matching'])):
            matching_data = np.vstack(
                (matching_data,[run, data[run]['matching'][i]]))
    
    if params['PLOT_ALT']:
    
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
    
    
    if params['PLOT_AUTO']:
        # get mean data
        auto_y_mu = np.mean(np.array([data[run]['auto_y'] for run in run_idxs]),axis=0)
        auto_y_sigma = np.std(np.array([data[run]['auto_y'] for run in run_idxs]),axis=0)
    
    
    # max min mean median and std of TPOT and TPOT + BO
    stats[problem] = {'runs': run_idxs, 'best':{}, 'worst': {}, 
                      'median': {}, 'mu': {},'sigma': {}}    
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
    stats[problem]['mu']['tpot']= np.mean(
        [data[run]['full_tpot_y'][-1] for run in run_idxs])
    stats[problem]['mu']['bo'] = np.mean(
        [data[run]['bo_y'][-1] for run in run_idxs])
    stats[problem]['sigma']['tpot'] = full_tpot_y_sigma[-1]
    stats[problem]['sigma']['bo'] = bo_y_sigma[-1]
    
    if params['PLOT_ALT']:
        stats[problem]['best']['alt'] = (
            alt_bo_y_b[len(data[run_idxs[-1]]['alt_bo_y'])-1][-1])
        stats[problem]['worst']['alt'] = (
            alt_bo_y_w[len(data[run_idxs[-1]]['alt_bo_y'])-1][-1])
        stats[problem]['mu']['alt'] = (
            alt_bo_y_mu[len(data[run_idxs[-1]]['alt_bo_y'])-1][-1])
        stats[problem]['sigma']['alt'] = (
            alt_bo_y_sigma[len(data[run_idxs[-1]]['alt_bo_y'])-1][-1])
        stats[problem]['median']['alt'] = np.median(
            [data[run]['alt_bo_y'][len(data[run_idxs[-1]])-1][-1] 
             for run in run_idxs])
    
    if params['PLOT_AUTO']:
        obj_vals = np.array([data[run]['auto_y'][-1,0] for run in run_idxs])
        best_auto_run = run_idxs[np.argmin(obj_vals)]
        worst_auto_run = run_idxs[np.argmax(obj_vals)]
        # find median run index (closest to median value)
        med_auto_run = run_idxs[np.abs(obj_vals - np.median(obj_vals)).argmin()]
        
        stats[problem]['best']['auto'] = obj_vals[best_auto_run]
        stats[problem]['worst']['auto'] = obj_vals[worst_auto_run]
        stats[problem]['mu']['auto'] = auto_y_mu[-1,0]
        stats[problem]['sigma']['auto'] = auto_y_sigma[-1,0]
        stats[problem]['median']['auto'] = obj_vals[med_auto_run]
    
    # compute plot limits
    y_mu_start = full_tpot_y_mu[params['SKIP_PLOT_INIT']]
    y_mu_end = bo_y_mu[-1]
    ylim_min = y_mu_end - 1.5*bo_y_sigma[-1]
    ylim_max = y_mu_start
    
    if params['PLOT_ALT']:
        y_mu_end = stats[problem]['mu']['alt']      
        ylim_min = y_mu_end - 1.5*stats[problem]['sigma']['alt']
    
    if params['PLOT_MIN_MAX']:
        # plot TPOT only data (min/max)
        fig1, ax_end_y = plt.subplots()
        ax_end_y.fill_between(range(len(full_tpot_y_mu)), 
                              full_tpot_y_w, full_tpot_y_b, alpha=.5, linewidth=0)
        ax_end_y.plot(range(len(full_tpot_y_mu)), full_tpot_y_mu, 
                      linewidth=2, label='TPOT evaluation'+mean_text)
        ax_end_y.legend()
        title_text = (f"{problem} - TPOT only [min/max]\n" 
                      + f"μ: {round(full_tpot_y_mu[-1],4)}, "
                      + f"min: {round(full_tpot_y_b[-1],4)}, "
                      + f"max: {round(full_tpot_y_w[-1],4)}")
        ax_end_y.set_title(title_text)
        ax_end_y.set_ylim([ylim_min, ylim_max])
        ax_end_y.set_xlabel("Evaluations")
        ax_end_y.set_ylabel("CV")
        plt.show()
    
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
    title_text = (f"{problem} - TPOT only\n" 
                  + f"μ: {round(full_tpot_y_mu[-1],4)}, "
                  + f"σ: {full_tpot_y_sigma[-1]:.3e}")
    ax_end_y_s.set_title(title_text)
    ax_end_y_s.set_ylim([ylim_min,ylim_max])
    ax_end_y_s.set_xlabel("Evaluations")
    ax_end_y_s.set_ylabel("CV")
    plt.show()
    
    
    if params['PLOT_MIN_MAX']:
        # plot TPOT and BO data (min/max)
        fig3, ax_tpot_bo_y = plt.subplots()
        ax_tpot_bo_y.fill_between(range(1,len(init_tpot_y_mu)+1), 
                                  init_tpot_y_w, init_tpot_y_b, 
                                  alpha=.5, linewidth=0)
        ax_tpot_bo_y.plot(range(1,len(init_tpot_y_mu)+1), 
                          init_tpot_y_mu, linewidth=2,
                          label='TPOT evaluation'+mean_text)
        ax_tpot_bo_y.fill_between(range(len(init_tpot_y_mu),
                                        len(init_tpot_y_mu) + len(bo_y_mu)), 
                                  bo_y_w, bo_y_b, alpha=.5, 
                                  linewidth=0, color='red')
        ax_tpot_bo_y.plot(range(len(init_tpot_y_mu), 
                                len(init_tpot_y_mu) + len(bo_y_mu)), 
                          bo_y_mu, linewidth=2,
                          label='Bayesian optimisation'+mean_text,color='red')
        ax_tpot_bo_y.legend()
        title_text = (f"{problem} - TPOT-BO [min/max] {add_text}\n"
                      + f"TPOT μ:{round(init_tpot_y_mu[-1],4)}, "
                      + f"BO μ: {round(bo_y_mu[-1],4)}, "
                      + f"min: {round(bo_y_b[-1],4)}, "
                      + f"max: {round(bo_y_w[-1],4)}")
        ax_tpot_bo_y.set_title(title_text)
        ax_tpot_bo_y.set_ylim([ylim_min,ylim_max])
        ax_tpot_bo_y.set_xlabel("Evaluations")
        ax_tpot_bo_y.set_ylabel("CV")
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
                        label='Bayesian optimisation'+mean_text,color='red')
    ax_tpot_bo_y_s.legend()
    title_text = (f"{problem} - TPOT-BO {add_text}\n"
                  + f"TPOT: μ: {round(init_tpot_y_mu[-1],4)}, "
                  + f"σ: {init_tpot_y_sigma[-1]:.3e}, "
                  + f"BO μ: {round(bo_y_mu[-1],4)}, "
                  + f"σ: {bo_y_sigma[-1]:.3e}")
    ax_tpot_bo_y_s.set_title(title_text)
    ax_tpot_bo_y_s.set_ylim([ylim_min,ylim_max])
    ax_tpot_bo_y_s.set_xlabel("Evaluations")
    ax_tpot_bo_y_s.set_ylabel("CV")
    plt.show()
    

    # plot alt results
    # compute plot limits
    # take max from the worst of the first bo iteration (excludes initial tpot)
    
    if params['PLOT_ALT']:
        if params['PLOT_MIN_MAX']:
            # plot alt results (min/max)
            fig9, ax_alt_tpot_bo = plt.subplots()
            alt_tpot_lines = {}
            alt_bo_lines = {}
            for i in range(len(data[run_idxs[-1]]['alt_bo_y'])):
                alt_tpot_start = i * pop_size * alt_tpot_gens + i * alt_bo_trials
                alt_bo_start = ((i+1) * pop_size * alt_tpot_gens 
                                + ((i>0) * (i)) * alt_bo_trials)
                ax_alt_tpot_bo.fill_between(
                    range(alt_tpot_start, alt_tpot_start + len(alt_tpot_y_mu[i])), 
                    alt_tpot_y_w[i], alt_tpot_y_b[i], alpha=.5, linewidth=0,color='C0')
                alt_tpot_lines[i], = ax_alt_tpot_bo.plot(
                    range(alt_tpot_start, alt_tpot_start + len(alt_tpot_y_mu[i])), 
                    alt_tpot_y_mu[i], linewidth=2,
                    label='TPOT evaluation'+mean_text,color='C0')
                ax_alt_tpot_bo.fill_between(range(alt_bo_start, 
                                                  alt_bo_start + len(alt_bo_y_mu[i])), 
                                            alt_bo_y_w[i], alt_bo_y_b[i], 
                                            alpha=.5, linewidth=0,color='r')
                alt_bo_lines[i], = ax_alt_tpot_bo.plot(
                    range(alt_bo_start, alt_bo_start + len(alt_bo_y_mu[i])), 
                    alt_bo_y_mu[i], linewidth=2,
                    label='BO evaluation'+mean_text,color='r')
            ax_alt_tpot_bo.legend(handles=[alt_tpot_lines[0], alt_bo_lines[0]])
                
            alt_title_text = (f"{problem} - Alt TPOT-BO [min/max] {add_text}\n"
                            + f"μ: {round(alt_bo_y_mu[len(data[run_idxs[-1]]['alt_bo_y'])-1][-1],4)}, "
                            + f"min: {round(alt_bo_y_b[len(data[run_idxs[-1]]['alt_bo_y'])-1][-1],4)}, "
                            + f"max: {round(alt_bo_y_w[len(data[run_idxs[-1]]['alt_bo_y'])-1][-1],4)}")
            ax_alt_tpot_bo.set_title(alt_title_text)
            ax_alt_tpot_bo.set_ylim([ylim_min, ylim_max])
            ax_alt_tpot_bo.set_xlabel("Evaluations")
            ax_alt_tpot_bo.set_ylabel("CV")
        
        
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
            
        alt_title_text_s = (f"{problem} - Alt TPOT-BO {add_text}\n"
                        + f"μ: {round(alt_bo_y_mu[len(data[run_idxs[-1]]['alt_bo_y'])-1][-1],4)}, "
                        + f"σ: {alt_bo_y_sigma[len(data[run_idxs[-1]]['alt_bo_y'])-1][-1]:.3e}")
        ax_alt_tpot_bo_s.set_title(alt_title_text_s)
        ax_alt_tpot_bo_s.set_ylim([ylim_min, ylim_max])
        ax_alt_tpot_bo_s.set_xlabel("Evaluations")
        ax_alt_tpot_bo_s.set_ylabel("CV")
    
    plt.show()
    
    if params['PLOT_AUTO']:
        # plot alt results (mu/sigma)
        fig11, ax_auto_s = plt.subplots()
        for i in range(200, auto_y_mu.shape[0], pop_size):
            ax_auto_s.plot(range(i,i+pop_size),auto_y_mu[i:i+pop_size,0], c=params['COLOURMAP'](auto_y_mu[i,1]),lw=4)
        
        tpot_line = Line2D([0], [0], label="TPOT evaluation", color='blue', linewidth=3)
        bo_line = Line2D([0], [0], label="BO evaluation", color='red', linewidth=3)
        labels = ['TPOT evaluation', 'BO evaluation']
        ax_auto_s.legend(handles=[tpot_line, bo_line])
        auto_title_text_s = (f"{problem} - Competing TPOT-BO {add_text} - mean\n"
                        + f"μ: {round(auto_y_mu[-1,0],4)}, "
                        + f"σ: {auto_y_sigma[-1,0]:.3e}")
        ax_auto_s.set_title(auto_title_text_s)
        # ax_auto_s.set_ylim([ylim_min, ylim_max])
        ax_auto_s.set_xlabel("Evaluations")
        ax_auto_s.set_ylabel("CV")
                
        data[med_auto_run]['auto_y'][-1,0]
        # plot alt results (median)
        fig12, ax_auto_m = plt.subplots()
        for i in range(200, data[med_auto_run]['auto_y'].shape[0], pop_size):
            colour = "red" if data[med_auto_run]['auto_y'][i,1] == 1 else "C0"
            ax_auto_m.plot(range(i,i+pop_size),data[med_auto_run]['auto_y'][i:i+pop_size,0], c=colour,lw=4)
        tpot_line = Line2D([0], [0], label="TPOT evaluation", color='C0', linewidth=3)
        bo_line = Line2D([0], [0], label="BO evaluation", color='red', linewidth=3)
        labels = ['TPOT evaluation', 'BO evaluation']
        ax_auto_m.legend(handles=[tpot_line, bo_line])
        
        auto_title_text_m = (f"{problem} - TPOT-BO AUTO {add_text}\n"
                        + f"median run: {med_auto_run}, "
                        +f"obj: {round(data[med_auto_run]['auto_y'][-1,0],4)}")
        ax_auto_m.set_title(auto_title_text_m)
        # ax_auto_s.set_ylim([ylim_min, ylim_max])
        ax_auto_m.set_xlabel("Evaluations")
        ax_auto_m.set_ylabel("CV")
        
        
    plt.show()        
        
        
        # alt_tpot_lines_s = {}
        # alt_bo_lines_s = {}
        # for i in range(len(data[run_idxs[-1]]['alt_bo_y'])):
        #     alt_tpot_start = i * pop_size * alt_tpot_gens + i * alt_bo_trials
        #     alt_bo_start = ((i+1) * pop_size * alt_tpot_gens 
        #                     + ((i>0) * (i)) * alt_bo_trials)
        #     ax_alt_tpot_bo_s.fill_between(
        #         range(alt_tpot_start, alt_tpot_start + len(alt_tpot_y_mu[i])), 
        #         alt_tpot_y_mu[i] + alt_tpot_y_sigma[i], 
        #         alt_tpot_y_mu[i] - alt_tpot_y_sigma[i], alpha=.5, 
        #         linewidth=0,color='C0')
        #     alt_tpot_lines_s[i], = ax_alt_tpot_bo_s.plot(
        #         range(alt_tpot_start, alt_tpot_start + len(alt_tpot_y_mu[i])), 
        #         alt_tpot_y_mu[i], linewidth=2,
        #         label='TPOT evaluation'+mean_text,color='C0')
        #     ax_alt_tpot_bo_s.fill_between(
        #         range(alt_bo_start, alt_bo_start + len(alt_bo_y_mu[i])), 
        #         alt_bo_y_mu[i] + alt_bo_y_sigma[i], 
        #         alt_bo_y_mu[i] - alt_bo_y_sigma[i], 
        #         alpha=.5, linewidth=0,color='r')
        #     alt_bo_lines_s[i], = ax_alt_tpot_bo_s.plot(
        #         range(alt_bo_start, alt_bo_start + len(alt_bo_y_mu[i])), 
        #         alt_bo_y_mu[i], linewidth=2,
        #         label='BO evaluation'+mean_text,color='r')
        # ax_alt_tpot_bo_s.legend(handles=[alt_tpot_lines_s[0], alt_bo_lines_s[0]])
            
        # alt_title_text_s = (f"{problem} - Alt TPOT-BO {add_text}\n"
        #                 + f"μ: {round(alt_bo_y_mu[len(data[run_idxs[-1]]['alt_bo_y'])-1][-1],4)}, "
        #                 + f"σ: {alt_bo_y_sigma[len(data[run_idxs[-1]]['alt_bo_y'])-1][-1]:.3e}")
        # ax_alt_tpot_bo_s.set_title(alt_title_text_s)
        # ax_alt_tpot_bo_s.set_ylim([ylim_min, ylim_max])
        # ax_alt_tpot_bo_s.set_xlabel("Evaluations")
        # ax_alt_tpot_bo_s.set_ylabel("CV")
    
    
    matching_idx = matching_data[:,0]
    
    r = 1
    r_old = 1
    for i in range(len(matching_idx)-1):
        r_old = matching_idx[i]
        matching_idx[i] = r
        if r_old != matching_idx[i+1]:
            r = r + 1
        
    matching_idx[-1] = r
       
    # box plot of matching
    box_data = [matching_data[np.where(matching_data[:,0]==i),1][0] 
                for i in range(1,len(run_idxs)+1)]
    
    fig7, ax_box = plt.subplots()
    ax_box.grid()
    boxplot_data = ax_box.boxplot(box_data,patch_artist=True)
    ax_box.set_xlim([0.25,max(matching_idx)+.75])
    ax_box.set_xlabel("Run")
    ax_box.set_ylabel("CV")
    ax_box.set_title(f"{problem} - TPOT "
                            + f"matching best @ gen {stop_gen} {add_text}")
    
    # box plot of matching without outliers
    fig8, ax_box2 = plt.subplots()
    ax_box2.grid()
    boxplot2_data = ax_box2.boxplot(box_data, showfliers=False,patch_artist=True)
    
    label_y_max = max([max(boxplot2_data['whiskers'][run*2+1].get_ydata()) for run in range(len(run_idxs))])
    label_y_min = min([min(boxplot2_data['whiskers'][run*2].get_ydata()) for run in range(len(run_idxs))])
    label_diff = label_y_max - label_y_min    
    for run in range(1,len(run_idxs)+1):
        label_y = label_y_max
        label_offset = 10
        if run % 2 == 1:
            label_y = label_y_min
            label_offset = -10
        
        ax_box2.annotate(f"[{len(np.where(matching_idx==run)[0])}]",
                        (run, label_y),
                        textcoords="offset points",
                        xytext=(0,label_offset),
                        ha='center')
    ax_box2.set_xlim([0.25,max(matching_idx)+.75])
    ax_box2.set_ylim([label_y_min-label_diff/8,label_y_max+label_diff/8])
    ax_box2.set_xlabel("Run [number of pipelines]")
    ax_box2.set_ylabel("CV")
    ax_box2.set_title(f"{problem} - TPOT "
                            + f"matching best @ gen {stop_gen} [no outliers] {add_text}")
    
    plt.show()
    
    if params['SAVE_PLOTS']:
        plot_dir = os.path.join(prob_dir, "Plots")
        
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        fname_tpot_plot = os.path.join(
            plot_dir, problem + "_tpot_min_max.png")
        fname_tpot_plot_s = os.path.join(
            plot_dir, problem + "_tpot_mu_sigma.png")
        fname_tpot_bo_plot = os.path.join(
            plot_dir, problem + "_bo_min_max.png")
        fname_tpot_bo_plot_s = os.path.join(
            plot_dir, problem + "_bo_mu_sigma.png")
        fname_matching_plot = os.path.join(
            plot_dir, problem + "_matching.png")
        fname_matching_plot_zoom = os.path.join(
            plot_dir, problem + "_matching_zoom.png")
        fname_box_plot = os.path.join(
            plot_dir, problem + "_box_plot.png")
        fname_box_plot2 = os.path.join(
            plot_dir, problem + "_box_plot_no_outliers.png")
        if params['PLOT_ALT']:
            fname_alt_plot = os.path.join(
                plot_dir, problem + "_alt_min_max.png")
            fname_alt_plot_s = os.path.join(
                plot_dir, problem + "_alt_mu_sigma.png")
        
        if params['PLOT_AUTO']:
            fname_auto_plot_s = os.path.join(
                plot_dir, problem + "_auto_mu_sigma.png")
        
        if params['PLOT_MIN_MAX']:
            fig1.savefig(fname_tpot_plot,bbox_inches='tight')
            fig3.savefig(fname_tpot_bo_plot,bbox_inches='tight')
        fig2.savefig(fname_tpot_plot_s,bbox_inches='tight')
        fig4.savefig(fname_tpot_bo_plot_s,bbox_inches='tight')
        # fig5.savefig(fname_matching_plot)
        # fig6.savefig(fname_matching_plot_zoom)
        fig7.savefig(fname_box_plot,bbox_inches='tight')
        fig8.savefig(fname_box_plot2,bbox_inches='tight')
        if params['PLOT_ALT']:
            fig10.savefig(fname_alt_plot_s,bbox_inches='tight')
            if params['PLOT_MIN_MAX']:
                fig9.savefig(fname_alt_plot,bbox_inches='tight')
        
        if params['PLOT_AUTO']:
            fig12.savefig(fname_auto_plot_s,bbox_inches='tight')
            
            
for problem in prob_list:
    print(f"\n{u.CYAN}{problem} statistics:{u.OFF}")
    print(f"{str(''):>{PRINT_COL}}{str('TPOT only'):>{PRINT_COL}}{str('TPOT + BO'):>{PRINT_COL}}{str('TPOT + BO (alt)'):>{PRINT_COL}}{str('TPOT + BO (auto)'):>{PRINT_COL}}")
    for _ in range(5*PRINT_COL):
        print("=",end='')
    print()
    
    for stat,methods in stats[problem].items():
        if stat == 'runs':
            continue
        print(f"{str(stat):>{PRINT_COL}}",end="")
        for method,val in methods.items():
            print(f"{round(val,12):>{PRINT_COL}}",end="")
        print()
            
if params['SAVE_STATS']:
    fname_stats = os.path.join(results_path, "stats.out")
    with open(fname_stats, 'a') as f:
        f.write(f"\n%TIME: {time.asctime()}\n\n")
        # print statistics
        for problem in prob_list:
            f.write(f"***** {problem} *****\n")
            f.write(f"!RUN LIST:{stats[problem]['runs']}\n")
            f.write(f"{str('#'):<{PRINT_COL}};{str('TPOT only'):>{PRINT_COL}};{str('TPOT + BO'):>{PRINT_COL}};{str('TPOT + BO (alt)'):>{PRINT_COL}};{str('TPOT + BO (auto)'):>{PRINT_COL}};\n")
            
            for stat,methods in stats[problem].items():
                if stat == 'runs':
                    continue
                f.write(f"{str(stat):>{PRINT_COL}};")
                for method,val in methods.items():
                    f.write(f"{round(val,12):>{PRINT_COL}};")
                f.write("\n")
            f.write("\n")
    
