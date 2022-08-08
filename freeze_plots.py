#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 12:28:54 2022

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
    'RESULTS_DIR'   : 'Results_abalone',
    'PROBLEMS'      : ['abalone'],
    'RUN_LIST'      : [],
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
fig = plt.figure()
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
    
    # validate and collect data from specified runs
    for run in run_idxs:
        skip_run = False
        run_str = str(run)    
        if run < 10:
            run_str = "0" + str(run)
            
        run_dir = os.path.join(prob_dir,"Run_" + run_str)
        
        bo_r_dir = os.path.join(run_dir, 'bo_restricted')
        
        fname_prog = os.path.join(bo_r_dir, "bo_restricted_progress.out")
        fname_plot_data = os.path.join(bo_r_dir, "bo_res_plot.out")
        fname_plot = os.path.join(bo_r_dir, "freeze_plot.png")
        
        
        for fname in [fname_prog, fname_plot_data]:
            if not os.path.exists(fname):
                print(f"{u.RED}Missing file error:{u.OFF} Run {run} of " 
                      + f"{problem} is missing file {os.path.basename(fname)}"
                      + " - skipping run..")
                skip_run = True
                
        if skip_run:
            continue
        
        plot_data = np.loadtxt(fname_plot_data, delimiter=',')
        
        with open(fname_prog, 'r') as f:
            for line in f:
                if "FROZEN" in line:
                    n_freeze = int(line.split(":")[-1])
        
        n_params = np.max(plot_data[:,0]) - n_freeze
        
        cutoff = 0.95 * np.max(plot_data[:,1])
        
        
        plt.plot(plot_data[:,0],plot_data[:,1])
        plt.axvline(n_params,c='black',linestyle='dashed',lw=1)
        plt.axhline(cutoff,c='black',linestyle='dashed',lw=1)
        plt.text(np.max(plot_data[:,0])-0.5, 0.95*cutoff, "95%")
        plt.ylabel("score")
        plt.xlabel("number of variables")
        plt.title(f"$R^2$ score plot for TPOT-BO-R\n{problem} :: Run {run_str}")
        plt.savefig(fname_plot,bbox_inches='tight')
        plt.show()