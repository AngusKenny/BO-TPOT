#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 18:29:51 2022

@author: gus
"""
import sys
import warnings
from tpot_tools import TestHandler
from tpot_config import reduced_tpot_config_dict, default_tpot_config_dict

''' General Parameters

Verbosity settings:
    0 = all output off (print errors only - some sklearn warnings might appear)
    1 = progress information
    2 = debug information
    3 = show everything, including warnings
'''
params = {
    'RUN_TPOT' : True,
    'RUN_BO' : True,
    'RUN_ALT' : True,
    'VERBOSITY' : 1,
    'DATA_DIR' : 'Data',
    'RESULTS_DIR' : 'Results',
    # if not generating TPOT data, RUNS can be a list of runs
    'RUNS' : 21,
    'PROBLEMS' : [
                # 'abalone',
                'quake',
    #             'house_16h',
    #             'brazilian_houses',
    #             'diamonds','elevators',
    #             'black_friday'
                 ],
    'TPOT_CONFIG_DICT' : default_tpot_config_dict,
    'nJOBS' : -1,
    # toggle between real and discrete parameter spaces
    'REAL_VALS' : False,
    # maximum time allowed for a single pipeline evaluation (mins)
    'PIPE_EVAL_TIMEOUT' : 5,
    #
    # TPOT data generation parameters
    #
    'START_SEED' : 42,
    'POP_SIZE' : 100,
    'nTOTAL_GENS' : 100,
    'STOP_GEN' : 80,
    #
    # BO and TPOT + BO alternating parameters
    #
    # stop optuna running forever if it cannot find enough new pipelines
    'OPTUNA_TIMEOUT_TRIALS' : 100,
    'nALT_ITERS' : 10,
    }

# suppress experimental warnings, etc if verbosity below 4
if params['VERBOSITY'] < 3:
    warnings.simplefilter("ignore")

tpot_handler = TestHandler(params)

for problem in params['PROBLEMS']:
    tpot_handler.write_problem(problem)
        
    for run_idx in range(len(tpot_handler.run_list)):
        # generate TPOT data - this creates a new run folder, so we need
        # to get the run number to pass to the other processes
        if params['RUN_TPOT']:
            run = tpot_handler.generate_tpot_data(run_idx, problem)
            if not run:
                continue
        else:
            run = tpot_handler.run_list[run_idx]
            # write run information to progress file
            tpot_handler.write_run(run)
            
        # run BO optimiser
        if params['RUN_BO']:
            tpot_handler.run_BO(run, problem)
        
        # run alternating TPOT + BO
        if params['RUN_ALT']:
            tpot_handler.run_alt(run, problem)
        
tpot_handler.write_end()