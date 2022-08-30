#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 18:29:51 2022

@author: gus
"""
import sys
import warnings
from utils.test_handler import TestHandler
from config.tpot_config import reduced_tpot_config_dict, default_tpot_config_dict

''' General Parameters

Verbosity settings:
    0 = all output off (print errors only - some sklearn warnings might appear)
    1 = progress information
    2 = debug information
    3 = show everything, including warnings
'''
params = {
    # clear BO and alt data from directories to be written to 
    # (will ask for confirmation)
    'CLEAN_DATA': False,
    'RUN_TPOT-BASE' : False,
    'RUN_TPOT-BO-S' : True,
    'RUN_TPOT-BO-ALT' : False,
    'RUN_TPOT-BO-AUTO' : False,
    'RESTRICT_BO' : True,
    'VERBOSITY' : 2,
    'DATA_DIR' : 'Data',
    'RESULTS_DIR' : 'Results_newcode',
    # if not generating TPOT data, RUNS can be a list of runs
    'RUNS' : [0,1,2],
    'PROBLEMS' : [
                # 'abalone',
# 		'socmob',
                'quake',
    #             'house_16h',
    #             'brazilian_houses',
    #             'diamonds',
                 # 'elevators',
    #             'black_friday'
                 ],
    'TPOT_CONFIG_DICT' : default_tpot_config_dict,
    'nJOBS' : 4,
    # toggle between discrete and continuous parameter spaces
    'DISCRETE_MODE' : True,
    # maximum time allowed for a single pipeline evaluation (mins)
    'PIPE_EVAL_TIMEOUT' : 1,
    #
    # TPOT data generation parameters
    #
    'START_SEED' : 442,
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

test_handler = TestHandler(params)

for problem in test_handler.prob_list:
    test_handler.write_problem(problem)
    test_handler.set_problem(problem)
    for run in test_handler.run_list:
        # generate TPOT data - this creates a new run directory, so we need
        # to get the run number to pass to the other processes
        if params['RUN_TPOT']:
            test_handler.set_run()
            tpot_data = test_handler.run_TPOT_BASE()
            if tpot_data is None:
                test_handler.vprint.verr("TPOT data not generated, skipping run..\n\n")
                test_handler.seed_inc()
                continue
        else:
            test_handler.set_run(run)
            tpot_data = test_handler.load_TPOT_data()
            
        # run BO optimiser
        if params['RUN_BO']:
            test_handler.run_TPOT_BO_S(tpot_data)
        
        # run alternating TPOT + BO
        if params['RUN_ALT']:
            test_handler.run_TPOT_BO_ALT(tpot_data)
            
        # run alternating TPOT + BO
        if params['RUN_AUTO']:
            test_handler.run_TPOT_BO_AUTO(tpot_data)
        
        test_handler.seed_inc()
        
test_handler.write_end()