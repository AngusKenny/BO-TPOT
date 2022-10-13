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
    'RUN_TPOT-BO-S' : False,
    'RUN_TPOT-BO-H' : False,
    'RUN_TPOT-BO-ND' : False,
    'RUN_TPOT-BO-Sr' : False,
    'RUN_TPOT-BO-ALT' : False,
    'RUN_TPOT-BO-AUTO' : True,
    'VERBOSITY' : 2,
    'DATA_DIR' : 'Data',
    'RESULTS_DIR' : 'Results',
    # if not generating TPOT data, RUNS can be a list of runs
    'RUNS' : [14,15,16,17,18,19,20],
    'PROBLEMS' : [
                  'quake',
# 		'socmob',
                #'abalone',
               # 'house_16h',
                #'brazilian_houses',
    #             'diamonds',
                 # 'elevators',
    #             'black_friday'
                 ],
    'TPOT_CONFIG_DICT' : default_tpot_config_dict,
    'nJOBS' : 8,
    # toggle between discrete and continuous parameter spaces
    'DISCRETE_MODE' : True,
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

test_handler = TestHandler(params)

seed = params['START_SEED']

for problem in test_handler.prob_list:
    test_handler.set_problem(problem)
    for run in test_handler.run_list:
        # generate TPOT data - this creates a new run directory, so we need
        # to get the run number to pass to the other processes
        if params['RUN_TPOT-BASE']:
            test_handler.set_run()
            tpot_data = test_handler.run_TPOT_BASE()
            if tpot_data is None:
                test_handler.vprint.verr("TPOT data not generated, skipping run..\n\n")
                test_handler.update_seed()
                continue
        else:
            test_handler.set_run(run)
            seed, pop_size, tpot_data = test_handler.load_TPOT_data()
            test_handler.params['POP_SIZE'] = pop_size
            test_handler.update_seed(seed)
            
        # run BO optimiser
        if params['RUN_TPOT-BO-S']:
            test_handler.run_TPOT_BO_S(tpot_data, restricted_hps=False)
        
        # run BO optimiser
        if params['RUN_TPOT-BO-H']:
            test_handler.run_TPOT_BO_H(tpot_data, restricted_hps=False)
        
        # run BO optimiser
        if params['RUN_TPOT-BO-ND']:
            test_handler.run_TPOT_BO_ND(tpot_data, restricted_hps=False)
        
        # run BO optimiser
        if params['RUN_TPOT-BO-Sr']:
            test_handler.run_TPOT_BO_S(tpot_data, restricted_hps=True)
        
        # run alternating TPOT + BO
        if params['RUN_TPOT-BO-ALT']:
            test_handler.run_TPOT_BO_ALT(tpot_data)
            
        # run alternating TPOT + BO
        if params['RUN_TPOT-BO-AUTO']:
            test_handler.run_TPOT_BO_AUTO(tpot_data)
        
        test_handler.update_seed()
        
test_handler.write_end()