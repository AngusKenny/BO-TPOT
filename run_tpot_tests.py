#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 18:29:51 2022

@author: gus
"""
import warnings
from tpot_tools import TestHandler
from tpot_config import default_tpot_config_dict

''' General Parameters

Verbosity settings:
    0 = all output off (print errors only - some sklearn warnings might appear)
    1 = progress information
    2 = debug information
    3 = show everything, including warnings
'''
params = {
            'RUN_TPOT' : False,
            'RUN_BO' : True,
            'RUN_ALT' : True,
            'VERBOSITY' : 1,
            'PROBLEMS' : [
                        # 'abalone',
                        'quake',
            #             'house_16h',
            #             'brazilian_houses',
            #             'diamonds','elevators',
            #             'black_friday'
                         ],
            'RESULTS_DIR' : 'Results',
            'DATA_DIR' : 'Data',
            'START_SEED' : 42,
            'TPOT_CONFIG_DICT' : default_tpot_config_dict,
            'nJOBS' : -1,
            # toggle between real and discrete parameter spaces
            'REAL_VALS' : True,
            # stop optuna running forever if it cannot find enough new pipelines
            'OPTUNA_TIMEOUT_TRIALS' : 100,
            # maximum time allowed for a single pipeline evaluation (mins)
            'PIPE_EVAL_TIMEOUT' : 5,
            #
            # TPOT data generation parameters
            #
            'nRUNS' : 1,
            'POP_SIZE' : 100,
            'nTOTAL_GENS' : 100,
            'STOP_GEN' : 80,
            #
            # TPOT + BO alternating parameters
            #
            'nALT_ITERS' : 10,
            }

# suppress experimental warnings, etc if verbosity below 4
if params['VERBOSITY'] < 3:
    warnings.simplefilter("ignore")

tpot_handler = TestHandler(params)

for problem in params['PROBLEMS']:
    tpot_handler.write_problem(problem)
    for run in range(params['nRUNS']):
        # write run information to progress file
        tpot_handler.write_run(run)
        
        # generate TPOT data if this fails, skip entire run
        if params['RUN_TPOT']:
            if not tpot_handler.generate_tpot_data(run, problem): continue
        
        # run BO optimiser
        if params['RUN_BO']:
            tpot_handler.run_BO(run, problem)
        
        # run alternating TPOT + BO
        if params['RUN_ALT']:
            tpot_handler.run_alt(run, problem)
        
tpot_handler.write_end()