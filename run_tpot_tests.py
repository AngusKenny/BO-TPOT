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
    'METHOD' : 
        'TPOT-BASE',
        # 'TPOT-BO-S',
        # 'TPOT-BO-ALT',
        # 'TPOT-BO-AUTO',
    # 
    'VERBOSITY' : 2,               
    'DATA_DIR' : 'Data',
    'RESULTS_DIR' : 'Results',
    # if not generating TPOT data, RUNS can be a list of runs
    'SEEDS' : [45],
    'PROBLEM' : 
        'quake',
		# 'socmob',
        # 'abalone',
        # 'house_16h',
        # 'brazilian_houses',
        # 'diamonds',
        # 'elevators',
        # 'black_friday'
    #       
    'TPOT_CONFIG_DICT' : default_tpot_config_dict,
    'nJOBS' : 8,
    # toggle between discrete and continuous parameter spaces
    'DISCRETE_MODE' : False,
    # maximum time allowed for a single pipeline evaluation (mins)
    'PIPE_EVAL_TIMEOUT' : 5,
    #
    # TPOT data generation parameters
    #
    # 'START_SEED' : 42, # only used for initial TPOT-BASE runs
    'POP_SIZE' : 100,
    'nTOTAL_GENS' : 100,
    'STOP_GEN' : 80,
    #
    # stop optuna running forever if it cannot find enough new pipelines
    'OPTUNA_TIMEOUT_TRIALS' : 100,
    #
    # BO and TPOT + BO alternating parameters
    'nALT_ITERS' : 10,
    }

# suppress experimental warnings, etc if verbosity below 3
if params['VERBOSITY'] < 3:
    warnings.simplefilter("ignore")

test_handler = TestHandler(params)
test_handler.set_problem(params['PROBLEM'])
run_method_name = f"run_{params['METHOD'].replace('-','_')}"
fn = getattr(test_handler,run_method_name)
for seed in test_handler.seed_list:
    fn(seed)
test_handler.write_end()