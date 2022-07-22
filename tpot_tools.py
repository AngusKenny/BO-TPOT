#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 18:31:18 2022

@author: gus
"""

import time
import os
import sys
import copy
import traceback
import utils as u
from shutil import rmtree
from optuna_pipeline_opt import PipelinePopOpt
from tpot import TPOTRegressor
from deap import creator
from tpot_config import default_tpot_config_dict
          
        
def get_tpot_data(tot_gens=100, 
                  pop_size=100, 
                  stop_gen=80, 
                  n_runs=1, 
                  start_seed=42, 
                  prob_list=[], 
                  data_dir='Data', 
                  results_dir='Results',
                  tpot_config_dict=default_tpot_config_dict,
                  n_jobs=-1,
                  vprint=u.Vprint(1),
                  pipe_eval_timeout=5):
    
    # set tpot verbosity to vprint.verbosity + 1 to give more information
    tpot_verb = vprint.verbosity + 1 if vprint.verbosity > 0 else 0
    
    if tot_gens < stop_gen:
        sys.exit(f"Total gens less than stop gen ({tot_gens} < {stop_gen})")
    
    # dependent variables (do not touch!)
    n_bo_evals = (tot_gens - stop_gen) * pop_size
    
    cwd = os.getcwd()
    data_path = os.path.join(cwd,data_dir)
    if not os.path.exists(data_path):
        sys.exit(f"Cannot find data directory {data_path}") 
        
    # if problem list is empty, search data directory for .data files
    if len(prob_list) == 0:
        prob_list = [f.split(".")[0] 
                     for f in os.listdir(data_path) if f.endswith(".data")]
    
    # iterate over problem list
    for problem in prob_list:
        # Reading the data file for the given problem
        fln=problem + ".data"
        fpath = os.path.join(cwd, data_dir, fln)
        X_train, X_test, y_train, y_test = u.load_data(fpath)
    
        # establish problem directory and make it if it doesn't exist
        prob_dir = os.path.join(cwd, results_dir, problem)
        if not os.path.exists(prob_dir):
            os.makedirs(prob_dir)
            
        for seed in range(start_seed,start_seed + n_runs):
            # copy config dict so any changes arent permanent across runs
            tpot_config_copy = copy.deepcopy(tpot_config_dict)
            
            # check last run in problem directory and create next run directory
            run_no = 0
            exist_runs = [int(f.path.split("_")[-1]) 
                          for f in os.scandir(prob_dir) if f.is_dir() 
                          and "Plots" not in f.path]
            if len(exist_runs) > 0:
                run_no = max(exist_runs) + 1
            run_str = str(run_no)
            if run_no < 10:
                run_str = "0" + str(run_no)
            
            run_dir = os.path.join(prob_dir, "Run_" + run_str)
    
            if not os.path.exists(run_dir):
                os.makedirs(run_dir)
            
            tpot_dir = os.path.join(run_dir, 'tpot')
            
            if not os.path.exists(tpot_dir):
                os.makedirs(tpot_dir)
            
            # establish filenames for data output
            fname_prog = os.path.join(tpot_dir, "tpot_progress.out")
            fname_tpot_pipes = os.path.join(tpot_dir, "tpot_pipes.out")
            fname_matching = os.path.join(tpot_dir, "matching_pipes.out")
                
            # start timer
            t_start = time.time()
            
            # create TPOT object and fit for tot_gens generations
            tpot = TPOTRegressor(generations=tot_gens-1,
                                  population_size=pop_size, 
                                  mutation_rate=0.9, 
                                  crossover_rate=0.1, 
                                  cv=5,
                                  verbosity=tpot_verb, 
                                  config_dict=tpot_config_copy, 
                                  random_state=seed, 
                                  n_jobs=n_jobs,
                                  warm_start=False,
                                  max_eval_time_mins=pipe_eval_timeout)
            
            vprint.v2(f"{u.CYAN}fitting tpot model with {tpot.generations}" 
                   + " generations (-1 to account for initial evaluations)" 
                   + f"..\n{u.WHITE}")
            
            tpot.fit(X_train, y_train)
                
            vprint.v1("")
            
            # record fitting time
            t_tpot = time.time()
            
            best_init_pipe = ""
            best_init_cv = -1e20
            best_tpot_pipe = ""
            best_tpot_cv = -1e20
            
            # write all evaluated pipes
            with open(fname_tpot_pipes, 'w') as f:
                for k,v in tpot.evaluated_individuals_.items():
                    f.write(f"{k};{v['generation']};{v['internal_cv_score']}\n")
                    # overall best tpot
                    if v['internal_cv_score'] > best_tpot_cv:
                        best_tpot_pipe = k
                        best_tpot_cv = v['internal_cv_score']
                    # best tpot up until stop gen
                    if v['generation'] < stop_gen:
                        if v['internal_cv_score'] > best_init_cv:
                            best_init_pipe = k
                            best_init_cv = v['internal_cv_score']
            
            vprint.v2(f"best_init_pipe: {best_init_pipe}\n")
            vprint.v2(f"best_tpot_pipe: {best_tpot_pipe}\n")
                           
            # make PipelinePopOpt object to use its member methods
            po = PipelinePopOpt(tpot, vprint=vprint)
            
            # get all pipelines that match the structure of best initial pipeline           
            matching_strs = po.get_matching_structures(best_init_pipe, 
                                                        stop_gen=stop_gen)
    
            with open(fname_matching, 'w') as f:
                for pipe_str in matching_strs:
                    f.write(f"{pipe_str};"
                            + f"{tpot.evaluated_individuals_[pipe_str]['generation']};" 
                            + f"{tpot.evaluated_individuals_[pipe_str]['internal_cv_score']}\n")
                    
            # write progress to file
            with open(fname_prog, 'w') as f:
                f.write(f"TIME:{time.asctime()}\n")
                f.write(f"SEED:{seed}\n")
                f.write(f"POP SIZE:{pop_size}\n")
                f.write(f"TOTAL TPOT GENS:{tot_gens}\n")
                f.write(f"TPOT STOP GEN:{stop_gen}\n")
                f.write(f"BAYESIAN OPTIMISATION EVALS TO RUN:{n_bo_evals}\n")
                f.write("\n")
                f.write(f"***** AFTER {stop_gen} INITIAL TPOT GENERATIONS *****\n")
                f.write(f"Best initial TPOT CV:{best_init_cv}\n")
                f.write(f"Best initial TPOT pipeline: ({len(matching_strs)} "
                        +"other pipelines with matching structure)\n")
                f.write(f"{best_init_pipe}\n")
                f.write("\n")
                f.write(f"***** AFTER {tot_gens} TPOT GENERATIONS *****\n")
                f.write(f"Time elapsed:{round(t_tpot-t_start,2)}\n")
                f.write(f"Best full TPOT CV:{best_tpot_cv}\n")
                f.write("Best full TPOT pipeline:\n")
                f.write(f"{best_tpot_pipe}\n")

    # if we are running this from run_tpot_tests.py return run number    
    if len(prob_list) == 1 and n_runs == 1:
        return run_no
    
                
def run_bo(run_list=[], 
           optuna_timeout_trials=100,
           force_bo_evals=None,
           ignore_results=True,
           prob_list=[], 
           data_dir='Data', 
           results_dir='Results',
           tpot_config_dict=default_tpot_config_dict,
           n_jobs=-1,
           vprint=u.Vprint(1),
           real_vals=True,
           pipe_eval_timeout=5):

    # set tpot verbosity to vprint.verbosity + 1 to give more information
    tpot_verb = vprint.verbosity + 1 if vprint.verbosity > 0 else 0
    
    cwd = os.getcwd()
    data_path = os.path.join(cwd,data_dir)
    if not os.path.exists(data_path):
        sys.exit(f"Cannot find data directory {data_path}")
        
    # if problem list is empty, search data directory for .data files
    if len(prob_list) == 0:
        prob_list = [f.split(".")[0] 
                     for f in os.listdir(data_path) if f.endswith(".data")]
        
    # iterate over problem list
    for problem in prob_list:
        # Reading the data file for the given problem
        cwd = os.getcwd()
        fln=problem + ".data"
        fpath = os.path.join(cwd, data_dir, fln)
        X_train, X_test, y_train, y_test = u.load_data(fpath)
    
        # establish problem directory and make it if it doesn't exist
        prob_dir = os.path.join(cwd, results_dir, problem)
        if not os.path.exists(prob_dir):
            if len(prob_list) == 1:
                sys.exit(f"Cannot find problem directory {prob_dir} - " 
                            + "skipping problem..")
            else:
                vprint.verr(f"Cannot find problem directory {prob_dir} - " 
                            + "skipping problem..")    
                continue
    
        # get available run directories
        if len(run_list) == 0:
            run_idxs = [int(d.path.split("_")[-1]) 
                        for d in os.scandir(prob_dir) if d.is_dir() 
                        and "Plots" not in d.path]
            run_idxs.sort()
        else:
            run_idxs = run_list
            
        for run_no in run_idxs:
            # copy config dict so any changes arent permanent across runs
            tpot_config_copy = copy.deepcopy(tpot_config_dict)
            
            run_str = str(run_no)
            if run_no < 10:
                run_str = "0" + str(run_no)
                
            run_dir = os.path.join(prob_dir, "Run_" + run_str)
            
            if not os.path.exists(run_dir):
                if len(prob_list) == 1 and len(run_idxs) == 1:
                    sys.exit(f"Cannot find run directory {run_dir} - " 
                                + "skipping run..")
                else:
                    vprint.verr(f"Cannot find run directory {run_dir} - " 
                                + "skipping run..")
                    continue
            
            bo_dir = os.path.join(run_dir, 'bo')
            if not os.path.exists(bo_dir):
                os.makedirs(bo_dir)
            
            tpot_dir = os.path.join(run_dir, 'tpot')
            
            # establish filenames for data output
            fname_tpot_prog = os.path.join(tpot_dir, "tpot_progress.out")
            fname_tpot_pipes = os.path.join(tpot_dir, "tpot_pipes.out")
            fname_bo_prog = os.path.join(bo_dir, "bo_progress.out")
            fname_bo_pipes = os.path.join(bo_dir, "bo_pipes.out")
            
            if os.path.exists(fname_bo_pipes) and not ignore_results:
                if len(prob_list) == 1 and len(run_idxs) == 1:
                    sys.exit(f"bo_pipes.out already exists in {run_dir} - " 
                                + "skipping run..")
                else:
                    vprint.verr(f"bo_pipes.out already exists in {run_dir} - " 
                                + "skipping run..")
                    continue
            else:
                vprint.v2(f"\nProcessing {run_dir}")
            
            if not os.path.exists(fname_tpot_pipes):
                if len(prob_list) == 1 and len(run_idxs) == 1:
                    sys.exit("Cannot find original TPOT pipes file " 
                                + f"{fname_tpot_pipes} - skipping run..")
                else:
                    vprint.verr("Cannot find original TPOT pipes file " 
                                + f"{fname_tpot_pipes} - skipping run..")
                    continue
            
            if not os.path.exists(fname_tpot_prog):
                if len(prob_list) == 1 and len(run_idxs) == 1:
                    sys.exit("Cannot find original TPOT progress file " 
                                + f"{fname_tpot_prog}")
                else:
                    vprint.verr("Cannot find original TPOT progress file " 
                                + f"{fname_tpot_prog}")
                    continue
            
            # load data from existing progress file
            (seed, 
             n_tot_gens, 
             tpot_stop_gen, 
             pop_size) = u.get_run_data(fname_tpot_prog)
            
            n_bo_evals = (n_tot_gens - tpot_stop_gen) * pop_size
            
            if force_bo_evals:
                n_bo_evals = force_bo_evals
            
            loaded_pipes = u.get_progress_pop(fname_tpot_pipes, tpot_stop_gen-1)
            
            best_init_pipe = ""
            best_init_cv = -1e20
            for k,v in loaded_pipes.items():
                if v['internal_cv_score'] > best_init_cv:
                    best_init_pipe = k
                    best_init_cv = v['internal_cv_score']
            
            # start timer
            t_start = time.time()        
            
            # create TPOT object and fit for 0 generations
            tpot = TPOTRegressor(generations=0,
                                  population_size=1, 
                                  mutation_rate=0.9, 
                                  crossover_rate=0.1, 
                                  cv=5,
                                  verbosity=tpot_verb, 
                                  config_dict=tpot_config_copy, 
                                  random_state=seed, 
                                  n_jobs=n_jobs,
                                  warm_start=True,
                                  max_eval_time_mins=pipe_eval_timeout)
            
            # initialise tpot object to generate pset
            tpot._fit_init()
            
            # replace evaluated individuals dict so we can get best and matching
            tpot.evaluated_individuals_ = loaded_pipes
            
            # create optimiser object
            po = PipelinePopOpt(tpot, vprint=vprint,real_vals=real_vals)
    
            # get all pipelines that match the structure of best initial pipeline           
            matching_strs = po.get_matching_structures(best_init_pipe, 
                                                       stop_gen=tpot_stop_gen)
            
            matching_dict = {best_init_pipe:loaded_pipes[best_init_pipe]}
            
            # remove generated pipeline and transplant saved from before
            tpot._pop = [creator.Individual.from_string(
                best_init_pipe, tpot._pset)]
            
            # convert pipe strings to parameter sets
            seed_samples = [(u.string_to_params(best_init_pipe),
                            loaded_pipes[best_init_pipe]['internal_cv_score'])]
            for pipe_str in matching_strs:
                matching_dict[pipe_str] = loaded_pipes[pipe_str]
                seed_samples.append((u.string_to_params(pipe_str),
                                    loaded_pipes[pipe_str]['internal_cv_score']))
            
            # replace tpot evaluated individuals with matching pipes
            tpot.evaluated_individuals_ = matching_dict
            
            vprint.v2(f"{u.CYAN}\nfitting tpot model with {tpot.generations}" 
                    + f" generations to initialise..\n{u.OFF}")
            
            tpot.fit(X_train, y_train)
            
            vprint.v1("")
            
            # reinitialise optimiser object with new values
            po = PipelinePopOpt(tpot, vprint=vprint, real_vals=real_vals)
            
            vprint.v2(f"\n{u.CYAN}Transplanting best pipe from earlier and " 
                      + f"optimising for {n_bo_evals+len(seed_samples)} "
                      + f"evaluations..{u.OFF}\n")
              
            # run bayesian optimisation with seed_dicts as initial samples
            po.optimise(0, X_train, y_train, n_evals=n_bo_evals,
                        seed_samples=seed_samples,real_vals=real_vals, 
                        timeout_trials=optuna_timeout_trials)
    
            # record time        
            t_end_bo = time.time()
            
            best_bo_pipe = ""
            best_bo_cv = -1e20
            
            # write all evaluated pipes
            with open(fname_bo_pipes, 'w') as f:
                for k,v in tpot.evaluated_individuals_.items():
                    f.write(f"{k};{v['internal_cv_score']}\n")
                    # overall best tpot
                    if v['internal_cv_score'] > best_bo_cv:
                        best_bo_pipe = k
                        best_bo_cv = v['internal_cv_score']
            
            # write final results to prog file
            with open(fname_bo_prog, 'w') as f:
                # update progress file for validation
                f.write(f"TIME:{time.asctime()}\n")
                f.write(f"SEED:{seed}\n")
                f.write(f"POP SIZE:{pop_size}\n")
                f.write(f"TOTAL TPOT GENS:{n_tot_gens}\n")
                f.write(f"TPOT STOP GEN:{tpot_stop_gen}\n")
                f.write(f"BAYESIAN OPTIMISATION EVALS:{n_bo_evals}\n")
                f.write(f"REAL_VALS:{real_vals}\n")
                f.write("\n")
                f.write(f"***** AFTER {tpot_stop_gen} INITIAL TPOT " 
                        + "GENERATIONS *****\n")
                f.write(f"Best CV:{best_init_cv}\n")
                f.write(f"Best pipeline: ({len(matching_strs)} pipelines "
                        + "with matching structures)\n")
                f.write(f"{best_init_pipe}\n")
                f.write("\n")
                f.write(f"\n***** AFTER {n_bo_evals} BAYESIAN OPTIMISATION " 
                        + f"EVALS ({n_tot_gens-tpot_stop_gen} TPOT " 
                        + "GENS EQUIVALENT) *****\n")
                f.write(f"Time elapsed:{round(t_end_bo-t_start,2)}\n")
                f.write(f"Best CV:{best_bo_cv}\n")
                f.write("Best pipeline:\n")
                f.write(f"{best_bo_pipe}\n")
            
            vprint.v2("Complete!\n")
            
            vprint.v1("Time elapsed: {round(t_end_bo-t_start,2)}")
            vprint.v1(f"Best CV: {best_bo_cv}")
            vprint.v1(f"Best pipeline:\n{best_bo_pipe}\n")
            
            
def run_tpot_bo_alt(n_iters=10,
                    run_list=[], 
                    optuna_timeout_trials=100,
                    force_bo_evals=None,
                    ignore_results=True,
                    prob_list=[], 
                    data_dir='Data', 
                    results_dir='Results',
                    tpot_config_dict=default_tpot_config_dict,
                    n_jobs=-1,
                    vprint=u.Vprint(1),
                    real_vals=True,
                    pipe_eval_timeout=5):

    # set tpot verbosity to vprint.verbosity + 1 to give more information
    tpot_verb = vprint.verbosity + 1 if vprint.verbosity > 0 else 0
    
    cwd = os.getcwd()
    data_path = os.path.join(cwd,data_dir)
    if not os.path.exists(data_path):
        sys.exit(f"Cannot find data directory {data_path}")
        
    # if problem list is empty, search data directory for .data files
    if len(prob_list) == 0:
        prob_list = [f.split(".")[0] 
                     for f in os.listdir(data_path) if f.endswith(".data")]
    
    # iterate over problem list
    for problem in prob_list:
        # Reading the data file for the given problem
        fln=problem + ".data"
        fpath = os.path.join(cwd, data_dir, fln)
        X_train, X_test, y_train, y_test = u.load_data(fpath)
    
        # find problem directory and skip if it doesn't exist
        prob_dir = os.path.join(cwd, results_dir, problem)
        if not os.path.exists(prob_dir):
            if len(prob_list) == 1:
                sys.exit(f"Cannot find problem directory {prob_dir} - " 
                            + "skipping problem..")
            else:
                vprint.verr(f"Cannot find problem directory {prob_dir} - " 
                            + "skipping problem..")
                continue
    
        # get available run directories
        if len(run_list) == 0:
            run_idxs = [int(d.path.split("_")[-1]) 
                        for d in os.scandir(prob_dir) if d.is_dir() 
                        and "Plots" not in d.path]
            run_idxs.sort()
        else:
            run_idxs = run_list
            
        for run_no in run_idxs:
            # copy config dict so any changes arent permanent across runs
            main_tpot_config_dict = copy.deepcopy(tpot_config_dict)
            
            run_str = str(run_no)
            if run_no < 10:
                run_str = "0" + str(run_no)
            
            run_dir = os.path.join(prob_dir, "Run_" + run_str)
            
            if not os.path.exists(run_dir):
                if len(prob_list) == 1 and len(run_idxs) == 1:
                    sys.exit(f"Cannot find run directory {run_dir}")
                else:
                    vprint.verr(f"Cannot find run directory {run_dir}")
                    continue
            
            tpot_dir = os.path.join(run_dir, 'tpot')
            alt_dir = os.path.join(run_dir, 'alt')
            
            if not os.path.exists(tpot_dir):
                if len(prob_list) == 1 and len(run_idxs) == 1:
                    sys.exit(f"Cannot find TPOT directory {tpot_dir}")
                else:
                    vprint.verr(f"Cannot find TPOT directory {tpot_dir}")
                    continue
            
            if not os.path.exists(alt_dir):
                os.makedirs(alt_dir)
            
            # establish filenames for data output
            fname_tpot_prog = os.path.join(tpot_dir, "tpot_progress.out")
            fname_tpot_pipes = os.path.join(tpot_dir, "tpot_pipes.out")
            fname_alt_prog = os.path.join(alt_dir, "alt_progress.out")
            fname_alt_tpot_pipes = os.path.join(alt_dir, "alt_tpot_pipes.out")
            fname_alt_bo_pipes = os.path.join(alt_dir, "alt_bo_pipes.out")
            
            if os.path.exists(fname_alt_bo_pipes) and not ignore_results:
                if len(prob_list) == 1 and len(run_idxs) == 1:
                    sys.exit(f"alt_bo_pipes.out already exists in {alt_dir} - " 
                                + "skipping run..")
                else:
                    vprint.verr(f"alt_bo_pipes.out already exists in {alt_dir} - " 
                                + "skipping run..")
                    continue
            else:
                # delete alt_bo_pipes
                f = open(fname_alt_bo_pipes, 'w')
                f.close()
                f = open(fname_alt_tpot_pipes, 'w')
                f.close()
                vprint.v2(f"Processing {run_dir}")
            
            if not os.path.exists(fname_tpot_pipes):
                if len(prob_list) == 1 and len(run_idxs) == 1:
                    sys.exit("Cannot find original tpot pipes file " 
                                + f"{fname_tpot_pipes} - skipping run..")
                else:
                    vprint.verr("Cannot find original tpot pipes file " 
                                + f"{fname_tpot_pipes} - skipping run..")
                    continue
            
            if not os.path.exists(fname_tpot_prog):
                if len(prob_list) == 1 and len(run_idxs) == 1:
                    sys.exit("Cannot find original progress file " 
                                + f"{fname_tpot_prog}")
                else:
                    vprint.verr("Cannot find original progress file " 
                                + f"{fname_tpot_prog}")
                    continue
            
            # load data from existing progress file
            (seed, 
             orig_tot_gens, 
             orig_stop_gen, 
             pop_size) = u.get_run_data(fname_tpot_prog)
            
            orig_bo_evals = (orig_tot_gens - orig_stop_gen) * pop_size
            
            if force_bo_evals:
                orig_bo_evals = force_bo_evals
            
            # start timer
            t_start = time.time() 
            t_iter = t_start
            
            # compute number of tpot generations and number of optuna evals
            # (-1 to account for initial evals)
            n_tpot_gens = int(orig_stop_gen/n_iters)
            n_bo_evals = int(orig_bo_evals/n_iters)
            
            vprint.v2("Loaded data, running tpot/bo alternating algorithm with -")
            vprint.v2(f"seed: {seed}")
            vprint.v2(f"pop size: {pop_size}")
            vprint.v2(f"tpot gens per iteration: {n_tpot_gens}")
            vprint.v2(f"optuna evals per iteration: {n_bo_evals}\n")
            
            with open(fname_alt_prog, 'w') as f:
                f.write(f"TIME:{time.asctime()}\n")
                f.write(f"SEED:{seed}\n")
                f.write(f"POP SIZE:{pop_size}\n")
                f.write(f"nITERS:{n_iters}\n")
                f.write(f"TPOT GENS PER ITER:{n_tpot_gens}\n")
                f.write(f"BO EVALS PER ITER:{n_bo_evals}\n")
                f.write(f"REAL_VALS:{real_vals}\n")                
                f.write("\n")
            
            # create TPOT object
            tpot = TPOTRegressor(generations=0,
                                  population_size=pop_size, 
                                  mutation_rate=0.9, 
                                  crossover_rate=0.1, 
                                  cv=5,
                                  verbosity=tpot_verb, 
                                  config_dict=main_tpot_config_dict, 
                                  random_state=seed, 
                                  n_jobs=n_jobs,
                                  warm_start=True,
                                  max_eval_time_mins=pipe_eval_timeout)
            
            # run initialisation for tpot to make pset etc,                
            tpot._fit_init()
            
            # load tpot data up to n_tpot_gens-1 (to account for initial evals)
            tpot.evaluated_individuals_ = u.get_progress_pop(
                fname_tpot_pipes, n_tpot_gens-1)
            
            # initialise loaded population
            tpot._pop = []       
            for k,v in tpot.evaluated_individuals_.items():
                if v['generation'] == n_tpot_gens-1:
                    tpot._pop.append(
                        creator.Individual.from_string(k, tpot._pset))        
            
            # fit for 0 gens to load into model
            vprint.v2(f"{u.CYAN}\nfitting tpot model with {tpot.generations}" 
               + " generations to initialise loaded population from first "
               + f"{n_tpot_gens-1} generations (-1 to account for initial "
               + f"evaluations)..\n{u.WHITE}")
            
            # fit tpot object
            tpot.fit(X_train, y_train)
            
            vprint.v1("")
            
            # reset generations
            tpot.generations = n_tpot_gens
            
            # make main PipelinePopOpt object
            po = PipelinePopOpt(tpot, vprint=vprint, real_vals=real_vals)
            
            old_eval_list = []
            
            for i in range(n_iters):
                vprint.v1(f"{u.CYAN_U}Iteration: {i}{u.OFF}")
                
                # get best pipe from tpot population and its 
                # corresponding string and params
                best_iter_idx, best_iter_cv = po.get_best_pipe_idx()
                best_iter_pipe = str(tpot._pop[best_iter_idx])
                best_iter_params = po.get_params(best_iter_idx)
                
                best_tpot_pipe = ""
                best_tpot_cv = -1e20
                with open(fname_alt_tpot_pipes,'a') as f:
                    for k,v in tpot.evaluated_individuals_.items():
                        if k not in old_eval_list:
                            f.write(f"{k};{i};{v['generation']};"
                                    + f"{v['internal_cv_score']}\n")
                        # overall best tpot
                        if v['internal_cv_score'] > best_tpot_cv:
                            best_tpot_pipe = k
                            best_tpot_cv = v['internal_cv_score']
                
                # copy evaluated list so we dont report same twice
                old_eval_list = list(tpot.evaluated_individuals_.keys())
                
                vprint.v1(f"\n{u.YELLOW}best pipe found by tpot for iteration "
                          + f"{i}:{u.OFF}")
                vprint.v2(f"{best_iter_pipe}")
                vprint.v1(f"{u.GREEN}* score: {u.OFF}{best_iter_cv}\n")
                
                # get all pipelines that match the structure of best pipe
                matching_strs = po.get_matching_structures(best_iter_pipe)
                
                # reset tpot_config for new tpot object
                bo_tpot_config_dict = copy.deepcopy(default_tpot_config_dict)
                
                # create BO TPOT object 
                bo_tpot = TPOTRegressor(generations=0,
                                      population_size=1, 
                                      mutation_rate=0.9, 
                                      crossover_rate=0.1, 
                                      cv=5,
                                      verbosity=tpot_verb, 
                                      config_dict=bo_tpot_config_dict, 
                                      random_state=seed, 
                                      n_jobs=n_jobs,
                                      warm_start=True,
                                      max_eval_time_mins=pipe_eval_timeout)
                
                # initialise bo tpot object to generate pset
                bo_tpot._fit_init()
                
                bo_tpot.evaluated_individuals_ = {
                    best_iter_pipe:{
                        'internal_cv_score':best_iter_cv,
                        'operator_count':len(u.string_to_ops(best_iter_pipe))}}
                
                # convert matching pipe strings to parameter sets
                seed_samples = [(u.string_to_params(best_iter_pipe),best_iter_cv)]
                for pipe_str in matching_strs:
                    bo_tpot.evaluated_individuals_[pipe_str] = {
                        'internal_cv_score':
                            tpot.evaluated_individuals_[pipe_str]['internal_cv_score'],
                        'operator_count':len(u.string_to_ops(best_iter_pipe))
                        }
                    seed_samples.append((u.string_to_params(pipe_str),
                                        tpot.evaluated_individuals_[pipe_str]['internal_cv_score']))
                
                # initialise bo pipe optimiser object
                bo_po = PipelinePopOpt(bo_tpot, vprint=vprint, real_vals=real_vals)
                
                # update pset of BO tpot object
                for (p,v) in best_iter_params:
                    bo_po.add_param_to_pset(p, v)
                
                # remove generated pipeline and transplant saved from before
                bo_tpot._pop = [creator.Individual.from_string(best_iter_pipe, bo_tpot._pset)]
                
                # fit for 0 gens to load into model
                vprint.v2(f"{u.CYAN}\n"
                          + f"({time.strftime('%d %b, %H:%M', time.localtime())})" 
                          + " fitting temporary bo tpot model with "  
                          + f"{bo_tpot.generations} generations..\n{u.WHITE}")
                
                bo_tpot.fit(X_train, y_train)
                
                vprint.v1("")
                
                vprint.v2("Transplanting best pipe and optimising for " 
                          + f"{n_bo_evals} evaluations..\n")
                
                # re-initialise bo pipe optimiser object with new values
                bo_po = PipelinePopOpt(bo_tpot, vprint=vprint, real_vals=real_vals)
                
                # run bayesian optimisation with seed_dicts as initial samples
                bo_po.optimise(0, X_train, y_train, n_evals=n_bo_evals,
                                   seed_samples=seed_samples,real_vals=real_vals, 
                                   timeout_trials=optuna_timeout_trials)
            
                best_bo_pipe = ""
                best_bo_cv = -1e20
                with open(fname_alt_bo_pipes,'a') as f:
                    for k,v in bo_tpot.evaluated_individuals_.items():
                        f.write(f"{k};{i};{v['internal_cv_score']}\n")
                        # overall best BO
                        if v['internal_cv_score'] > best_bo_cv:
                            best_bo_pipe = k
                            best_bo_cv = v['internal_cv_score']
                
                vprint.v1(f"{u.YELLOW}* best pipe found by BO:{u.OFF}")
                vprint.v2(f"{best_bo_pipe}")
                vprint.v1(f"{u.GREEN}* score: {u.OFF}{best_bo_cv}\n")
                
                vprint.v1(f"{u.YELLOW}* best pipe found by tpot for iteration "
                         + f"{i}:{u.OFF}")
                vprint.v2(f"{best_iter_pipe}")
                vprint.v1(f"{u.GREEN}* score: {u.OFF}{best_iter_cv}\n")
                
                bo_success = best_bo_cv > best_iter_cv
                
                if bo_success:
                    vprint.v1(f"{u.GREEN}BO successful!{u.OFF}")
                    # update main pset with new individual
                    vprint.v2("updating main pset..")
                    new_params = u.string_to_params(best_bo_pipe)
                    for (p,v) in new_params:
                        po.add_param_to_pset(p, v)
                
                    # create new pipe object with best params
                    new_pipe = creator.Individual.from_string(best_bo_pipe, tpot._pset)
                
                    vprint.v2(f"replacing pipe {best_iter_idx} with best BO pipe and re-evaluating..\n")
                
                    tpot._pop[best_iter_idx] = new_pipe
                    
                    # add to evaluated individuals
                    tpot.evaluated_individuals_[best_bo_pipe] = bo_tpot.evaluated_individuals_[best_bo_pipe]
                    
                    # evaluate new pipe
                    po.evaluate(best_iter_idx, X_train, y_train)
                    
                    vprint.v1(f"CV of new evaluated tpot pipe: {tpot._pop[best_iter_idx].fitness.values[1]}")
                    
                    # set new 
                    tpot.evaluated_individuals_[best_bo_pipe]['generation'] = -1
                else:
                    vprint.v1(f"{u.RED}BO unsuccessful - reverting to original"
                              + f" TPOT population..{u.OFF}")
                    
                t_iter_old = t_iter
                t_iter = time.time()
                
                with open(fname_alt_prog, 'a') as f:
                    f.write(f"{time.strftime('%d %b, %H:%M', time.localtime())}\n")
                    f.write(f"****** ITERATION {i} ******\n")
                    f.write(f"Best TPOT pipe:\n{best_iter_pipe}\n")
                    f.write(f"Best TPOT CV:{best_iter_cv}\n\n")
                    f.write(f"Best BO pipe:\n{best_bo_pipe}\n")
                    f.write(f"Best BO CV:{best_bo_cv}\n\n")
                    f.write(f"BO successful:{bo_success}\n")
                    f.write(f"Time elapsed:{round(t_iter-t_iter_old,2)}\n")
                    f.write(f"Total time elapsed:{round(t_iter-t_start,2)}\n")
                    f.write("\n")
                
                if i < n_iters-1:
                    vprint.v2(f"{u.CYAN}\nfitting tpot model with "
                              + f"{tpot.generations} generations..\n{u.WHITE}")
                    # fit tpot object
                    tpot.fit(X_train, y_train)
                    
                    vprint.v1("")
                    
            t_end = time.time() 
            
            vprint.v1(f"\n{u.YELLOW}* best pipe found by tpot:{u.OFF}")
            vprint.v1(f"{best_tpot_pipe}")
            vprint.v1(f"{u.GREEN} * score:{u.OFF} {best_tpot_cv}")
            vprint.v1(f"\n{u.YELLOW}best pipe found by BO:{u.OFF}")
            vprint.v1(f"{best_bo_pipe}\n{u.GREEN}* score:{u.OFF} {best_bo_cv}")
            vprint.v1(f"\nTotal time elapsed: {round(t_end-t_start,2)} sec\n")
            
            
class TestHandler(object):
    def __init__(self, params):
        self.params = params
        
        self.t_start = time.time()
        
        if params['RUN_TPOT'] and type(params['RUNS']) is not int:
            sys.exit('Cannot specify list of runs when generating new TPOT data')       
        
        self.run_list = range(params['RUNS']) if type(params['RUNS']) is int else params['RUNS']
        
        # set up verbosity printer
        self.vprint = u.Vprint(params['VERBOSITY'])
        
        cwd = os.getcwd()        
        
        self.data_path = os.path.join(cwd,params['DATA_DIR'])
        if not os.path.exists(self.data_path):
            sys.exit(f"Cannot find data directory {self.data_path}") 
            
        # if problem list is empty, search data directory for .data files
        if len(params['PROBLEMS']) == 0:
            self.prob_list = [f.split(".")[0] 
                              for f in os.listdir(self.data_path) 
                              if f.endswith(".data")]
        else:
            self.prob_list = params['PROBLEMS']
        
        self.results_path = os.path.join(cwd, params['RESULTS_DIR'])
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)
                    
        # if CLEAN_DATA flag is set, confirm and delete data
        if params['CLEAN_DATA'] and not params['RUN_TPOT']:
            rmv_txt = ("BO and alt" if params['RUN_BO'] and params['RUN_ALT'] 
                       else "BO" if params['RUN_BO'] 
                       else "alt" if params['RUN_ALT'] 
                       else "nothing (check 'CLEAN_DATA' flag)")
            self.vprint.vwarn(f"about to remove {rmv_txt} data from runs:\n"
                              + f"{self.run_list}\n"
                              +f"of problem(s):\n{self.prob_list}\n")
            rmv_conf = input("Are you sure you want to do this? [y/N] ")
            if rmv_conf in "yY":
                self.clean_data()
            else:
                sys.exit(f"Exiting.. Make sure to check CLEAN_DATA flag!")
            
        self.fname_prog = os.path.join(self.results_path,"progress.out")          

        with open(self.fname_prog, 'w') as f:
            f.write(f"TIME STARTED:{time.asctime()}\n")
            f.write(f"USER SPECIFIED PARAMETERS:\n")
            for k,v in params.items():
                if 'CONFIG' in k and v == default_tpot_config_dict:
                    f.write(f"{k}:default_tpot_config_dict\n")
                else:
                    f.write(f"{k}:{v}\n")
            
    def clean_data(self):
        self.vprint.vwarn("Cleaning data..")
        for prob in self.prob_list:
            for run in self.run_list:
                run_str = str(run) if run > 9 else f"0{run}"
                run_path = prob_dir = os.path.join(self.results_path, prob, f"Run_{run_str}")
                if self.params['RUN_BO']: 
                    rmtree(os.path.join(run_path,"bo"),ignore_errors=True)
                if self.params['RUN_ALT']: 
                    rmtree(os.path.join(run_path,"alt"),ignore_errors=True)
        self.vprint.v0("Done!\n")
        cont_conf = input("Do you want to continue executing the script? [Y/n] ")
        if cont_conf in "nN":
            sys.exit(f"Exiting..")
            
    def write_problem(self, problem):
        with open(self.fname_prog, 'a') as f:
            f.write(f"\n****** {problem} ******\n")

    def write_end(self):
        t_end = time.time()
        with open(self.fname_prog, 'a') as f:
            f.write(f"\nTests complete!\nTotal time elapsed: {round(t_end-self.t_start,2)}s\n")

    def generate_tpot_data(self, run_idx, problem):
        try:
            t_tpot_start = time.time()
            self.vprint.v1(f"{u.CYAN_U}****** Generating tpot data for problem '{problem}' ******{u.OFF}\n")
            new_run = get_tpot_data(pop_size=self.params['POP_SIZE'],
                          tot_gens=self.params['nTOTAL_GENS'],
                          stop_gen=self.params['STOP_GEN'],
                          n_runs=1,
                          start_seed=self.params['START_SEED'] + run_idx,
                          prob_list=[problem],
                          data_dir=self.params['DATA_DIR'],
                          results_dir=self.params['RESULTS_DIR'],
                          tpot_config_dict=self.params['TPOT_CONFIG_DICT'],
                          n_jobs=self.params['nJOBS'],
                          vprint=self.vprint,
                          pipe_eval_timeout=self.params['PIPE_EVAL_TIMEOUT'])
            t_tpot_end = time.time()
            self.write_run(new_run)
            with open(self.fname_prog, 'a') as f:
                f.write(f"({time.strftime('%d %b, %H:%M', time.localtime())}) Generate TPOT data (run {new_run}): Successful ({round(t_tpot_end-t_tpot_start,2)}s)\n")
        except:
            trace = traceback.format_exc()
            self.vprint.verr(f"FAILED:\n{trace}")
            with open(self.fname_prog, 'a') as f:
                f.write(f"Generate TPOT data (run {run_idx}): Failed..\n{trace}\n\n")
            return None
        
        return new_run
    
    def run_BO(self, run, problem):
        try:
            t_bo_start = time.time()
            self.vprint.v1(f"{u.CYAN_U}****** Running BO (run {run}) for problem '{problem}' ******{u.OFF}\n")
            # run BO on generated TPOT data
            run_bo(run_list=[run],
                        optuna_timeout_trials=self.params['OPTUNA_TIMEOUT_TRIALS'],
                        prob_list=[problem],
                        data_dir=self.params['DATA_DIR'],
                        results_dir=self.params['RESULTS_DIR'],
                        tpot_config_dict=self.params['TPOT_CONFIG_DICT'],
                        n_jobs=self.params['nJOBS'],
                        vprint=self.vprint,
                        real_vals=self.params['REAL_VALS'],
                        pipe_eval_timeout=self.params['PIPE_EVAL_TIMEOUT'])
            t_bo_end = time.time()
            with open(self.fname_prog, 'a') as f:
                f.write(f"({time.strftime('%d %b, %H:%M', time.localtime())}) Run {run} (BO): Successful ({round(t_bo_end-t_bo_start,2)}s)\n")
        except:
            trace = traceback.format_exc()
            self.vprint.verr(f"FAILED:\n{trace}")
            with open(self.fname_prog, 'a') as f:
                f.write(f"Run {run} (BO): Failed..\n{trace}\n\n")

    
    def run_alt(self, run, problem):
        try:
            t_alt_start = time.time()
            self.vprint.v1(f"{u.CYAN_U}****** Running TPOT + BO alternating (run {run}) for problem '{problem}' ******{u.OFF}\n")
            # run TPOT + BO alternating
            run_tpot_bo_alt(n_iters=self.params['nALT_ITERS'],
                            run_list=[run],
                            optuna_timeout_trials=self.params['OPTUNA_TIMEOUT_TRIALS'],
                            prob_list=[problem],
                            data_dir=self.params['DATA_DIR'],
                            results_dir=self.params['RESULTS_DIR'],
                            tpot_config_dict=self.params['TPOT_CONFIG_DICT'],
                            n_jobs=self.params['nJOBS'],
                            vprint=self.vprint,
                            real_vals=self.params['REAL_VALS'],
                            pipe_eval_timeout=self.params['PIPE_EVAL_TIMEOUT'])
            t_alt_end = time.time()
            with open(self.fname_prog, 'a') as f:
                f.write(f"({time.strftime('%d %b, %H:%M', time.localtime())}) Run {run} (alt): Successful ({round(t_alt_end-t_alt_start,2)}s)\n")
        except:
            trace = traceback.format_exc()
            self.vprint.verr(f"FAILED:\n{trace}")
            with open(self.fname_prog, 'a') as f:
                f.write(f"Run {run} (BO): Failed..\n{trace}\n\n")
