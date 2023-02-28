#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 17:01:28 2022

@author: gus
"""

import time
import os
import sys
import copy
import traceback
import utils.tpot_utils as u
from shutil import rmtree
from BO_TPOT.tpot_base import TPOT_Base
from BO_TPOT.tpot_bo_s import TPOT_BO_S
from BO_TPOT.tpot_bo_alt import TPOT_BO_ALT
from BO_TPOT.tpot_bo_auto import TPOT_BO_AUTO
from config.tpot_config import default_tpot_config_dict

class TestHandler(object):
    def __init__(self, params):
        self.params = params
        
        self.t_start = time.time()
        
        if type(params['SEEDS']) is not list:
            sys.exit('Must specify list of seeds')       
        
        self.seed_list = params['SEEDS']
        
        self.disc_txt = "discrete" if params['DISCRETE_MODE'] else "continuous"
        self.type_flag = "d" if params['DISCRETE_MODE'] else "c"
        
        # set up verbosity printer
        self.vprint = u.Vprint(params['VERBOSITY'])
        
        cwd = os.getcwd()        
        
        self.data_path = os.path.join(cwd,params['DATA_DIR'])
        if not os.path.exists(self.data_path):
            sys.exit(f"Cannot find data directory {self.data_path}") 
        
        self.results_path = os.path.join(cwd, params['RESULTS_DIR'])
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)
            
        self.fname_prog = os.path.join(self.results_path,"BO_TPOT.progress")          

        with open(self.fname_prog, 'a') as f:
            f.write("====================================\n\n")
            f.write(f"TIME STARTED:{time.asctime()}\n")
            f.write("USER SPECIFIED PARAMETERS:\n")
            for k,v in params.items():
                if 'CONFIG' in k and v == default_tpot_config_dict:
                    f.write(f"{k}:default_tpot_config_dict\n")
                else:
                    f.write(f"{k}:{v}\n")

    def write_end(self):
        t_end = time.time()
        with open(self.fname_prog, 'a') as f:
            f.write(f"\nTests complete!\nTotal time elapsed: {round(t_end-self.t_start,2)}s\n")

    def run_TPOT_BASE(self, seed):
        try:
            tpot_path = self.get_path('TPOT-BASE',seed)
            t_tpot_start = time.time()
            self.vprint.v1(f"{u.CYAN_U}****** Generating tpot data for problem '{self.problem}', seed: {seed} ******{u.OFF}\n")
            
            tb = TPOT_Base(n_gens=self.params['nTOTAL_GENS'],
                           pop_size=self.params['POP_SIZE'],
                           seed=seed,
                           config_dict=self.params['TPOT_CONFIG_DICT'],
                           n_jobs=self.params['nJOBS'],
                           pipe_eval_timeout=self.params['PIPE_EVAL_TIMEOUT'],
                           vprint=self.vprint)
            
            res_txt = tb.optimize(self.X_train, self.y_train, out_path=tpot_path)
            t_tpot_end = time.time()
            
            with open(self.fname_prog, 'a') as f:
                f.write(f"({time.strftime('%d %b, %H:%M', time.localtime())}) - {self.problem} - TPOT-BASE (seed {seed}): {res_txt} ({round(t_tpot_end-t_tpot_start,2)}s)\n")
        except:
            trace = traceback.format_exc()
            self.vprint.verr(f"FAILED:\n{trace}")
            with open(self.fname_prog, 'a') as f:
                f.write(f"({time.strftime('%d %b, %H:%M', time.localtime())}) - {self.problem} - TPOT-BASE (seed {seed}): Failed..\n{trace}\n\n")
            return None
        
        fname_tpot_prog = os.path.join(tpot_path,'TPOT-BASE.progress')
        best_tpot_pipe,best_tpot_cv = u.get_best(tb.pipes)
        
        # write progress to file
        with open(fname_tpot_prog, 'w') as f:
            f.write("TPOT-BASE\n")
            f.write(f"TIME:{time.asctime()}\n")
            f.write(f"SEED:{tb.seed}\n")
            f.write(f"POP SIZE:{tb.pop_size}\n")
            f.write(f"TOTAL TPOT GENS:{tb.n_gens}\n")
            f.write("\n")
            f.write(f"***** AFTER {tb.n_gens} TPOT-BASE GENERATIONS *****\n")
            f.write(f"Time elapsed:{round(t_tpot_end-t_tpot_start,2)}\n")
            f.write(f"Best full TPOT CV:{best_tpot_cv}\n")
            f.write("Best full TPOT pipeline:\n")
            f.write(f"{best_tpot_pipe}\n")
        
        return tb.pipes

    def run_TPOT_BO_S(self, seed):
        try:
            self.pop_size, init_pipes = self.load_TPOT_data(seed)
            
            bo_path = self.get_path(f'TPOT-BO-S{self.type_flag}',seed)

            init_bo_pipes = u.truncate_pop(init_pipes, self.params['STOP_GEN'])        
            
            n_bo_evals = ((self.params['nTOTAL_GENS'] - self.params['STOP_GEN']) 
                            * self.params['POP_SIZE'])
            t_bo_start = time.time()
            self.vprint.v1(f"{u.CYAN_U}****** Running TPOT-BO-S{self.type_flag} (seed {seed}) for problem '{self.problem}' ******{u.OFF}\n")
            # run BO on generated TPOT data
            tbs = TPOT_BO_S(init_bo_pipes,
                            seed=seed,
                            n_bo_evals=n_bo_evals,
                            discrete_mode=self.params['DISCRETE_MODE'],
                            optuna_timeout_trials=self.params['OPTUNA_TIMEOUT_TRIALS'],
                            config_dict=self.params['TPOT_CONFIG_DICT'],
                            pipe_eval_timeout=self.params['PIPE_EVAL_TIMEOUT'],
                            vprint=self.vprint)
            
            res_txt = tbs.optimize(self.X_train, self.y_train, out_path=bo_path)
                
            t_bo_end = time.time()
            
            with open(self.fname_prog, 'a') as f:
                f.write(f"({time.strftime('%d %b, %H:%M', time.localtime())}) - {self.problem} - Seed {seed} (TPOT-BO-S{self.type_flag}): {res_txt} ({round(t_bo_end-t_bo_start,2)}s)\n")
        except:
            trace = traceback.format_exc()
            self.vprint.verr(f"FAILED:\n{trace}")
            with open(self.fname_prog, 'a') as f:
                f.write(f"({time.strftime('%d %b, %H:%M', time.localtime())}) - {self.problem} - Seed {seed} (TPOT-BO-S{self.type_flag}): Failed..\n{trace}\n\n")
            return
                
        fname_bo_prog = os.path.join(bo_path,f'TPOT-BO-S{self.type_flag}.progress')
        best_bo_pipe,best_bo_cv = u.get_best(tbs.pipes)
        best_init_pipe, best_init_cv = u.get_best(init_bo_pipes)
        
        # write final results to prog file
        with open(fname_bo_prog, 'w') as f:
            # update progress file for validation
            f.write(f"TPOT-BO-S{self.type_flag}\n")
            f.write(f"TIME:{time.asctime()}\n")
            f.write(f"SEED:{tbs.seed}\n")
            f.write(f"TOTAL TPOT GENS:{self.params['nTOTAL_GENS']}\n")
            f.write(f"TPOT STOP GEN:{self.params['STOP_GEN']}\n")
            f.write(f"BAYESIAN OPTIMISATION EVALS:{n_bo_evals}\n")
            f.write(f"DISCRETE_MODE:{tbs.discrete_mode}\n")
            
            f.write("\n")
            f.write(f"***** AFTER {self.params['STOP_GEN']} INITIAL TPOT " 
                    + "GENERATIONS *****\n")
            f.write(f"Best CV:{best_init_cv}\n")
            f.write("Best pipeline:\n")
            f.write(f"{best_init_pipe}\n")
            f.write("\n")
            f.write(f"\n***** AFTER {n_bo_evals} BAYESIAN OPTIMISATION *****\n")
            f.write(f"Time elapsed:{round(t_bo_end-t_bo_start,2)}\n")
            f.write(f"Best CV:{best_bo_cv}\n")
            f.write("Best pipeline:\n")
            f.write(f"{best_bo_pipe}\n")    
    
    def run_TPOT_BO_ALT(self, seed):
        try:
            alt_path = self.get_path(f'TPOT-BO-ALT{self.type_flag}',seed)
            n_tpot_gens = int(self.params['STOP_GEN']/self.params['nALT_ITERS'])
            
            # alt_init_pipes = u.truncate_pop(init_pipes, n_tpot_gens-1)
            
            t_alt_start = time.time()
            self.vprint.v1(f"{u.CYAN_U}****** Running TPOT-BO-ALT (seed {seed}) for problem '{self.problem}' ******{u.OFF}\n")
            # run TPOT + BO alternating
            tba = TPOT_BO_ALT(n_iters=self.params['nALT_ITERS'],
                              pop_size=self.params['POP_SIZE'],
                              n_tpot_gens=n_tpot_gens,
                              n_total_gens=self.params['nTOTAL_GENS'],
                              seed=seed,
                              discrete_mode=self.params['DISCRETE_MODE'],
                              optuna_timeout_trials=self.params['OPTUNA_TIMEOUT_TRIALS'],
                              config_dict=self.params['TPOT_CONFIG_DICT'],
                              n_jobs=self.params['nJOBS'],
                              pipe_eval_timeout=self.params['PIPE_EVAL_TIMEOUT'],
                              vprint=self.vprint)
                                
            res_txt = tba.optimize(self.X_train, self.y_train, out_path=alt_path)
            
            t_alt_end = time.time()
            with open(self.fname_prog, 'a') as f:
                f.write(f"({time.strftime('%d %b, %H:%M', time.localtime())}) - {self.problem} - Seed {seed} (TPOT-BO-ALT): {res_txt} ({round(t_alt_end-t_alt_start,2)}s)\n")
        except:
            trace = traceback.format_exc()
            self.vprint.verr(f"FAILED:\n{trace}")
            with open(self.fname_prog, 'a') as f:
                f.write(f"({time.strftime('%d %b, %H:%M', time.localtime())}) - {self.problem} - Seed {seed} (TPOT-BO-ALT): Failed..\n{trace}\n\n")
            return
        
        fname_alt_prog = os.path.join(alt_path,f'TPOT-BO-ALT{self.type_flag}.progress')
        
        best_tpot_pipe,best_tpot_cv = u.get_best(tba.pipes,source=f"TPOT-BO-ALT{self.type_flag}(TPOT)")
        best_bo_pipe,best_bo_cv = u.get_best(tba.pipes, source=f"TPOT-BO-ALT{self.type_flag}(BO)")
        
        with open(fname_alt_prog, 'w') as f:
            f.write("TPOT-BO-ALT\n")
            f.write(f"TIME:{time.asctime()}\n")
            f.write(f"SEED:{tba.seed}\n")
            f.write(f"POP SIZE:{tba.pop_size}\n")
            f.write(f"nITERS:{tba.n_iters}\n")
            f.write(f"TPOT GENS PER ITER:{n_tpot_gens}\n")
            f.write(f"BO EVALS PER ITER:{tba.n_bo_evals}\n")
            f.write(f"DISCRETE_MODE:{tba.discrete_mode}\n")                
            f.write("\n")
            f.write(f"****** ITERATION {tba.n_iters-1} ******\n")
            f.write(f"Best TPOT pipe:\n{best_tpot_pipe}\n")
            f.write(f"Best TPOT CV:{best_tpot_cv}\n\n")
            f.write(f"Best BO pipe:\n{best_bo_pipe}\n")
            f.write(f"Best BO CV:{best_bo_cv}\n\n")
            f.write(f"Total time elapsed:{round(t_alt_end-t_alt_start,2)}\n")
            f.write("\n")
            
    def run_TPOT_BO_AUTO(self, seed):
        try:           
            auto_path = self.get_path(f'TPOT-BO-AUTO{self.type_flag}',seed)
            
            t_auto_start = time.time()
            self.vprint.v1(f"{u.CYAN_U}****** Running TPOT-BO-AUTO (seed {seed}) for problem '{self.problem}' ******{u.OFF}\n")
            # run TPOT + BO Auto
            tbat = TPOT_BO_AUTO(pop_size=self.params['POP_SIZE'],
                                n_gens=self.params['nTOTAL_GENS'],
                                seed=seed,
                                discrete_mode=self.params['DISCRETE_MODE'],
                                optuna_timeout_trials=self.params['OPTUNA_TIMEOUT_TRIALS'],
                                config_dict=self.params['TPOT_CONFIG_DICT'],
                                n_jobs=self.params['nJOBS'],
                                pipe_eval_timeout=self.params['PIPE_EVAL_TIMEOUT'],
                                vprint=self.vprint)
            
            res_txt = tbat.optimize(self.X_train, self.y_train, out_path=auto_path)
            
            t_auto_end = time.time()
            with open(self.fname_prog, 'a') as f:
                f.write(f"({time.strftime('%d %b, %H:%M', time.localtime())}) - {self.problem} - Seed {seed} (TPOT-BO-AUTO{self.type_flag}): {res_txt} ({round(t_auto_end-t_auto_start,2)}s)\n")
        except:
            trace = traceback.format_exc()
            self.vprint.verr(f"FAILED:\n{trace}")
            with open(self.fname_prog, 'a') as f:
                f.write(f"({time.strftime('%d %b, %H:%M', time.localtime())}) - {self.problem} - Seed {seed} (TPOT-BO-AUTO{self.type_flag}): Failed..\n{trace}\n\n")
            return
        
        fname_auto_prog = os.path.join(auto_path,f'TPOT-BO-AUTO{self.type_flag}.progress')
        
        best_pipe,best_cv = u.get_best(tbat.pipes)
        
        with open(fname_auto_prog, 'w') as f:
            f.write(f"TPOT-BO-AUTO{self.type_flag}\n")
            f.write(f"TIME:{time.asctime()}\n")
            f.write(f"SEED:{tbat.seed}\n")
            f.write(f"POP SIZE:{tbat.pop_size}\n")
            f.write(f"DISCRETE_MODE:{tbat.discrete_mode}\n")                
            f.write("\n")
            f.write(f"****** GENERATION {tbat.n_gens-1} ******\n")
            f.write(f"Best pipe:\n{best_pipe}\n")
            f.write(f"Best CV:{best_cv}\n")
            f.write(f"Pipe source:{tbat.pipes[best_pipe]['source']}\n\n")
            f.write(f"Total time elapsed:{round(t_auto_end-t_auto_start,2)}\n")
            f.write("\n")
        
    def set_problem(self, problem):
        self.problem = problem
        self.prob_path = os.path.join(self.results_path, problem)
        if not os.path.exists(self.prob_path):
            os.makedirs(self.prob_path)
            
        fname = problem + ".data"
        fpath = os.path.join(self.data_path, fname)
        self.X_train, self.X_test, self.y_train, self.y_test = u.load_data(fpath)
    
    def get_path(self, method, seed):
        self.method = f"{method}" if "TPOT-BASE" not in method else method
        seed_txt = f"Seed_{seed}"
        
        seed_path = os.path.join(self.prob_path, self.method, seed_txt)
        
        if not os.path.exists(seed_path):
            os.makedirs(seed_path)     
        
        return seed_path   
        
    def load_TPOT_data(self, seed):
        tpot_path = os.path.join(self.prob_path,'TPOT-BASE')
        seed_path = os.path.join(tpot_path,f'Seed_{seed}')
        pipe_path = os.path.join(seed_path,'TPOT-BASE.pipes')
        prog_path = os.path.join(seed_path,'TPOT-BASE.progress')
        pipes = u.get_progress_pop(pipe_path)
        for v in pipes.values():
            v['source'] = 'TPOT-BASE'
        seed, n_tot_gens, tpot_stop_gen, pop_size = u.get_run_data(prog_path)
        return pop_size, pipes
        