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
from BO_TPOT.tpot_bo_nd import TPOT_BO_ND
from BO_TPOT.tpot_bo_h import TPOT_BO_H, TPOT_BO_Hs
from BO_TPOT.tpot_bo_alt import TPOT_BO_ALT
from BO_TPOT.tpot_bo_auto import TPOT_BO_AUTO
from config.tpot_config import default_tpot_config_dict

class TestHandler(object):
    def __init__(self, params):
        self.params = params
        
        self.t_start = time.time()
        
        if params['RUN_TPOT-BASE'] and type(params['RUNS']) is not int:
            sys.exit('Cannot specify list of runs when generating new TPOT data')       
        
        self.run_list = range(params['RUNS']) if type(params['RUNS']) is int else params['RUNS']
        
        self.seed = params['START_SEED']
        
        self.disc_txt = "discrete" if params['DISCRETE_MODE'] else "continuous"
        
        # set up verbosity printer
        self.vprint = u.Vprint(params['VERBOSITY'])
        
        cwd = os.getcwd()        
        
        self.data_path = os.path.join(cwd,params['DATA_DIR'])
        if not os.path.exists(self.data_path):
            sys.exit(f"Cannot find data directory {self.data_path}") 
            
        self.prob_list = params['PROBLEMS']
        
        self.results_path = os.path.join(cwd, params['RESULTS_DIR'])
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)
                    
        # if CLEAN_DATA flag is set, confirm and delete data
        if params['CLEAN_DATA']: 
            if not params['RUN_TPOT-BASE']:
                rmv_txt = []
                if params['RUN_TPOT-BO-S']: rmv_txt.append("TPOT-BO-S")
                if params['RUN_TPOT-BO-H']: rmv_txt.append("TPOT-BO-H")
                if params['RUN_TPOT-BO-ND']: rmv_txt.append("TPOT-BO-ND")
                if params['RUN_TPOT-BO-Sr']: rmv_txt.append("TPOT-BO-Sr")
                if params['RUN_TPOT-BO-ALT']: rmv_txt.append("TPOT-BO-ALT")
                if params['RUN_TPOT-BO-AUTO']: rmv_txt.append("TPOT-BO-AUTO")
                self.vprint.vwarn(f"about to remove {rmv_txt} data from {self.disc_txt} runs:\n"
                                  + f"{self.run_list}\n"
                                  +f"of problem(s):\n{self.prob_list}\n")
                rmv_conf = input("Are you sure you want to do this? [y/N] ")
                if rmv_conf in "yY":
                    self.clean_data()
            else:
                sys.exit("Exiting.. Make sure to check CLEAN_DATA flag!")
            
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
            
    def clean_data(self):
        self.vprint.vwarn("Cleaning data..")
        for prob in self.prob_list:
            prob_path = os.path.join(self.results_path, prob)
            for run in self.run_list:
                run_str = str(run) if run > 9 else f"0{run}"
                run_path = os.path.join(prob_path, f"Run_{run_str}")
                if self.params['RUN_TPOT-BO-S']: 
                    rmtree(os.path.join(run_path,"TPOT-BO-S",self.disc_txt),ignore_errors=True)
                if self.params['RUN_TPOT-BO-H']: 
                    rmtree(os.path.join(run_path,"TPOT-BO-H",self.disc_txt),ignore_errors=True)
                if self.params['RUN_TPOT-BO-ND']: 
                    rmtree(os.path.join(run_path,"TPOT-BO-ND",self.disc_txt),ignore_errors=True)
                if self.params['RUN_TPOT-BO-Sr']:
                    rmtree(os.path.join(run_path,"TPOT-BO-Sr",self.disc_txt),ignore_errors=True)
                if self.params['RUN_TPOT-BO-ALT']: 
                    rmtree(os.path.join(run_path,"TPOT-BO-ALT",self.disc_txt),ignore_errors=True)
                if self.params['RUN_TPOT-BO-AUTO']: 
                    rmtree(os.path.join(run_path,"TPOT-BO-AUTO",self.disc_txt),ignore_errors=True)
        self.vprint.v0("Done!\n")
        cont_conf = input("Do you want to continue executing the script? [Y/n] ")
        if cont_conf in "nN":
            sys.exit("Exiting..")

    def write_end(self):
        t_end = time.time()
        with open(self.fname_prog, 'a') as f:
            f.write(f"\nTests complete!\nTotal time elapsed: {round(t_end-self.t_start,2)}s\n")

    def run_TPOT_BASE(self):
        run = self.run_path.split("_")[-1]
        tpot_path = os.path.join(self.run_path,'TPOT-BASE')
        if not os.path.exists(tpot_path):
            os.makedirs(tpot_path)
        try:
            t_tpot_start = time.time()
            self.vprint.v1(f"{u.CYAN_U}****** Generating tpot data for problem '{self.problem}' ******{u.OFF}\n")
            
            tb = TPOT_Base(n_gens=self.params['nTOTAL_GENS'],
                           pop_size=self.params['POP_SIZE'],
                           seed=self.seed,
                           config_dict=self.params['TPOT_CONFIG_DICT'],
                           n_jobs=self.params['nJOBS'],
                           pipe_eval_timeout=self.params['PIPE_EVAL_TIMEOUT'],
                           vprint=self.vprint)
            
            res_txt = tb.optimize(self.X_train, self.y_train, out_path=tpot_path)
            t_tpot_end = time.time()
            
            with open(self.fname_prog, 'a') as f:
                f.write(f"({time.strftime('%d %b, %H:%M', time.localtime())}) - {self.problem} - TPOT-BASE (run {run}): {res_txt} ({round(t_tpot_end-t_tpot_start,2)}s)\n")
        except:
            trace = traceback.format_exc()
            self.vprint.verr(f"FAILED:\n{trace}")
            with open(self.fname_prog, 'a') as f:
                f.write(f"({time.strftime('%d %b, %H:%M', time.localtime())}) - TPOT-BASE (run {run}): Failed..\n{trace}\n\n")
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

        def run_TPOT_BO_S(self, init_pipes, restricted_hps=False):
            run = self.run_path.split("_")[-1]
            init_bo_pipes = u.truncate_pop(init_pipes, self.params['STOP_GEN'])        
            
            r_txt = "r" if restricted_hps else ""
            
            bo_dir = f"TPOT-BO-S{r_txt}"
            bo_path = os.path.join(self.run_path,bo_dir,self.disc_txt)
            
            if not os.path.exists(bo_path):
                os.makedirs(bo_path)
            
            n_bo_evals = ((self.params['nTOTAL_GENS'] - self.params['STOP_GEN']) 
                          * self.params['POP_SIZE'])
            
            try:
                t_bo_start = time.time()
                self.vprint.v1(f"{u.CYAN_U}****** Running TPOT-BO-S{r_txt} (run {run}) for problem '{self.problem}' ******{u.OFF}\n")
                # run BO on generated TPOT data
                tbs = TPOT_BO_S(init_bo_pipes,
                                seed=self.seed,
                                n_bo_evals=n_bo_evals,
                                discrete_mode=self.params['DISCRETE_MODE'],
                                restricted_hps=restricted_hps,
                                optuna_timeout_trials=self.params['OPTUNA_TIMEOUT_TRIALS'],
                                config_dict=self.params['TPOT_CONFIG_DICT'],
                                pipe_eval_timeout=self.params['PIPE_EVAL_TIMEOUT'],
                                vprint=self.vprint)
                
                res_txt = tbs.optimize(self.X_train, self.y_train, out_path=bo_path)
                    
                t_bo_end = time.time()
                
                with open(self.fname_prog, 'a') as f:
                    f.write(f"({time.strftime('%d %b, %H:%M', time.localtime())}) - {self.problem} - Run {run} (TPOT-BO-S{r_txt}): {res_txt} ({round(t_bo_end-t_bo_start,2)}s)\n")
            except:
                trace = traceback.format_exc()
                self.vprint.verr(f"FAILED:\n{trace}")
                with open(self.fname_prog, 'a') as f:
                    f.write(f"({time.strftime('%d %b, %H:%M', time.localtime())}) - Run {run} (TPOT-BO-S{r_txt}): Failed..\n{trace}\n\n")
                return
                    
            fname_bo_prog = os.path.join(bo_path,f'TPOT-BO-S{r_txt}.progress')
            best_bo_pipe,best_bo_cv = u.get_best(tbs.pipes)
            best_init_pipe, best_init_cv = u.get_best(init_bo_pipes)
            
            # write final results to prog file
            with open(fname_bo_prog, 'w') as f:
                # update progress file for validation
                f.write(f"TPOT-BO-S{r_txt}\n")
                f.write(f"TIME:{time.asctime()}\n")
                f.write(f"SEED:{tbs.seed}\n")
                f.write(f"TOTAL TPOT GENS:{self.params['nTOTAL_GENS']}\n")
                f.write(f"TPOT STOP GEN:{self.params['STOP_GEN']}\n")
                f.write(f"BAYESIAN OPTIMISATION EVALS:{n_bo_evals}\n")
                f.write(f"DISCRETE_MODE:{tbs.discrete_mode}\n")
                f.write(f"RESTRICTED:{restricted_hps}\n")
                f.write(f"BO_PARAMS:{tbs.n_params}\n")
                f.write(f"FROZEN:{tbs.n_freeze}\n")
                
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
    
    
    def run_TPOT_BO_H(self, init_pipes, restricted_hps=False):
        run = self.run_path.split("_")[-1]
        init_tbh_pipes = u.truncate_pop(copy.deepcopy(init_pipes), self.params['STOP_GEN'])        
        
        r_txt = "r" if restricted_hps else ""
        
        tbh_dir = f"TPOT-BO-H{r_txt}"
        tbh_path = os.path.join(self.run_path,tbh_dir,self.disc_txt)
        
        if not os.path.exists(tbh_path):
            os.makedirs(tbh_path)
        
        n_bo_evals = ((self.params['nTOTAL_GENS'] - self.params['STOP_GEN']) 
                      * self.params['POP_SIZE'])
        
        try:
            t_tbh_start = time.time()
            self.vprint.v1(f"{u.CYAN_U}****** Running TPOT-BO-H{r_txt} - {self.disc_txt} (run {run}) for problem '{self.problem}' ******{u.OFF}\n")
            # run BO on generated TPOT data
            tbh = TPOT_BO_H(init_tbh_pipes,
                            seed=self.seed,
                            pop_size=self.params['POP_SIZE'],
                            n_bo_evals=n_bo_evals,
                            discrete_mode=self.params['DISCRETE_MODE'],
                            restricted_hps=restricted_hps,
                            optuna_timeout_trials=self.params['OPTUNA_TIMEOUT_TRIALS'],
                            config_dict=self.params['TPOT_CONFIG_DICT'],
                            pipe_eval_timeout=self.params['PIPE_EVAL_TIMEOUT'],
                            vprint=self.vprint)
            
            res_txt = tbh.optimize(self.X_train, self.y_train, out_path=tbh_path)
                
            t_tbh_end = time.time()
            
            with open(self.fname_prog, 'a') as f:
                f.write(f"({time.strftime('%d %b, %H:%M', time.localtime())}) - {self.problem} - Run {run} {self.problem} (TPOT-BO-H{r_txt} - {self.disc_txt}): {res_txt} ({round(t_tbh_end-t_tbh_start,2)}s)\n")
        except:
            trace = traceback.format_exc()
            self.vprint.verr(f"FAILED:\n{trace}")
            with open(self.fname_prog, 'a') as f:
                f.write(f"({time.strftime('%d %b, %H:%M', time.localtime())}) - Run {run} {self.problem} (TPOT-BO-H{r_txt} - {self.disc_txt}): Failed..\n{trace}\n\n")
            return
                
        fname_bo_prog = os.path.join(tbh_path,f'TPOT-BO-H{r_txt}.progress')
        
        best_bo_pipe,best_bo_cv = u.get_best(tbh.pipes)
        best_init_pipe, best_init_cv = u.get_best(init_tbh_pipes)
        
        # write final results to prog file
        with open(fname_bo_prog, 'w') as f:
            # update progress file for validation
            f.write(f"TPOT-BO-H{r_txt} - {res_txt}\n")
            f.write(f"TIME:{time.asctime()}\n")
            f.write(f"SEED:{tbh.seed}\n")
            f.write(f"TOTAL TPOT GENS:{self.params['nTOTAL_GENS']}\n")
            f.write(f"TPOT STOP GEN:{self.params['STOP_GEN']}\n")
            f.write(f"BAYESIAN OPTIMISATION EVALS:{n_bo_evals}\n")
            f.write(f"DISCRETE_MODE:{tbh.discrete_mode}\n")
            f.write(f"RESTRICTED:{restricted_hps}\n")
            # f.write(f"BO_PARAMS:{tbh.n_params}\n")
            # f.write(f"FROZEN:{tbh.n_freeze}\n")
            
            f.write("\n")
            f.write(f"***** AFTER {self.params['STOP_GEN']} INITIAL TPOT " 
                    + "GENERATIONS *****\n")
            f.write(f"Best CV:{best_init_cv}\n")
            f.write("Best pipeline:\n")
            f.write(f"{best_init_pipe}\n")
            f.write("\n")
            f.write(f"\n***** AFTER {n_bo_evals} BAYESIAN OPTIMISATION *****\n")
            f.write(f"Time elapsed:{round(t_tbh_end-t_tbh_start,2)}\n")
            f.write(f"Best CV:{best_bo_cv}\n")
            f.write("Best pipeline:\n")
            f.write(f"{best_bo_pipe}\n")
            
            
    def run_TPOT_BO_Hs(self, init_pipes):
        run = self.run_path.split("_")[-1]
        tbh_dir = "TPOT-BO-H"
        tbh_path = os.path.join(self.run_path,tbh_dir)
        tbhd_path = os.path.join(tbh_path,"discrete")
        tbhd_pipes = os.path.join(tbhd_path,"TPOT-BO-H.pipes")
        tbhs_path = os.path.join(tbh_path,"sequential")
        
        # load previous pipes
        init_tbh_pipes = u.load_bhs_pipes(tbhd_pipes)
        
        n_tbh_pipes = len([k for k,v in init_tbh_pipes.items() if 'TPOT-BO-H' in v['source']])
        
        if not os.path.exists(tbhs_path):
            os.makedirs(tbhs_path)
        
        n_bo_evals = ((self.params['nTOTAL_GENS']-self.params['STOP_GEN']) * self.params['POP_SIZE'] - n_tbh_pipes)
        
        
        try:
            t_tbhs_start = time.time()
            self.vprint.v1(f"{u.CYAN_U}****** Running TPOT-BO-Hs - {self.disc_txt} (run {run}) for problem '{self.problem}' ******{u.OFF}\n")
            # run BO on generated TPOT data
            tbhs = TPOT_BO_Hs(init_tbh_pipes,
                            seed=self.seed,
                            pop_size=self.params['POP_SIZE'],
                            n_bo_evals=n_bo_evals,
                            optuna_timeout_trials=self.params['OPTUNA_TIMEOUT_TRIALS'],
                            config_dict=self.params['TPOT_CONFIG_DICT'],
                            pipe_eval_timeout=self.params['PIPE_EVAL_TIMEOUT'],
                            vprint=self.vprint)
            
            res_txt = tbhs.optimize(self.X_train, self.y_train, out_path=tbhs_path)
                
            t_tbhs_end = time.time()
            
            with open(self.fname_prog, 'a') as f:
                f.write(f"({time.strftime('%d %b, %H:%M', time.localtime())}) - {self.problem} - Run {run} {self.problem} (TPOT-BO-Hs - sequential): {res_txt} ({round(t_tbhs_end-t_tbhs_start,2)}s)\n")
        except:
            trace = traceback.format_exc()
            self.vprint.verr(f"FAILED:\n{trace}")
            with open(self.fname_prog, 'a') as f:
                f.write(f"({time.strftime('%d %b, %H:%M', time.localtime())}) - Run {run} {self.problem} (TPOT-BO-Hs - sequential): Failed..\n{trace}\n\n")
            return
                
        fname_bo_prog = os.path.join(tbhs_path,f'TPOT-BO-Hs.progress')
        
        # best_bo_pipe,best_bo_cv = u.get_best(tbhs.pipes)
        # best_init_pipe, best_init_cv = u.get_best(init_tbh_pipes)
        
        best_tpot_pipe, best_tpot_cv = u.get_best(tbhs.pipes, source='TPOT-BASE')
        best_tbh_pipe, best_tbh_cv = u.get_best(tbhs.pipes, source='TPOT-BO-H')
        best_tbhs_pipe, best_tbhs_cv = u.get_best(tbhs.pipes, source='TPOT-BO-Hs')
        
        if best_tpot_cv > best_tbh_cv:
            best_tbh_pipe, best_tbh_cv = best_tpot_pipe, best_tpot_cv
        
        if best_tbh_cv > best_tbhs_cv:
            best_tbhs_pipe, best_tbhs_cv = best_tbh_pipe, best_tbh_cv
        
        # write final results to prog file
        with open(fname_bo_prog, 'w') as f:
            # update progress file for validation
            f.write("TPOT-BO-Hs\n")
            f.write(f"TIME:{time.asctime()}\n")
            f.write(f"SEED:{tbhs.seed}\n")
            f.write(f"TOTAL TPOT GENS:{self.params['nTOTAL_GENS']}\n")
            f.write(f"TPOT STOP GEN:{self.params['STOP_GEN']}\n")
            f.write(f"BAYESIAN OPTIMISATION EVALS:{n_bo_evals}\n")
            f.write(f"DISCRETE_MODE:sequential\n")
            # f.write(f"BO_PARAMS:{tbh.n_params}\n")
            # f.write(f"FROZEN:{tbh.n_freeze}\n")
            
            f.write("\n")
            f.write(f"***** AFTER {self.params['STOP_GEN']} INITIAL TPOT " 
                    + "GENERATIONS *****\n")
            f.write(f"Best CV:{best_tpot_cv}\n")
            f.write("Best pipeline:\n")
            f.write(f"{best_tpot_pipe}\n")
            f.write("\n")
            f.write(f"\n***** AFTER INITIAL {n_tbh_pipes} DISCRETE TPOT-BO-H EVALS *****\n")
            f.write(f"Best CV:{best_tbh_cv}\n")
            f.write("Best pipeline:\n")
            f.write(f"{best_tbh_pipe}\n")
            f.write("\n")
            f.write(f"\n***** AFTER FINAL {n_bo_evals} CONTINUOUS TPOT-BO-Hs EVALS *****\n")
            f.write(f"Time elapsed:{round(t_tbhs_end-t_tbhs_start,2)}\n")
            f.write(f"Best CV:{best_tbhs_cv}\n")
            f.write("Best pipeline:\n")
            f.write(f"{best_tbhs_pipe}\n")
    
    
    def run_TPOT_BO_ND(self, init_pipes, restricted_hps=False):
        run = self.run_path.split("_")[-1]
        init_bnd_pipes = u.truncate_pop(copy.deepcopy(init_pipes), self.params['STOP_GEN'])        
        
        r_txt = "r" if restricted_hps else ""
        
        bnd_dir = f"TPOT-BO-ND{r_txt}"
        bnd_path = os.path.join(self.run_path,bnd_dir,self.disc_txt)
        
        if not os.path.exists(bnd_path):
            os.makedirs(bnd_path)
        
        n_bo_evals = ((self.params['nTOTAL_GENS'] - self.params['STOP_GEN']) 
                      * self.params['POP_SIZE'])
        
        try:
            t_bnd_start = time.time()
            self.vprint.v1(f"{u.CYAN_U}****** Running TPOT-BO-ND{r_txt} (run {run}) for problem '{self.problem}' ******{u.OFF}\n")
            # run BO on generated TPOT data
            tbh = TPOT_BO_ND(init_bnd_pipes,
                            seed=self.seed,
                            pop_size=self.params['POP_SIZE'],
                            n_bo_evals=n_bo_evals,
                            discrete_mode=self.params['DISCRETE_MODE'],
                            restricted_hps=restricted_hps,
                            optuna_timeout_trials=self.params['OPTUNA_TIMEOUT_TRIALS'],
                            config_dict=self.params['TPOT_CONFIG_DICT'],
                            pipe_eval_timeout=self.params['PIPE_EVAL_TIMEOUT'],
                            vprint=self.vprint)
            
            res_txt = tbh.optimize(self.X_train, self.y_train, out_path=bnd_path)
                
            t_bnd_end = time.time()
            
            with open(self.fname_prog, 'a') as f:
                f.write(f"({time.strftime('%d %b, %H:%M', time.localtime())}) - {self.problem} - Run {run} (TPOT-BO-H{r_txt}): {res_txt} ({round(t_bnd_end-t_bnd_start,2)}s)\n")
        except:
            trace = traceback.format_exc()
            self.vprint.verr(f"FAILED:\n{trace}")
            with open(self.fname_prog, 'a') as f:
                f.write(f"({time.strftime('%d %b, %H:%M', time.localtime())}) - Run {run} (TPOT-BO-ND{r_txt}): Failed..\n{trace}\n\n")
            return
                
        fname_bo_prog = os.path.join(bnd_path,f'TPOT-BO-ND{r_txt}.progress')
        
        best_bo_pipe,best_bo_cv = u.get_best(tbh.pipes)
        best_init_pipe, best_init_cv = u.get_best(init_bnd_pipes)
        
        # write final results to prog file
        with open(fname_bo_prog, 'w') as f:
            # update progress file for validation
            f.write(f"TPOT-BO-ND{r_txt}\n")
            f.write(f"TIME:{time.asctime()}\n")
            f.write(f"SEED:{tbh.seed}\n")
            f.write(f"TOTAL TPOT GENS:{self.params['nTOTAL_GENS']}\n")
            f.write(f"TPOT STOP GEN:{self.params['STOP_GEN']}\n")
            f.write(f"BAYESIAN OPTIMISATION EVALS:{n_bo_evals}\n")
            f.write(f"DISCRETE_MODE:{tbh.discrete_mode}\n")
            f.write(f"RESTRICTED:{restricted_hps}\n")
            # f.write(f"BO_PARAMS:{tbh.n_params}\n")
            # f.write(f"FROZEN:{tbh.n_freeze}\n")
            
            f.write("\n")
            f.write(f"***** AFTER {self.params['STOP_GEN']} INITIAL TPOT " 
                    + "GENERATIONS *****\n")
            f.write(f"Best CV:{best_init_cv}\n")
            f.write("Best pipeline:\n")
            f.write(f"{best_init_pipe}\n")
            f.write("\n")
            f.write(f"\n***** AFTER {n_bo_evals} BAYESIAN OPTIMISATION *****\n")
            f.write(f"Time elapsed:{round(t_bnd_end-t_bnd_start,2)}\n")
            f.write(f"Best CV:{best_bo_cv}\n")
            f.write("Best pipeline:\n")
            f.write(f"{best_bo_pipe}\n")
            f.write("\n")
            f.write(f"\n***** NON-DOMINATED PIPELINES *****\n")
            for i,(grp,v) in enumerate(tbh.nd_grps.items()):
                f.write(f"ND_{i}:{v['best_pipe']};{v['n_bo_params']};{v['cv_best']}\n")


    def run_TPOT_BO_ALT(self, init_pipes=None):
        run = self.run_path.split("_")[-1]
        alt_path = os.path.join(self.run_path,'TPOT-BO-ALT',self.disc_txt)
        if not os.path.exists(alt_path):
            os.makedirs(alt_path)
        n_tpot_gens = int((self.params['nTOTAL_GENS'] - self.params['STOP_GEN'])/self.params['nALT_ITERS'])
        n_bo_evals = int(((self.params['nTOTAL_GENS'] - self.params['STOP_GEN']) 
                      * self.params['POP_SIZE'])/self.params['nALT_ITERS'])
        
        alt_init_pipes = u.truncate_pop(init_pipes, n_tpot_gens-1)
        
        try:
            t_alt_start = time.time()
            self.vprint.v1(f"{u.CYAN_U}****** Running TPOT-BO-ALT (run {run}) for problem '{self.problem}' ******{u.OFF}\n")
            # run TPOT + BO alternating
            tba = TPOT_BO_ALT(n_iters=self.params['nALT_ITERS'],
                              pop_size=self.params['POP_SIZE'],
                              n_tpot_gens=n_tpot_gens,
                              n_bo_evals=n_bo_evals,
                              seed=self.seed,
                              discrete_mode=self.params['DISCRETE_MODE'],
                              optuna_timeout_trials=self.params['OPTUNA_TIMEOUT_TRIALS'],
                              config_dict=self.params['TPOT_CONFIG_DICT'],
                              n_jobs=self.params['nJOBS'],
                              init_pipes=alt_init_pipes,
                              pipe_eval_timeout=self.params['PIPE_EVAL_TIMEOUT'],
                              vprint=self.vprint)
                                
            res_txt = tba.optimize(self.X_train, self.y_train, out_path=alt_path)
            
            t_alt_end = time.time()
            with open(self.fname_prog, 'a') as f:
                f.write(f"({time.strftime('%d %b, %H:%M', time.localtime())}) - {self.problem} - Run {run} (TPOT-BO-ALT): {res_txt} ({round(t_alt_end-t_alt_start,2)}s)\n")
        except:
            trace = traceback.format_exc()
            self.vprint.verr(f"FAILED:\n{trace}")
            with open(self.fname_prog, 'a') as f:
                f.write(f"({time.strftime('%d %b, %H:%M', time.localtime())}) - Run {run} (TPOT-BO-ALT): Failed..\n{trace}\n\n")
            return
        
        fname_alt_prog = os.path.join(alt_path,'TPOT-BO-ALT.progress')
        
        best_tpot_pipe,best_tpot_cv = u.get_best(tba.pipes,source="TPOT-BO-AUTO(TPOT)")
        best_bo_pipe,best_bo_cv = u.get_best(tba.pipes, source="TPOT-BO-AUTO(BO)")
        
        with open(fname_alt_prog, 'w') as f:
            f.write("TPOT-BO-ALT\n")
            f.write(f"TIME:{time.asctime()}\n")
            f.write(f"SEED:{tba.seed}\n")
            f.write(f"POP SIZE:{tba.pop_size}\n")
            f.write(f"nITERS:{tba.n_iters}\n")
            f.write(f"TPOT GENS PER ITER:{n_tpot_gens}\n")
            f.write(f"BO EVALS PER ITER:{n_bo_evals}\n")
            f.write(f"DISCRETE_MODE:{tba.discrete_mode}\n")                
            f.write("\n")
            f.write(f"****** ITERATION {tba.n_iters-1} ******\n")
            f.write(f"Best TPOT pipe:\n{best_tpot_pipe}\n")
            f.write(f"Best TPOT CV:{best_tpot_cv}\n\n")
            f.write(f"Best BO pipe:\n{best_bo_pipe}\n")
            f.write(f"Best BO CV:{best_bo_cv}\n\n")
            f.write(f"Total time elapsed:{round(t_alt_end-t_alt_start,2)}\n")
            f.write("\n")
        
    def run_TPOT_BO_AUTO(self, init_pipes=None):
        run = self.run_path.split("_")[-1]
        auto_path = os.path.join(self.run_path,'TPOT-BO-AUTO',self.disc_txt)
        if not os.path.exists(auto_path):
            os.makedirs(auto_path)
        auto_init_pipes = u.truncate_pop(init_pipes, 0)
        
        try:
            t_auto_start = time.time()
            self.vprint.v1(f"{u.CYAN_U}****** Running TPOT-BO-AUTO (run {run}) for problem '{self.problem}' ******{u.OFF}\n")
            # run TPOT + BO Auto
            tbat = TPOT_BO_AUTO(pop_size=self.params['POP_SIZE'],
                                n_gens=self.params['nTOTAL_GENS'],
                                seed=self.seed,
                                discrete_mode=self.params['DISCRETE_MODE'],
                                optuna_timeout_trials=self.params['OPTUNA_TIMEOUT_TRIALS'],
                                config_dict=self.params['TPOT_CONFIG_DICT'],
                                n_jobs=self.params['nJOBS'],
                                init_pipes=auto_init_pipes,
                                pipe_eval_timeout=self.params['PIPE_EVAL_TIMEOUT'],
                                vprint=self.vprint)
            
            res_txt = tbat.optimize(self.X_train, self.y_train, out_path=auto_path)
            
            t_auto_end = time.time()
            with open(self.fname_prog, 'a') as f:
                f.write(f"({time.strftime('%d %b, %H:%M', time.localtime())}) - {self.problem} - Run {run} (TPOT-BO-AUTO): {res_txt} ({round(t_auto_end-t_auto_start,2)}s)\n")
        except:
            trace = traceback.format_exc()
            self.vprint.verr(f"FAILED:\n{trace}")
            with open(self.fname_prog, 'a') as f:
                f.write(f"({time.strftime('%d %b, %H:%M', time.localtime())}) - Run {run} (TPOT-BO-AUTO): Failed..\n{trace}\n\n")
            return
        
        fname_auto_prog = os.path.join(auto_path,'TPOT-BO-AUTO.progress')
        
        best_pipe,best_cv = u.get_best(tbat.pipes)
        
        with open(fname_auto_prog, 'w') as f:
            f.write("TPOT-BO-AUTO\n")
            f.write(f"TIME:{time.asctime()}\n")
            f.write(f"SEED:{tbat.seed}\n")
            f.write(f"POP SIZE:{tbat.pop_size}\n")
            f.write(f"DISCRETE_MODE:{tbat.discrete_mode}\n")                
            f.write("\n")
            f.write(f"****** GENERATION {tbat.n_gens-1} ******\n")
            f.write(f"Best pipe:\n{best_pipe}\n")
            f.write(f"Best CV:{best_cv}\n\n")
            f.write(f"Total time elapsed:{round(t_auto_end-t_auto_start,2)}\n")
            f.write("\n")

    def update_seed(self, new_seed=None):
        if new_seed:
            self.seed = new_seed
        else:
            self.seed = self.seed + 1
        
    def set_problem(self, problem):
        self.problem = problem
        self.prob_path = os.path.join(self.results_path, problem)
        if not os.path.exists(self.prob_path):
            os.makedirs(self.prob_path)
            
        fname = problem + ".data"
        fpath = os.path.join(self.data_path, fname)
        self.X_train, self.X_test, self.y_train, self.y_test = u.load_data(fpath)
        
    def set_run(self, run=None):
        if run is None:
            run = 0
            exist_runs = [int(f.path.split("_")[-1]) 
                          for f in os.scandir(self.prob_path) if f.is_dir() 
                          and "Plots" not in f.path]
            if len(exist_runs) > 0:
                run = max(exist_runs) + 1
        
        run_str = f"{run}" if run > 9 else f"0{run}"
        self.run_path = os.path.join(self.prob_path, f"Run_{run_str}")
        
    def load_TPOT_data(self):
        tpot_path = os.path.join(self.run_path,'TPOT-BASE')
        pipe_path = os.path.join(tpot_path,'TPOT-BASE.pipes')
        prog_path = os.path.join(tpot_path,'TPOT-BASE.progress')
        pipes = u.get_progress_pop(pipe_path)
        seed, n_tot_gens, tpot_stop_gen, pop_size = u.get_run_data(prog_path)
        return seed, pop_size, pipes
        