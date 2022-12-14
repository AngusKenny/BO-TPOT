#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 13:39:58 2022

@author: gus
"""

import sys
import os
import copy
from utils import tpot_utils as u
from config.tpot_config import default_tpot_config_dict
import numpy as np
import optuna
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.lines import Line2D

RESULTS_PATH = 'Results'
PROBLEMS = [            
    # 'quake',
    # 'socmob',
    # 'abalone',
    'brazilian_houses',
    # 'house_16h',
    # 'elevators'
    ]
MODES = ['discrete']#,'continuous']
WTL = ['TPOT-BASE','TPOT-BO-S']#,'TPOT-BO-H']
RUNS = [0]
PRINT_COL = 20
SAVE_PLOTS = True
SAVE_PLOTS = False
WIN_TOL = 1e-14
ANIMATE = False
SHOW_TITLE = False
YLIM = None
# YLIM = [3.54e-6,3.685e-6]
# LEGEND_POS = 'lower right'

med_stats = {problem: {method: {mode: -1 for mode in MODES} for method in WTL} for problem in PROBLEMS}
for problem in PROBLEMS:
    med_stats[problem]['TPOT-BO-O'] = -1

wtl_stats = {problem: {method: {mode: "" for mode in MODES} for method in WTL} for problem in PROBLEMS}
for problem in PROBLEMS:
    wtl_stats[problem]['TPOT-BO-O'] = ""


for PROBLEM in PROBLEMS:
    
    cwd = os.getcwd()
    prob_path = os.path.join(cwd,RESULTS_PATH,PROBLEM)
    
    plot_path = os.path.join(prob_path,'Plots')
    
    for mode in MODES:
    
        raw_data = {}
        raw_tbo_pipes = {}
        raw_tbs_pipes = {}
        
        skip_runs = []
        
        for run in RUNS:
            run_txt = f"Run_{run}" if run > 9 else f"Run_0{run}"
            
            run_path = os.path.join(prob_path, run_txt)
            
            tb_path = os.path.join(run_path,'TPOT-BASE')
            f_tb_prog = os.path.join(tb_path,'TPOT-BASE.progress')
            f_tb_pipes = os.path.join(tb_path,'TPOT-BASE.pipes')
            
            tbs_path = os.path.join(run_path,'TPOT-BO-S',mode)
            f_tbs_prog = os.path.join(tbs_path,'TPOT-BO-S.progress')
            f_tbs_pipes = os.path.join(tbs_path,'TPOT-BO-S.pipes')
            
            tbo_path = os.path.join(run_path,'TPOT-BO-O',mode)
            f_tbo_prog = os.path.join(tbo_path,'TPOT-BO-O.progress')
            f_tbo_pipes = os.path.join(tbo_path,'TPOT-BO-O.pipes')
            
            check_files = [f_tb_prog, f_tb_pipes, f_tbs_prog, f_tbs_pipes, f_tbo_prog, f_tbo_pipes]
            
            for check_file in check_files:
                if not os.path.exists(check_file):
                    print(f"Cannot find {check_file}\nskipping {run_txt}..")
                    skip_runs.append(run)
                    break
            
            if run in skip_runs: continue
        
            raw_data[run] = {}
            raw_tbo_pipes[run] = {}
            raw_tbs_pipes[run] = {'n_bo_hp':None, 'cv_best':[]}
            
            with open(f_tb_prog, 'r') as f:
                for line in f:
                    if 'INITIAL TPOT GENERATIONS' in line:
                        cv_line = next(f)
                        raw_data[run]['init TPOT'] = -float(cv_line.split(":")[-1])
                    if 'AFTER' in line and 'INITIAL' not in line:
                        next(f)
                        cv_line = next(f)
                        raw_data[run]['TPOT-BASE'] = -float(cv_line.split(":")[-1])
            
            with open(f_tbs_prog, 'r') as f:
                for line in f:
                    if 'INITIAL TPOT GENERATIONS' in line:
                        cv_line = next(f)
                        if -float(cv_line.split(":")[-1]) != raw_data[run]['init TPOT']:
                            print(f"{run_txt} different initial TPOT values")
                    if 'AFTER' in line and 'BAYESIAN' in line:
                        next(f)
                        cv_line = next(f)
                        raw_data[run]['TPOT-BO-S'] = -float(cv_line.split(":")[-1])
            
            with open(f_tbo_prog, 'r') as f:
                for line in f:
                    if 'INITIAL TPOT GENERATIONS' in line:
                        cv_line = next(f)
                        if -float(cv_line.split(":")[-1]) != raw_data[run]['init TPOT']:
                            print(f"{run_txt} different initial TPOT values")
                    if 'AFTER' in line and 'BAYESIAN' in line:
                        next(f)
                        cv_line = next(f)
                        raw_data[run]['TPOT-BO-O'] = -float(cv_line.split(":")[-1])
                                    
            best_cvs = {}
                        
            with open(f_tbo_pipes, 'r') as f:
                for line in f:
                    ls = line.split(";")
                    pipe = ls[0]
                    struc = ls[1]
                    gen = int(ls[2])
                    n_nd = int(ls[3])
                    source = ls[4].strip("cd")
                    cv = -float(ls[5])
                           
                    if struc not in best_cvs:
                        best_cvs[struc] = cv
                    else:
                        best_cvs[struc] = min(best_cvs[struc],cv)
                    
                    if gen not in raw_tbo_pipes[run]:
                        raw_tbo_pipes[run][gen] = {}
                        
                    if struc not in raw_tbo_pipes[run][gen]:
                        raw_tbo_pipes[run][gen][struc] = {'pipe':pipe,'n_hp': len(u.string_to_params(pipe,config_dict=default_tpot_config_dict))}
                    
                    raw_tbo_pipes[run][gen][struc]['cv'] = copy.deepcopy(best_cvs[struc])
                    
            
            with open(f_tbs_pipes) as f:
                for line in f:
                    ls = line.split(";")
                    pipe = ls[0]
                    cv = -float(ls[-1])
                    
                    if not raw_tbs_pipes[run]['n_bo_hp']:
                        raw_tbs_pipes[run]['n_bo_hp'] = len(u.string_to_params(pipe,config_dict=default_tpot_config_dict))
                        raw_tbs_pipes[run]['cv_best'].append(cv)
                        raw_tbs_pipes[run]['pipe'] = pipe
                        raw_tbs_pipes[run]['struc'] = str(u.string_to_bracket(pipe))
                    else:
                        raw_tbs_pipes[run]['cv_best'].append(min(raw_tbs_pipes[run]['cv_best'][-1],cv))
                        
        last_gens = {run: max(raw_tbo_pipes[run].keys()) for run in raw_tbo_pipes}
        
        for run, v in raw_tbs_pipes.items():
            v['cv_gens'] = np.interp(np.linspace(0, len(v['cv_best']), last_gens[run]+1), range(len(v['cv_best'])), v['cv_best'])      
        
        # nd_plots = {run:np.array([[raw_tbh_pipes[run][last_gens[run]][struc]['n_bo_params'],raw_tbh_pipes[run][last_gens[run]][struc]['cv']] for struc in raw_tbh_pipes[run][last_gens[run]]]) for run in raw_tbh_pipes}                
        
        data={}
        
        data['init TPOT'] = [v['init TPOT'] for k,v in raw_data.items()]
        data['TPOT-BASE'] = [v['TPOT-BASE'] for k,v in raw_data.items()]
        data['TPOT-BO-S'] = [v['TPOT-BO-S'] for k,v in raw_data.items()]
        data['TPOT-BO-O'] = [v['TPOT-BO-O'] for k,v in raw_data.items()]
        
        stats = {'best':{},'worst':{},'median':{},'mean':{},'std dev':{}}
        
        stats['best'] = {k: np.min(v) for k,v in data.items()}
        stats['worst'] = {k: np.max(v) for k,v in data.items()}
        stats['median'] = {k: np.median(v) for k,v in data.items()}
        stats['mean'] = {k: np.mean(v) for k,v in data.items()}
        stats['std dev'] = {k: np.std(v) for k,v in data.items()}
        
        
        # find median run index (closest to median value)
        med_auto_run_idx = np.abs(data['TPOT-BO-O'] - np.median(data['TPOT-BO-O'])).argmin()
        med_run = list(raw_data.keys())[med_auto_run_idx]
        
        # print(len(raw_tbh_pipes[med_run][0].keys()))
        
        struc_idxs = {struc:i for i,struc in enumerate(raw_tbo_pipes[med_run][0].keys()) if i < 50}
        
        # for struc in struc_idxs:
        #     print(struc)
        
        struc_hps = {struc:v['n_hp'] for struc,v in raw_tbo_pipes[med_run][0].items() if struc in struc_idxs}
               
        if raw_tbs_pipes[med_run]['struc'] not in struc_idxs:
            struc_idxs[raw_tbs_pipes[med_run]['struc']] = len(struc_idxs)
        
        idx_plots = {gen:np.array([[struc_idxs[struc],raw_tbo_pipes[med_run][gen][struc]['cv']] for struc in raw_tbo_pipes[med_run][gen] if struc in struc_idxs]) for gen in raw_tbo_pipes[med_run]}
        
        # print([[struc_hps[struc],raw_tbh_pipes[med_run][gen][struc]['cv']] for struc in raw_tbh_pipes[med_run][gen] if struc in struc_idxs])
        
        hp_plots = {gen:np.array([[struc_hps[struc],raw_tbo_pipes[med_run][gen][struc]['cv']] for struc in raw_tbo_pipes[med_run][gen] if struc in struc_idxs]) for gen in raw_tbo_pipes[med_run]}
        
        # idx_plots = {gen:np.vstack(([np.hstack((struc_idxs[struc]*np.ones((len(raw_tbh_pipes[med_run][gen][struc]),1)),np.array(raw_tbh_pipes[med_run][gen][struc]).reshape(-1,1))) for struc in raw_tbh_pipes[med_run][gen]])) for gen in raw_tbh_pipes[med_run]}                
        
        if len(idx_plots[list(idx_plots.keys())[-1]]) == 0:
            idx_plots.pop(list(idx_plots.keys())[-1])
        
        # print(f"\n{idx_plots}\n")
        
        ymax = np.max(idx_plots[0][:,1])
        ymin = np.min(idx_plots[list(idx_plots.keys())[-1]][:,1])
        
        xmax = np.max(idx_plots[0][:,0])
        
        ylims = [ymin-(ymax-ymin)*0.1,ymax + (ymax-ymin)*0.1]
        xlims = [0,xmax + 1]
        
        colour_grads = np.linspace(0,1,last_gens[med_run]+1)
        # blues = plt.get_cmap('plasma')
        
        blues = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#7BC8F6","#030764"])
        reds = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#FFA500","#E50000"])
        
        fig3,ax3 = plt.subplots()
        # ax3.set_xlim([0,30])
        for gen in idx_plots:
            # ax3.set_ylim([0.03535,0.03555])
            if YLIM:
                ax3.set_ylim(YLIM)
            ax3.scatter(struc_hps[raw_tbs_pipes[med_run]['struc']],raw_tbs_pipes[med_run]['cv_gens'][gen],color=reds(colour_grads[gen]),label='TPOT-BO-S',marker='o',facecolors='none',s=70)
            size = 10 if gen> 0 else 40
            ax3.scatter(hp_plots[gen][:,0],hp_plots[gen][:,1],label='TPOT-BO-O',marker='o',s=size,color=blues(colour_grads[gen]))
            
        l_tbo80 = Line2D([0], [0], label="TPOT-BO-O(@80 gens)", color='#7BC8F6', lw=0,marker='o',markersize=8)
        l_tbo = Line2D([0], [0], label="TPOT-BO-O", color='#030764', lw=0,marker='o',markersize=4)
        l_tbs80 = Line2D([0], [0], label="TPOT-BO-S(@80 gens)", color='#FFA500', lw=0,marker='o',markerfacecolor='none',markersize=9)
        l_tbs = Line2D([0], [0], label="TPOT-BO-S", color='#E50000', lw=0,marker='o',markerfacecolor='none',markersize=9)
        # labels = ['TPOT evaluation', 'BO evaluation']
        ax3.legend(handles=[l_tbo80,l_tbo,l_tbs80,l_tbs],prop={'size': 9})
        # ax2.legend()
        if SHOW_TITLE:
            ax3.set_title(f"{PROBLEM} :: Run {med_run} :: generation {gen}")
        ax3.set_ylabel("best CV")
        ax3.set_xlabel("no. HPs")
        # ax3.set_ylim(ylims)
        # ax3.set_xlim(xlims)
        # if SAVE_PLOTS:
        #     f_hvs_hps = os.path.join(plot_path,f'{PROBLEM}_TPOT-BO-HvS_HPs_run_{med_run}_{disc_flag}.png')
        #     fig3.savefig(f_hvs_hps,bbox_inches='tight')
        plt.show()
        
        
        fig4,ax4 = plt.subplots()
        for gen in idx_plots:
            # ax3.set_ylim([0.03535,0.03555])
            if YLIM:
                ax4.set_ylim(YLIM)
            ax4.scatter(struc_idxs[raw_tbs_pipes[med_run]['struc']],raw_tbs_pipes[med_run]['cv_gens'][gen],color=reds(colour_grads[gen]),label='TPOT-BO-S',marker='o',facecolors='none',s=70)
            size = 10 if gen> 0 else 40
            ax4.scatter(idx_plots[gen][:,0],idx_plots[gen][:,1],label='TPOT-BO-H',marker='o',s=size,color=blues(colour_grads[gen]))
            
        l_tbo80 = Line2D([0], [0], label="TPOT-BO-O(@80 gens)", color='#7BC8F6', lw=0,marker='o',markersize=8)
        l_tbo = Line2D([0], [0], label="TPOT-BO-O", color='#030764', lw=0,marker='o',markersize=4)
        l_tbs80 = Line2D([0], [0], label="TPOT-BO-S(@80 gens)", color='#FFA500', lw=0,marker='o',markerfacecolor='none',markersize=9)
        l_tbs = Line2D([0], [0], label="TPOT-BO-S", color='#E50000', lw=0,marker='o',markerfacecolor='none',markersize=9)
        # labels = ['TPOT evaluation', 'BO evaluation']
        ax4.legend(handles=[l_tbo80,l_tbo,l_tbs80,l_tbs],prop={'size': 9})
        # ax2.legend()
        if SHOW_TITLE:
            ax4.set_title(f"{PROBLEM} :: Run {med_run} :: generation {gen}")
        ax4.set_ylabel("best CV")
        ax4.set_xlabel("pipeline index")
        # ax3.set_ylim(ylims)
        # ax3.set_xlim(xlims)
        # if SAVE_PLOTS:
        #     f_hvs = os.path.join(plot_path,f'{PROBLEM}_TPOT-BO-HvS_run_{med_run}_{disc_flag}.png')
        #     fig4.savefig(f_hvs,bbox_inches='tight')
        plt.show()
        
        
#         print(f"\n{u.CYAN}{PROBLEM} ({mode}) statistics:{u.OFF}")
#         print(f"{str(''):>{PRINT_COL}}{str('init TPOT'):>{PRINT_COL}}{str('TPOT-BASE'):>{PRINT_COL}}{str('TPOT-BO-S'):>{PRINT_COL}}{str('TPOT-BO-O'):>{PRINT_COL}}")
            
#         print("="*(6*PRINT_COL + 2))
        
#         for stat,methods in stats.items():
#             print(f"{str(stat):>{PRINT_COL}}",end="")
#             for method,val in methods.items():
#                 print(f"{val:>{PRINT_COL}.6e}",end="")
#             print()
        
#         for method in WTL:
#             med_stats[PROBLEM][method][mode] = stats['median'][method]
        
#         med_stats[PROBLEM]['TPOT-BO-Hs'] = stats['median']['TPOT-BO-Hs']
        
#         wtl = {}
        
#         for method in WTL:
#             wtl[method] = {'w':0,'t':0,'l':0}
#             for i,v in enumerate(data['TPOT-BO-Hs']):
#                 if abs(v - data[method][i]) < WIN_TOL:
#                     wtl[method]['t'] = wtl[method]['t'] + 1
#                 elif v < data[method][i]:
#                     wtl[method]['w'] = wtl[method]['w'] + 1
#                 else:
#                     wtl[method]['l'] = wtl[method]['l'] + 1
            
#             print(f"\nWin/Tie/Loss (TPOT-BO-Hs vs. {method}):")
#             print(f"{wtl[method]['w']}/{wtl[method]['t']}/{wtl[method]['l']}")
#             wtl_stats[PROBLEM][method][mode] = f"{wtl[method]['w']}/{wtl[method]['t']}/{wtl[method]['l']}"

            
# print("\nmed stats:\n")
# print(f"{str(''):>{PRINT_COL}}{str('TPOT-BASE'):>{PRINT_COL}}{str('TPOT-BO-Sd'):>{PRINT_COL}}{str('TPOT-BO-Sc'):>{PRINT_COL}}{str('TPOT-BO-Hd'):>{PRINT_COL}}{str('TPOT-BO-Hc'):>{PRINT_COL}}{str('TPOT-BO-Hs'):>{PRINT_COL}}")
# for prob in PROBLEMS:
#     print(f"{prob:>{PRINT_COL}}{med_stats[prob]['TPOT-BASE']['discrete']:>{PRINT_COL}.6e}{med_stats[prob]['TPOT-BO-S']['discrete']:>{PRINT_COL}.6e}{med_stats[prob]['TPOT-BO-S']['continuous']:>{PRINT_COL}.6e}{med_stats[prob]['TPOT-BO-H']['discrete']:>{PRINT_COL}.6e}{med_stats[prob]['TPOT-BO-H']['continuous']:>{PRINT_COL}.6e}{med_stats[prob]['TPOT-BO-Hs']:>{PRINT_COL}.6e}")
    
    
# print("\nwtl stats:\n")
# print(f"{str(''):>{PRINT_COL}}{str('TPOT-BASE'):>{PRINT_COL}}{str('TPOT-BO-Sd'):>{PRINT_COL}}{str('TPOT-BO-Sc'):>{PRINT_COL}}{str('TPOT-BO-Hd'):>{PRINT_COL}}{str('TPOT-BO-Hc'):>{PRINT_COL}}")
# for prob in PROBLEMS:
#     print(f"{prob:>{PRINT_COL}}{wtl_stats[prob]['TPOT-BASE']['discrete']:>{PRINT_COL}}{wtl_stats[prob]['TPOT-BO-S']['discrete']:>{PRINT_COL}}{wtl_stats[prob]['TPOT-BO-S']['continuous']:>{PRINT_COL}}{wtl_stats[prob]['TPOT-BO-H']['discrete']:>{PRINT_COL}}{wtl_stats[prob]['TPOT-BO-H']['continuous']:>{PRINT_COL}}{wtl_stats[prob]['TPOT-BO-Hs']:>{PRINT_COL}}")