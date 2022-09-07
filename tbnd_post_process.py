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
import pygmo as pg
import optuna
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.lines import Line2D

RESULTS_PATH = 'Results_Hr'
PROBLEM = 'quake'
DISCRETE_MODE = False
RUNS = range(20)
PRINT_COL = 20
SAVE_PLOTS = False
WIN_TOL = 1e-6
ANIMATE = False
SHOW_TITLE = False

disc_txt = "discrete" if DISCRETE_MODE else "continuous"

cwd = os.getcwd()
prob_path = os.path.join(cwd,RESULTS_PATH,PROBLEM)

plot_path = os.path.join(prob_path,'Plots')

raw_data = {}
raw_tbnd_pipes = {}
raw_tbs_pipes = {}

skip_runs = []

for run in RUNS:
    run_txt = f"Run_{run}" if run > 9 else f"Run_0{run}"
    
    run_path = os.path.join(prob_path, run_txt)
    
    tb_path = os.path.join(run_path,'TPOT-BASE')
    f_tb_prog = os.path.join(tb_path,'TPOT-BASE.progress')
    f_tb_pipes = os.path.join(tb_path,'TPOT-BASE.pipes')
    
    tbs_path = os.path.join(run_path,'TPOT-BO-S')
    f_tbs_prog = os.path.join(tbs_path,'bo_progress.out')
    f_tbs_pipes = os.path.join(tbs_path,'bo_pipes.out')
    
    tbnd_path = os.path.join(run_path,'TPOT-BO-ND',disc_txt)
    f_tbnd_prog = os.path.join(tbnd_path,'TPOT-BO-ND.progress')
    f_tbnd_pipes = os.path.join(tbnd_path,'TPOT-BO-ND.pipes')
    
    check_files = [f_tb_prog, f_tb_pipes, f_tbs_prog, f_tbs_pipes, f_tbnd_prog, f_tbnd_pipes]
    
    for check_file in check_files:
        if not os.path.exists(check_file):
            print(f"Cannot find {check_file}\nskipping {run_txt}..")
            skip_runs.append(run)
            break
    
    if run in skip_runs: continue

    raw_data[run] = {}
    raw_tbnd_pipes[run] = {}
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
    
    with open(f_tbnd_prog, 'r') as f:
        for line in f:
            if 'INITIAL TPOT GENERATIONS' in line:
                cv_line = next(f)
                if -float(cv_line.split(":")[-1]) != raw_data[run]['init TPOT']:
                    print(f"{run_txt} different initial TPOT values")
            if 'AFTER' in line and 'BAYESIAN' in line:
                next(f)
                cv_line = next(f)
                raw_data[run]['TPOT-BO-ND'] = -float(cv_line.split(":")[-1])
    
    with open(f_tbnd_pipes, 'r') as f:
        for line in f:
            ls = line.split(";")
            pipe = ls[0]
            gen = int(ls[1])
            n_nd = int(ls[2])
            struc = ls[3].split("(")[-1].strip(")")
            cv = -float(ls[4])
            
            if gen not in raw_tbnd_pipes[run]:
                raw_tbnd_pipes[run][gen] = {} if gen == 0 else copy.deepcopy(raw_tbnd_pipes[run][gen-1])
            
            if struc not in raw_tbnd_pipes[run][gen]:
                # new_cv = 1e20 if gen == 0 else raw_tbnd_pipes[run][gen-1][struc]['cv']
                raw_tbnd_pipes[run][gen][struc] = {'cv': 1e20, 'n_bo_params':len(u.string_to_params(pipe,default_tpot_config_dict))}
            
            raw_tbnd_pipes[run][gen][struc]['cv'] = min(raw_tbnd_pipes[run][gen][struc]['cv'],cv)
    
    with open(f_tbs_pipes) as f:
        for line in f:
            ls = line.split(";")
            pipe = ls[0]
            cv = -float(ls[1])
            
            if not raw_tbs_pipes[run]['n_bo_hp']:
                raw_tbs_pipes[run]['n_bo_hp'] = len(u.string_to_params(pipe,default_tpot_config_dict))
                raw_tbs_pipes[run]['cv_best'].append(cv)
            else:
                raw_tbs_pipes[run]['cv_best'].append(min(raw_tbs_pipes[run]['cv_best'][-1],cv))
                
last_gens = {run: max(raw_tbnd_pipes[run].keys()) for run in raw_tbnd_pipes}

for run, v in raw_tbs_pipes.items():
    # interpolate between 0 and bo_trials
    v['cv_gens'] = np.interp(np.linspace(0, len(v['cv_best']), last_gens[run]+1), range(len(v['cv_best'])), v['cv_best'])      

nd_plots = {run:np.array([[raw_tbnd_pipes[run][last_gens[run]][struc]['n_bo_params'],raw_tbnd_pipes[run][last_gens[run]][struc]['cv']] for struc in raw_tbnd_pipes[run][last_gens[run]]]) for run in raw_tbnd_pipes}                

data={}

data['init TPOT'] = [v['init TPOT'] for k,v in raw_data.items()]
data['TPOT-BASE'] = [v['TPOT-BASE'] for k,v in raw_data.items()]
data['TPOT-BO-S'] = [v['TPOT-BO-S'] for k,v in raw_data.items()]
data['TPOT-BO-ND'] = [v['TPOT-BO-ND'] for k,v in raw_data.items()]


stats = {'best':{},'worst':{},'median':{},'mean':{},'std dev':{}}

stats['best'] = {k: np.min(v) for k,v in data.items()}
stats['worst'] = {k: np.max(v) for k,v in data.items()}
stats['median'] = {k: np.median(v) for k,v in data.items()}
stats['mean'] = {k: np.mean(v) for k,v in data.items()}
stats['std dev'] = {k: np.std(v) for k,v in data.items()}


fig1,ax1 = plt.subplots()
ax1.set_ylabel("best CV")
ax1.set_xlabel("no. HPs")
if SHOW_TITLE:
    ax1.set_title(f"{PROBLEM} - nd plots")
for i,run in enumerate(nd_plots):
    marker = '.' if i < 10 else '+' if i < 20 else '2'
    if nd_plots[run].shape[0] > 1:
        ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points = nd_plots[run])
        idxs = ndf[0]
    else:
        idxs = [0]
    ax1.scatter(nd_plots[run][idxs,0],nd_plots[run][idxs,1],marker=marker)
if SAVE_PLOTS:
    f_nd_plots = os.path.join(plot_path,f'{PROBLEM}_TPOT-BO-ND_nd_plots.png')
    fig1.savefig(f_nd_plots,bbox_inches='tight')

# find median run index (closest to median value)
med_auto_run_idx = np.abs(data['TPOT-BO-ND'] - np.median(data['TPOT-BO-ND'])).argmin()
med_run = list(raw_data.keys())[med_auto_run_idx]
   
med_plots = {gen:np.array([[raw_tbnd_pipes[med_run][gen][struc]['n_bo_params'],raw_tbnd_pipes[med_run][gen][struc]['cv']] for struc in raw_tbnd_pipes[med_run][gen]]) for gen in raw_tbnd_pipes[med_run]}                

ymax = np.max(med_plots[0][:,1])
ymin = np.min(med_plots[list(med_plots.keys())[-1]][:,1])

xmax = np.max(med_plots[0][:,0])

ylims = [ymin-(ymax-ymin)*0.1,ymax + (ymax-ymin)*0.1]
xlims = [0,xmax + 1]

frames = med_plots.keys() if ANIMATE else [0,last_gens[med_run]]

for gen in frames:
    fig2,ax2 = plt.subplots()
    ax2.set_xlim([0,30])
    # ax2.set_ylim([0.03528,0.0359])
    if med_plots[gen].shape[0] > 1:
        ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points = med_plots[gen])
        idxs = ndf[0]
    else:
        idxs = [0]
    # ax2.scatter(raw_tbs_pipes[med_run]['n_bo_hp'],raw_tbs_pipes[med_run]['cv_gens'][0],color='C1',s=50,label='TPOT-BO-S(@80)')
    ax2.scatter(med_plots[0][:,0],med_plots[0][:,1],color='C1',label='TPOT-BO-ND(@80)')
    ax2.scatter(raw_tbs_pipes[med_run]['n_bo_hp'],raw_tbs_pipes[med_run]['cv_gens'][gen],color='C3',s=70,label='TPOT-BO-S')
    ax2.scatter(med_plots[gen][idxs,0],med_plots[gen][idxs,1],color='C0',label='TPOT-BO-ND')
    ax2.legend()
    if SHOW_TITLE:
        ax2.set_title(f"{PROBLEM} :: Run {med_run} :: generation {gen}")
    ax2.set_ylabel("best CV")
    ax2.set_xlabel("no. HPs")
    ax2.set_ylim(ylims)
    ax2.set_xlim(xlims)
    if SAVE_PLOTS and ANIMATE:
        gen_txt = f"0{gen}" if gen > 9 else f"{gen}"
        anim_path = os.path.join(plot_path,'anim')
        if not os.path.exists(anim_path):
            os.makedirs(anim_path)
        f_anim = os.path.join(anim_path,f'{PROBLEM}_TPOT-BO-ND_run_{med_run}_gen_{gen_txt}.png')
        fig2.savefig(f_anim,bbox_inches='tight')
    plt.show()


alphas = np.linspace(0.4,1,last_gens[med_run]+1)
colour_grads = np.linspace(0,1,last_gens[med_run]+1)
# blues = plt.get_cmap('plasma')

blues = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#7BC8F6","#030764"])
reds = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#FFA500","#E50000"])

fig3,ax3 = plt.subplots()
ax3.set_xlim([0,30])
for gen in med_plots:
    # ax2.set_ylim([0.03528,0.0359])
    if med_plots[gen].shape[0] > 1:
        ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points = med_plots[gen])
        idxs = ndf[0]
    else:
        idxs = [0]
    # ax2.scatter(raw_tbs_pipes[med_run]['n_bo_hp'],raw_tbs_pipes[med_run]['cv_gens'][0],color='C1',s=50,label='TPOT-BO-S(@80)')
    # ax2.scatter(med_plots[0][:,0],med_plots[0][:,1],color='C1',label='TPOT-BO-ND(@80)')
    ax3.scatter(raw_tbs_pipes[med_run]['n_bo_hp'],raw_tbs_pipes[med_run]['cv_gens'][gen],color=reds(colour_grads[gen]),label='TPOT-BO-S',marker='o',facecolors='none',s=70)
    size = 20 if gen> 0 else 60
    ax3.scatter(med_plots[gen][idxs,0],med_plots[gen][idxs,1],label='TPOT-BO-ND',marker='o',s=size,color=blues(colour_grads[gen]))
    
l_tbnd80 = Line2D([0], [0], label="TPOT-BO-ND(@80 gens)", color='#7BC8F6', lw=0,marker='o',markersize=8)
l_tbnd = Line2D([0], [0], label="TPOT-BO-ND", color='#030764', lw=0,marker='o',markersize=5)
l_tbs80 = Line2D([0], [0], label="TPOT-BO-S(@80 gens)", color='#FFA500', lw=0,marker='o',markerfacecolor='none',markersize=9)
l_tbs = Line2D([0], [0], label="TPOT-BO-S", color='#E50000', lw=0,marker='o',markerfacecolor='none',markersize=9)
# labels = ['TPOT evaluation', 'BO evaluation']
ax3.legend(handles=[l_tbnd80,l_tbnd,l_tbs80,l_tbs])
# ax2.legend()
if SHOW_TITLE:
    ax3.set_title(f"{PROBLEM} :: Run {med_run} :: generation {gen}")
ax3.set_ylabel("best CV (median run)")
ax3.set_xlabel("no. HPs")
ax3.set_ylim(ylims)
ax3.set_xlim(xlims)
if SAVE_PLOTS:
    f_dnvs = os.path.join(plot_path,f'{PROBLEM}_TPOT-BO-NDvS_run_{med_run}.png')
    fig3.savefig(f_dnvs,bbox_inches='tight')
plt.show()



print(f"\n{u.CYAN}{PROBLEM} statistics:{u.OFF}")
print(f"{str(''):>{PRINT_COL}}{str('init TPOT'):>{PRINT_COL}}{str('TPOT-BASE'):>{PRINT_COL}}{str('TPOT-BO-S'):>{PRINT_COL}}{str('TPOT-BO-ND'):>{PRINT_COL}}")
    
print("="*(5*PRINT_COL + 2))

for stat,methods in stats.items():
    print(f"{str(stat):>{PRINT_COL}}",end="")
    for method,val in methods.items():
        print(f"{val:>{PRINT_COL}.6e}",end="")
    print()

wtl = {'w':0,'t':0,'l':0}

for i,v in enumerate(data['TPOT-BO-ND']):
    if abs(v - data['TPOT-BO-S'][i]) < WIN_TOL:
        wtl['t'] = wtl['t'] + 1
    elif v < data['TPOT-BO-S'][i]:
        wtl['w'] = wtl['w'] + 1
    else:
        wtl['l'] = wtl['l'] + 1

print("\nWin/Tie/Loss (TPOT-BO-ND vs. TPOT-BO-S):")
print(f"{wtl['w']}/{wtl['t']}/{wtl['l']}")