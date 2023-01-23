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
import matplotlib as mpl
from matplotlib import cm 
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
import matplotlib.colors
from matplotlib.lines import Line2D
import cmasher as cmr

RESULTS_PATH = 'Results'
PROBLEMS = [            
    'quake',
    # 'socmob',
    # 'abalone',
    # 'brazilian_houses',
    # 'house_16h',
    # 'elevators'
    ]
# MODES = ['discrete']#,'continuous']
# WTL = ['TPOT-BASE','TPOT-BO-S']#,'TPOT-BO-H']
METHOD = 'oTPOT-BASE'
POP_SIZE = 100
SEEDS = [43]
PRINT_COL = 20
SAVE_PLOTS = False
SAVE_PLOTS = False
# WIN_TOL = 1e-14
# ANIMATE = False
SHOW_TITLE = True
# YLIM = None
# YLIM = [3.54e-6,3.685e-6]
# LEGEND_POS = 'lower right'

for problem in PROBLEMS:
    
    cwd = os.getcwd()
    prob_path = os.path.join(cwd,RESULTS_PATH,problem)
    
    plot_path = os.path.join(prob_path,'Plots')
    
    skip_seeds = []
    
    sparse_tracker = {}
    
    for seed in SEEDS:
        seed_txt = f"Seed_{seed}"
        
        seed_path = os.path.join(prob_path, f'{METHOD}', seed_txt)
    
        # otb_path = os.path.join(run_path,'{METHOD}')
        # f_otb_prog = os.path.join(tb_path,'{METHOD}.progress')
        # f_otb_pipes = os.path.join(tb_path,'{METHOD}.pipes')
        f_otb_tracker = os.path.join(seed_path,f'{METHOD}.tracker')
        f_otb_times = os.path.join(seed_path,f'{METHOD}.times')

        check_files = [f_otb_tracker]
        
        for check_file in check_files:
            if not os.path.exists(check_file):
                print(f"Cannot find {check_file}\nskipping seed {seed_txt}..")
                skip_seeds.append(seed)
                break
        
        if seed in skip_seeds: continue
        
        sparse_tracker[seed] = {}
        
        full_tracker = {}
                            
        op_tracker = {}
                 
        max_op_tracker = []                               
                                       
        with open(f_otb_tracker, 'r') as f:
            for line in f:
                ls = line.split(";")
                gen = int(ls[0])
                struc = ls[1]
                n_selected = int(ls[2])
                # n_ops = int(ls[3])
                strip_struc = struc.replace("{input_matrix}","")
                n_ops = len(strip_struc.split("{"))
                
                if gen not in sparse_tracker[seed]:
                    sparse_tracker[seed][gen] = {}
                    max_op_tracker.append(0)
                
                sparse_tracker[seed][gen][struc] = n_selected
                max_op_tracker[gen] = max(max_op_tracker[gen],n_ops)
                op_tracker[struc] = n_ops
                                
                full_tracker[struc] = []
        
        time_tracker = []
        
        if os.path.exists(f_otb_times):
            with open(f_otb_times) as f:
                for line in f:
                    ls = line.split(";")
                    time_val = float(ls[1])
                    time_tracker.append(time_val)
                    
        # populate full tracker
        for struc,tracking in full_tracker.items():
            for gen, gen_strucs in sparse_tracker[seed].items():
                if struc in gen_strucs:
                    tracking.append(gen_strucs[struc])
                else:
                    tracking.append(0)
       
        # print(full_tracker)
        
        points = np.empty((0,3))
        
        op_points = np.empty((0,2))
        
        for i, (s,v) in enumerate(full_tracker.items()):
            op_points = np.vstack((op_points,[i, op_tracker[s]]))
            for g,n in enumerate(v):
                if n > 0:
                    points = np.vstack((points, [i, g, n]))
        
        # # modified hsv in 256 color class
        # cm_modified = cm.get_cmap('viridis_r', 256)
        # # create new hsv colormaps in range of 0.3 (green) to 0.7 (blue)
        # newcmp = ListedColormap(cm_modified(np.linspace(0, 1, POP_SIZE)))
        
        norm = mpl.colors.Normalize(vmin=1, vmax=POP_SIZE)
        
        # cmap=mpl.colormaps['viridis_r']
        # cmap=mpl.colormaps['cool']
        
        cmap = plt.get_cmap('cmr.cosmic_r')
        # cmap = plt.get_cmap('cmr.dusk_r')
        
        c_tick_grads = POP_SIZE//10
        c_ticks = [1] + [i * c_tick_grads for i in range(1,10)] + [POP_SIZE]
        
        max_gen = max(points[:,1])
        marksize = 400//max_gen
        
        plt.rcParams["figure.figsize"] = (15,9)
        
        # plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
        
        fig, axs = plt.subplots(2)
        fig.suptitle(f"{METHOD} - {problem} - seed: {seed}")
        

        
        gen_plot=axs[0].scatter(points[:,0], points[:,1], c=points[:,2], cmap=cmap,norm=norm,marker='.',s=marksize)
        # plt.scatter(points[:,0], points[:,1], c=points[:,2], cmap=newcmp)
        axs[0].set_ylabel("Generation")
        axs[0].set_xlabel("Structure index")
        fig.colorbar(gen_plot,ax=axs[0],label="No. selected in generation (max = pop size)",ticks=c_ticks)
        
        op_ax = axs[0].twinx()
        # op_plot = op_ax.scatter(op_points[:,0],op_points[:,1],c="tomato",marker='.',s=2)
        op_plot = op_ax.bar(op_points[:,0],op_points[:,1],color="linen",width=1)
        op_ax.set_ylabel("Number of operators")
        
        axs[0].legend([gen_plot,op_plot],['Selected','No. Operators'],loc='lower right')
        
        axs[0].set_zorder(2.5)
        axs[0].set_frame_on(False)
        
        # axs[0].set_title("TPOT-BASE")
        
        # axs[0].colorbar(label="No. selected in generation (max = pop size)",ticks=c_ticks)
        
        
        max_op_plot, = axs[1].plot(max_op_tracker)
        axs[1].set_xlabel("Generation")
        axs[1].set_ylabel("Max number of operators")
        if os.path.exists(f_otb_times):
            time_ax = axs[1].twinx()
            time_plot, = time_ax.plot(time_tracker,c='C1')
            time_ax.set_ylabel("Time taken by TPOT [s]")
            axs[1].legend([max_op_plot,time_plot],['No. Operators','TPOT Time'],loc='lower right')
        else:
            axs[1].legend([max_op_plot],['No. Operators'],loc='lower right')
        fig.tight_layout(h_pad=3.5)
        plt.show()
        
        # axs[1].scatter(points[:,0], points[:,1], c=points[:,2], cmap=mpl.colormaps['viridis_r'],norm=norm,marker='.',s=2)
        # # plt.scatter(points[:,0], points[:,1], c=points[:,2], cmap=newcmp)
        # axs[1].set_ylabel("Generation")
        # axs[1].set_xlabel("Structure index")
        # axs[1].set_title("{METHOD}")
        

        
        # plt.scatter(points[:,0], points[:,1], c=points[:,2], cmap=mpl.colormaps['viridis_r'],norm=norm,marker='.',s=marksize)
        # # plt.scatter(points[:,0], points[:,1], c=points[:,2], cmap=newcmp)
        # plt.ylabel("Generation")
        # plt.xlabel("Structure index")
        # plt.title(f"{METHOD} - {problem} - seed: {seed}")
        # plt.colorbar(label="No. selected in generation (max = pop size)",ticks=c_ticks)
        
        # plt.show()
        
        # plt.plot(max_op_tracker)
        # plt.show()
        
#         last_gens = {run: max(raw_tbo_pipes[run].keys()) for run in raw_tbo_pipes}
        
#         for run, v in raw_tbs_pipes.items():
#             v['cv_gens'] = np.interp(np.linspace(0, len(v['cv_best']), last_gens[run]+1), range(len(v['cv_best'])), v['cv_best'])      
        
#         # nd_plots = {run:np.array([[raw_tbh_pipes[run][last_gens[run]][struc]['n_bo_params'],raw_tbh_pipes[run][last_gens[run]][struc]['cv']] for struc in raw_tbh_pipes[run][last_gens[run]]]) for run in raw_tbh_pipes}                
        
#         data={}
        
#         data['init TPOT'] = [v['init TPOT'] for k,v in raw_data.items()]
#         data['TPOT-BASE'] = [v['TPOT-BASE'] for k,v in raw_data.items()]
#         data['TPOT-BO-S'] = [v['TPOT-BO-S'] for k,v in raw_data.items()]
#         data['TPOT-BO-O'] = [v['TPOT-BO-O'] for k,v in raw_data.items()]
        
#         stats = {'best':{},'worst':{},'median':{},'mean':{},'std dev':{}}
        
#         stats['best'] = {k: np.min(v) for k,v in data.items()}
#         stats['worst'] = {k: np.max(v) for k,v in data.items()}
#         stats['median'] = {k: np.median(v) for k,v in data.items()}
#         stats['mean'] = {k: np.mean(v) for k,v in data.items()}
#         stats['std dev'] = {k: np.std(v) for k,v in data.items()}
        
        
#         # find median run index (closest to median value)
#         med_auto_run_idx = np.abs(data['TPOT-BO-O'] - np.median(data['TPOT-BO-O'])).argmin()
#         med_run = list(raw_data.keys())[med_auto_run_idx]
        
#         # print(len(raw_tbh_pipes[med_run][0].keys()))
        
#         struc_idxs = {struc:i for i,struc in enumerate(raw_tbo_pipes[med_run][0].keys()) if i < 50}
        
#         # for struc in struc_idxs:
#         #     print(struc)
        
#         struc_hps = {struc:v['n_hp'] for struc,v in raw_tbo_pipes[med_run][0].items() if struc in struc_idxs}
               
#         if raw_tbs_pipes[med_run]['struc'] not in struc_idxs:
#             struc_idxs[raw_tbs_pipes[med_run]['struc']] = len(struc_idxs)
        
#         idx_plots = {gen:np.array([[struc_idxs[struc],raw_tbo_pipes[med_run][gen][struc]['cv']] for struc in raw_tbo_pipes[med_run][gen] if struc in struc_idxs]) for gen in raw_tbo_pipes[med_run]}
        
#         # print([[struc_hps[struc],raw_tbh_pipes[med_run][gen][struc]['cv']] for struc in raw_tbh_pipes[med_run][gen] if struc in struc_idxs])
        
#         hp_plots = {gen:np.array([[struc_hps[struc],raw_tbo_pipes[med_run][gen][struc]['cv']] for struc in raw_tbo_pipes[med_run][gen] if struc in struc_idxs]) for gen in raw_tbo_pipes[med_run]}
        
#         # idx_plots = {gen:np.vstack(([np.hstack((struc_idxs[struc]*np.ones((len(raw_tbh_pipes[med_run][gen][struc]),1)),np.array(raw_tbh_pipes[med_run][gen][struc]).reshape(-1,1))) for struc in raw_tbh_pipes[med_run][gen]])) for gen in raw_tbh_pipes[med_run]}                
        
#         if len(idx_plots[list(idx_plots.keys())[-1]]) == 0:
#             idx_plots.pop(list(idx_plots.keys())[-1])
        
#         # print(f"\n{idx_plots}\n")
        
#         ymax = np.max(idx_plots[0][:,1])
#         ymin = np.min(idx_plots[list(idx_plots.keys())[-1]][:,1])
        
#         xmax = np.max(idx_plots[0][:,0])
        
#         ylims = [ymin-(ymax-ymin)*0.1,ymax + (ymax-ymin)*0.1]
#         xlims = [0,xmax + 1]
        
#         colour_grads = np.linspace(0,1,last_gens[med_run]+1)
#         # blues = plt.get_cmap('plasma')
        
#         blues = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#7BC8F6","#030764"])
#         reds = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#FFA500","#E50000"])
        
#         fig3,ax3 = plt.subplots()
#         # ax3.set_xlim([0,30])
#         for gen in idx_plots:
#             # ax3.set_ylim([0.03535,0.03555])
#             if YLIM:
#                 ax3.set_ylim(YLIM)
#             ax3.scatter(struc_hps[raw_tbs_pipes[med_run]['struc']],raw_tbs_pipes[med_run]['cv_gens'][gen],color=reds(colour_grads[gen]),label='TPOT-BO-S',marker='o',facecolors='none',s=70)
#             size = 10 if gen> 0 else 40
#             ax3.scatter(hp_plots[gen][:,0],hp_plots[gen][:,1],label='TPOT-BO-O',marker='o',s=size,color=blues(colour_grads[gen]))
            
#         l_tbo80 = Line2D([0], [0], label="TPOT-BO-O(@80 gens)", color='#7BC8F6', lw=0,marker='o',markersize=8)
#         l_tbo = Line2D([0], [0], label="TPOT-BO-O", color='#030764', lw=0,marker='o',markersize=4)
#         l_tbs80 = Line2D([0], [0], label="TPOT-BO-S(@80 gens)", color='#FFA500', lw=0,marker='o',markerfacecolor='none',markersize=9)
#         l_tbs = Line2D([0], [0], label="TPOT-BO-S", color='#E50000', lw=0,marker='o',markerfacecolor='none',markersize=9)
#         # labels = ['TPOT evaluation', 'BO evaluation']
#         ax3.legend(handles=[l_tbo80,l_tbo,l_tbs80,l_tbs],prop={'size': 9})
#         # ax2.legend()
#         if SHOW_TITLE:
#             ax3.set_title(f"{problem} :: Run {med_run} :: generation {gen}")
#         ax3.set_ylabel("best CV")
#         ax3.set_xlabel("no. HPs")
#         # ax3.set_ylim(ylims)
#         # ax3.set_xlim(xlims)
#         # if SAVE_PLOTS:
#         #     f_hvs_hps = os.path.join(plot_path,f'{PROBLEM}_TPOT-BO-HvS_HPs_run_{med_run}_{disc_flag}.png')
#         #     fig3.savefig(f_hvs_hps,bbox_inches='tight')
#         plt.show()
        
        
#         fig4,ax4 = plt.subplots()
#         for gen in idx_plots:
#             # ax3.set_ylim([0.03535,0.03555])
#             if YLIM:
#                 ax4.set_ylim(YLIM)
#             ax4.scatter(struc_idxs[raw_tbs_pipes[med_run]['struc']],raw_tbs_pipes[med_run]['cv_gens'][gen],color=reds(colour_grads[gen]),label='TPOT-BO-S',marker='o',facecolors='none',s=70)
#             size = 10 if gen> 0 else 40
#             ax4.scatter(idx_plots[gen][:,0],idx_plots[gen][:,1],label='TPOT-BO-H',marker='o',s=size,color=blues(colour_grads[gen]))
            
#         l_tbo80 = Line2D([0], [0], label="TPOT-BO-O(@80 gens)", color='#7BC8F6', lw=0,marker='o',markersize=8)
#         l_tbo = Line2D([0], [0], label="TPOT-BO-O", color='#030764', lw=0,marker='o',markersize=4)
#         l_tbs80 = Line2D([0], [0], label="TPOT-BO-S(@80 gens)", color='#FFA500', lw=0,marker='o',markerfacecolor='none',markersize=9)
#         l_tbs = Line2D([0], [0], label="TPOT-BO-S", color='#E50000', lw=0,marker='o',markerfacecolor='none',markersize=9)
#         # labels = ['TPOT evaluation', 'BO evaluation']
#         ax4.legend(handles=[l_tbo80,l_tbo,l_tbs80,l_tbs],prop={'size': 9})
#         # ax2.legend()
#         if SHOW_TITLE:
#             ax4.set_title(f"{PROBLEM} :: Run {med_run} :: generation {gen}")
#         ax4.set_ylabel("best CV")
#         ax4.set_xlabel("pipeline index")
#         # ax3.set_ylim(ylims)
#         # ax3.set_xlim(xlims)
#         # if SAVE_PLOTS:
#         #     f_hvs = os.path.join(plot_path,f'{PROBLEM}_TPOT-BO-HvS_run_{med_run}_{disc_flag}.png')
#         #     fig4.savefig(f_hvs,bbox_inches='tight')
#         plt.show()
        
        
# #         print(f"\n{u.CYAN}{PROBLEM} ({mode}) statistics:{u.OFF}")
# #         print(f"{str(''):>{PRINT_COL}}{str('init TPOT'):>{PRINT_COL}}{str('TPOT-BASE'):>{PRINT_COL}}{str('TPOT-BO-S'):>{PRINT_COL}}{str('TPOT-BO-O'):>{PRINT_COL}}")
            
# #         print("="*(6*PRINT_COL + 2))
        
# #         for stat,methods in stats.items():
# #             print(f"{str(stat):>{PRINT_COL}}",end="")
# #             for method,val in methods.items():
# #                 print(f"{val:>{PRINT_COL}.6e}",end="")
# #             print()
        
# #         for method in WTL:
# #             med_stats[PROBLEM][method][mode] = stats['median'][method]
        
# #         med_stats[PROBLEM]['TPOT-BO-Hs'] = stats['median']['TPOT-BO-Hs']
        
# #         wtl = {}
        
# #         for method in WTL:
# #             wtl[method] = {'w':0,'t':0,'l':0}
# #             for i,v in enumerate(data['TPOT-BO-Hs']):
# #                 if abs(v - data[method][i]) < WIN_TOL:
# #                     wtl[method]['t'] = wtl[method]['t'] + 1
# #                 elif v < data[method][i]:
# #                     wtl[method]['w'] = wtl[method]['w'] + 1
# #                 else:
# #                     wtl[method]['l'] = wtl[method]['l'] + 1
            
# #             print(f"\nWin/Tie/Loss (TPOT-BO-Hs vs. {method}):")
# #             print(f"{wtl[method]['w']}/{wtl[method]['t']}/{wtl[method]['l']}")
# #             wtl_stats[PROBLEM][method][mode] = f"{wtl[method]['w']}/{wtl[method]['t']}/{wtl[method]['l']}"

            
# # print("\nmed stats:\n")
# # print(f"{str(''):>{PRINT_COL}}{str('TPOT-BASE'):>{PRINT_COL}}{str('TPOT-BO-Sd'):>{PRINT_COL}}{str('TPOT-BO-Sc'):>{PRINT_COL}}{str('TPOT-BO-Hd'):>{PRINT_COL}}{str('TPOT-BO-Hc'):>{PRINT_COL}}{str('TPOT-BO-Hs'):>{PRINT_COL}}")
# # for prob in PROBLEMS:
# #     print(f"{prob:>{PRINT_COL}}{med_stats[prob]['TPOT-BASE']['discrete']:>{PRINT_COL}.6e}{med_stats[prob]['TPOT-BO-S']['discrete']:>{PRINT_COL}.6e}{med_stats[prob]['TPOT-BO-S']['continuous']:>{PRINT_COL}.6e}{med_stats[prob]['TPOT-BO-H']['discrete']:>{PRINT_COL}.6e}{med_stats[prob]['TPOT-BO-H']['continuous']:>{PRINT_COL}.6e}{med_stats[prob]['TPOT-BO-Hs']:>{PRINT_COL}.6e}")
    
    
# # print("\nwtl stats:\n")
# # print(f"{str(''):>{PRINT_COL}}{str('TPOT-BASE'):>{PRINT_COL}}{str('TPOT-BO-Sd'):>{PRINT_COL}}{str('TPOT-BO-Sc'):>{PRINT_COL}}{str('TPOT-BO-Hd'):>{PRINT_COL}}{str('TPOT-BO-Hc'):>{PRINT_COL}}")
# # for prob in PROBLEMS:
# #     print(f"{prob:>{PRINT_COL}}{wtl_stats[prob]['TPOT-BASE']['discrete']:>{PRINT_COL}}{wtl_stats[prob]['TPOT-BO-S']['discrete']:>{PRINT_COL}}{wtl_stats[prob]['TPOT-BO-S']['continuous']:>{PRINT_COL}}{wtl_stats[prob]['TPOT-BO-H']['discrete']:>{PRINT_COL}}{wtl_stats[prob]['TPOT-BO-H']['continuous']:>{PRINT_COL}}{wtl_stats[prob]['TPOT-BO-Hs']:>{PRINT_COL}}")