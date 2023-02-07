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
from matplotlib.ticker import MaxNLocator
import matplotlib.colors
import matplotlib as mpl
from matplotlib.lines import Line2D

RESULTS_PATH = 'Results'
PROBLEM = 'abalone'
DISCRETE_MODE = False
# DISCRETE_MODE = True
SEED = 44
PRINT_COL = 20
SAVE_PLOTS = True
SAVE_PLOTS = False
WIN_TOL = 1e-14
ANIMATE = False
SHOW_TITLE = False
YLIM = None
FIGURE_SIZE = (4.8,4)
YLIM = [4.18,4.26]
LEGEND_LOC = 'lower right'

plt.rcParams["figure.figsize"] = FIGURE_SIZE
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
plt.rcParams["font.size"] = 18
# plt.rcParams["font.family"] = "serif"
# plt.rcParams["mathtext.fontset"] = "dejavuserif"

mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
mpl.rc('text', usetex=True)

disc_flag = "d" if DISCRETE_MODE else "c"

cwd = os.getcwd()
prob_path = os.path.join(cwd,RESULTS_PATH,PROBLEM)

plot_path = os.path.join(prob_path,'Plots')

tbex_path = os.path.join(prob_path,f'TPOT-BO-EX{disc_flag}',f'Seed_{SEED}')
tbex_imp = os.path.join(tbex_path,'TPOT-BO-EX.improvements')

if not os.path.exists(tbex_imp):
    print(f"Cannot find {tbex_imp}\nquitting..")
    sys.exit()

data = np.empty((0,3))

with open(tbex_imp,'r') as f:
    for i,line in enumerate(f):
        ls = line.split(";")
        struc = ls[0]
        old_cv = -1*float(ls[1])
        new_cv = -1*float(ls[2])
        
        data = np.vstack((data,[i,old_cv,new_cv]))

# test data (uncomment to use)
# data = np.empty((20,3))
# data[:,0] = range(20)
# data[:,1] = [4.24416113, 4.25120968, 4.25228764, 4.25275039, 4.25276742, 4.25276742, 4.25276742, 4.25276742, 4.25276742, 4.25276742, 4.25276742, 4.25276742, 4.25276742, 4.25276742, 4.25629026, 4.25707512, 4.25707512, 4.25707512, 4.25707512, 4.25707512 ]
# data[:,2] = [4.23316113, 4.246120968, 4.22228764, 4.24275039, 4.25276742, 4.19276742, 4.24276742, 4.24226742, 4.15276742, 4.25271742, 4.21276742, 4.25276742, 4.25276742, 4.25276742, 4.23629026, 4.20707512, 4.24507512, 4.235707512, 4.25107512, 4.25707512 ]

fig,ax = plt.subplots()

ax.plot((data[:,0],data[:,0]),([i1 for i1 in data[:,1]], [i2 for i2 in data[:,2]]),c='black',linestyle='dotted',zorder=0.1,label='_nolegend_')

ax.scatter(data[:,0],data[:,1],color='#7BC8F6',marker='o',s=50,label='TPOT-BASE$_{80}$')
ax.scatter(data[:,0],data[:,2],color='#030764',marker='o',s=20,label='After BO')
ax.scatter(data[0,0],data[0,1],color='#FFA500',linewidth=2,marker='o',facecolors='none',s=70,label='TPOT-BO-S$_{80}$')
ax.scatter(data[0,0],data[0,2],color='#E50000',linewidth=2,marker='o',facecolors='none',s=70,label='TPOT-BO-S')

# ax.annotate("PLACEHOLDER PLOT\nreal data will be\nready by tomorrow\n(plot will look the same though)",[0.7,4.195],color="red")

ax.set_ylabel("Best CV for structure")
ax.set_xlabel("Structure index")
ax.legend(loc=LEGEND_LOC, prop={'size': 12})

ax.set_ylim(YLIM)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

fig.tight_layout(h_pad=3.5)
plt.show()

