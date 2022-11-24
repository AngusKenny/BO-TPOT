#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 18:14:11 2022

@author: gus

NOTES:
seems to be mostly clustering into 2-3 clusters
each generation is adding 80+ structures, so thousands of structures
computing kmedoids for 2->N is wasting *heaps* of time
force it to have more clusters than 2 to get the first main peak after initial?
cap possible number of clusters? 
proportional to number of structures?



"""

from tpot import TPOTRegressor
import utils.tpot_utils as u
import copy
import os
import time
import numpy as np
import apted as at
from sklearn.metrics import silhouette_score
import kmedoids
import pygmo as pg
from deap import creator

class dTPOT_Base(object):
    pipes = {}
    
    def __init__(self,
                 n_gens=100,
                 pop_size=100,
                 seed=42,
                 config_dict=default_tpot_config_dict,
                 n_jobs=-1,
                 pipe_eval_timeout=5,
                 vprint=u.Vprint(1)):
        
        self.n_gens=n_gens
        self.pop_size=pop_size
        self.vprint=vprint
        self.seed=seed
        self.config_dict = copy.deepcopy(config_dict)
        self.n_jobs = n_jobs
        self.pipe_eval_timeout = pipe_eval_timeout
        
    def optimize(self, X_train, y_train, out_path=None):
        t_start = time.time()

        self.vprint.v2(f"{u.CYAN}fitting tpot model to initialise" 
               + f"..\n{u.WHITE}")
        
        # set tpot verbosity to vprint.verbosity + 1 to give more information
        tpot_verb = self.vprint.verbosity + 1 if self.vprint.verbosity > 0 else 0
        
        # create TPOT object and fit for tot_gens generations
        tpot = TPOTRegressor(generations=1,
                            population_size=self.pop_size, 
                            mutation_rate=0.9, 
                            crossover_rate=0.1, 
                            cv=5,
                            verbosity=tpot_verb, 
                            config_dict = self.config_dict, 
                            random_state=self.seed, 
                            n_jobs=self.n_jobs,
                            warm_start=True,
                            max_eval_time_mins=self.pipe_eval_timeout)
        
        # in case something has been done to pipes externally,
        # before optimize is called, update tpot pipes
        tpot.evaluated_individuals_ = copy.deepcopy(self.pipes)
        
        # fit tpot model to training data
        tpot.fit(X_train, y_train)
        
        
        for v in tpot.evaluated_individuals_.values():
            v['source'] = 'dTPOT-BASE'
        
        # partition by structures and get master structure list
        grps = u.get_unique_groups(tpot.evaluated_individuals_, config_dict=default_tpot_config_dict)
        
        g_keys = list(grps.keys())
        
        # create distance matrix
        d_matrix = np.zeros((len(g_keys),len(g_keys)))
        
        for i in range(d_matrix.shape[0]):
                src = at.helpers.Tree.from_text(g_keys[i])
                for j in range(i+1, d_matrix.shape[0]):
                    tgt = at.helpers.Tree.from_text(g_keys[j])
                    d = at.APTED(src,tgt).compute_edit_distance()
                    d_matrix[i,j] = d
                    d_matrix[j,i] = d
        
        gen = 1
        
        while len(tpot.evaluated_individuals_) < (self.pop_size * self.n_gens):
            gen = gen + 1
            self.vprint.v2(f"\n{u.CYAN}({time.strftime('%d %b, %H:%M', time.localtime())}) Generation {gen} complete, archive size: {len(tpot.evaluated_individuals_)} of {(self.pop_size * self.n_gens)}{u.OFF}\n")
            
            n_new_pipes = 0
            n_new_groups = 0
            # update distance matrix with new groups
            for p,v in tpot.evaluated_individuals_.items():
                if 'group' in v:
                    continue
                n_new_pipes = n_new_pipes + 1
                v['group'] = u.string_to_bracket(p)
                v['source'] = 'dTPOT-BASE'
                v['generation'] = gen
                
                # if group already exists, add to existing group
                if v['group'] in grps:
                    grps[v['group']]['matching'][p] = copy.deepcopy(v)
                    grps[v['group']] = u.update_group(grps[v['group']])
                else:
                    n_new_groups = n_new_groups + 1
                    # create new group
                    grps[v['group']] = u.make_new_group(p,v,default_tpot_config_dict)
                                        
                    # add to g_keys
                    g_keys.append(v['group'])
                    
                    # update distance matrix
                    d_matrix = np.vstack((d_matrix,np.zeros((1,d_matrix.shape[1]))))
                    d_matrix = np.hstack((d_matrix,np.zeros((d_matrix.shape[0],1))))
                    src = at.helpers.Tree.from_text(v['group'])
                    for i in range(0,len(g_keys)-1):
                        tgt = at.helpers.Tree.from_text(g_keys[j])
                        d = at.APTED(src,tgt).compute_edit_distance()
                        d_matrix[i,-1] = d
                        d_matrix[-1,i] = d
            
            self.vprint.v2(f"{n_new_pipes} new pipelines added, with {n_new_groups} new structures")
            
            # compute clusters
            s_scores = []
            labels = np.empty((0,d_matrix.shape[0]))            
            
            for k in range(2,d_matrix.shape[0]):
                km = kmedoids.KMedoids(k, method='fasterpam',random_state=self.seed)
                km.fit_predict(d_matrix)
                labels = np.vstack((labels,np.array(km.labels_).reshape(1,-1)))
                s_score = silhouette_score(d_matrix,labels=km.labels_,metric='precomputed')
                s_scores.append(s_score)
                
            # find k with highest silhouette index
            k = np.argmax(s_scores) + 2
            
            self.vprint.v2(f"{len(grps)} structures partitioned into {k} clusters\nfitting tpot object...\n")
            
            # assign cluster labels
            k_labels = labels[k-2,:]
            # print(f"k_labels: {k_labels}")
            for i,(g,v) in enumerate(grps.items()):
                v['cluster'] = int(k_labels[i])
            
            # do ND sort on evalutated individuals:
            points = np.hstack((np.array([-v['internal_cv_score'] for v in tpot.evaluated_individuals_.values()]).reshape(-1,1), 
                                np.array([grps[v['group']]['n_bo_params'] for v in tpot.evaluated_individuals_.values()]).reshape(-1,1)))
            
            # ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points=points)
            nd_idxs = pg.sort_population_mo(points=points).tolist()
            
            # print(f"nd_idxs: {nd_idxs}\n\n")
            
            # clear tpot population
            tpot._pop = []

            # create vectors to keep track of which pipes and how many from each cluster are chosen
            c_picked = [0] * k
            p_picked = [0] * len(tpot.evaluated_individuals_)
                        
            # set max from cluster at 1
            max_from_cluster = 1
            # while tpot population is less than required size:
            while len(tpot._pop) < self.pop_size:
                p_keys = list(tpot.evaluated_individuals_.keys())
                # iterate over sorted pipelines
                for idx in nd_idxs:
                    # if selected number from cluster of current pipeline is less than max cluster, add to population
                    if c_picked[grps[tpot.evaluated_individuals_[p_keys[idx]]['group']]['cluster']] < max_from_cluster and p_picked[idx] == 0:
                        tpot._pop.append(creator.Individual.from_string(p_keys[idx], tpot._pset))
                        p_picked[idx] = 1
                        
                    # if population size is required size then break
                    if len(tpot._pop) >= self.pop_size: break
                    
                max_from_cluster = max_from_cluster + 1
            
            # do one generation of TPOT 
            # tpot._fit_init()
            tpot.fit(X_train, y_train)
                    
        # copy evaluated individuals dictionary
        self.pipes = copy.deepcopy(tpot.evaluated_individuals_)
        
        for k,v in self.pipes.items():
            v['source'] = 'dTPOT-BASE'
        
        t_end = time.time()
        
        best_tpot_pipe, best_tpot_cv = u.get_best(self.pipes)
        
        self.vprint.v1(f"\n{u.YELLOW}* best pipe found by TPOT:{u.OFF}")
        self.vprint.v1(f"{best_tpot_pipe}")
        self.vprint.v1(f"{u.GREEN} * score:{u.OFF} {best_tpot_cv}")
        self.vprint.v1(f"\nTotal time elapsed: {round(t_end-t_start,2)} sec\n")
        
        
        # if out_path exists then write pipes to file
        if out_path:
            print(out_path)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            fname_tpot_pipes = os.path.join(out_path,'dTPOT-BASE.pipes')
            print(fname_tpot_pipes)
            # write all evaluated pipes
            with open(fname_tpot_pipes, 'w') as f:
                for k,v in self.pipes.items():
                    f.write(f"{k};{v['generation']};{v['internal_cv_score']}\n")
                    
        return "Successful"