#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import shutil

RESULTS_DIR = "Results"

cwd = os.getcwd()

results_path = os.path.join(cwd,RESULTS_DIR)

probs = [f for f in os.scandir(results_path) if f.is_dir()]

for prob in probs:
    prob_path = os.path.join(results_path,prob)
    runs = [f for f in os.scandir(prob_path) if f.is_dir() and "Run" in f.path]

    for run in runs:
        run_path = os.path.join(prob_path,run)

        tb_path = os.path.join(run_path,'TPOT-BASE')
        f_tb_prog = os.path.join(tb_path,'TPOT-BASE.progress')
        
        seed = None

        if os.path.exists(f_tb_prog):
            with open(f_tb_prog, 'r') as f:
                for line in f:
                    if 'SEED' in line:
                        seed = int(line.split(":")[-1])

        methods = [f for f in os.scandir(run_path) if f.is_dir()]

        for method in methods:
            tgt_method_path = os.path.join(prob_path,method.name)
            src_method_path = os.path.join(run_path,method)

            if not os.path.exists(tgt_method_path):
                os.makedirs(tgt_method_path)

            seed_path = os.path.join(tgt_method_path,f"Seed_{seed}")
            if not os.path.exists(seed_path):
                os.makedirs(seed_path)

            dir_contents = os.listdir(src_method_path)            
            for obj in dir_contents:
                shutil.move(os.path.join(src_method_path, obj), seed_path)

            # shutil.move(src_method_path,seed_path)

        shutil.rmtree(run_path)
            


