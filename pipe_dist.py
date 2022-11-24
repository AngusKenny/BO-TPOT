import os
import sys
import utils.tpot_utils as u
import numpy as np
import apted as at
import time
PRINT_COL = 10
'''
***** Parameter constants *****
PROBLEM:        String with problem name defined by its filename, and also
                where the run data is stored. e.g., /Data/abalone.data 
                would be 'abalone'    
RUN_LIST:           List of runs to plot. Set to [] to plot using all data.
SAVE_PLOTS:     Save generated plots to file in ./<RESULTS_DIR>/Plots/
'''

params = {
    'RESULTS_DIR'       : 'Results',
    'PROBLEMS'          : [
        'quake',
        'socmob',
        'abalone',
        'brazilian_houses',
        'house_16h',
        'elevators'
        ],
    'METHODS'           : [
        'TPOT-BASE',
        # 'TPOT-BO-Sd','TPOT-BO-Sc',
        'TPOT-BO-ALTd','TPOT-BO-ALTc',
        # 'TPOT-BO-AUTOd','TPOT-BO-AUTOc',
        # 'TPOT-BO-Hd','TPOT-BO-Hc',
        # 'TPOT-BO-Hs'
        ],
    'RUN_LIST'          : [0,1,2,3,4,5,6,7,8,9,10],
    'SAVE_STATS'        : False,
    'CONFIDENCE_LEVEL'  : 0.1,
    'STOP_GEN'          : 80,
    }

t_start = time.time()

cwd = os.getcwd()


results_path = os.path.join(cwd,params['RESULTS_DIR'])
if not os.path.exists(results_path):
    sys.exit(f"Cannot find results directory {results_path}")

f_dist = os.path.join(results_path,'pipe_dist.dat')
f = open(f_dist,'w')
f.close()

params['RUN_LIST'] = list(params['RUN_LIST'])

prob_list = params['PROBLEMS']
# if problem list is empty, search problem directory for problems
if len(prob_list) == 0:
    prob_list = [os.path.basename(d) 
                 for d in os.scandir(results_path) if d.is_dir()]
 
data = {}

for problem in prob_list:
    data[problem] = {}
    
    prob_path = os.path.join(results_path, problem)
    
    print(f"Processing results from {prob_path}")
    
    if len(params['RUN_LIST']) == 0:
        run_idxs = [int(d.path.split("_")[-1]) 
                        for d in os.scandir(prob_path) 
                        if d.is_dir() and "Plots" not in d.path]
        run_idxs.sort()
    else:
        run_idxs = params['RUN_LIST']
       
    for method in params['METHODS']:
        mode = '' if method == 'TPOT-BASE' else 'discrete' if 'd' in method else 'continuous' if 'c' in method else 'sequential'
        raw_method = method.strip('dcs')
        
        # validate and collect data from specified runs
        for run in run_idxs:
            run_str = str(run)    
            if run < 10:
                run_str = "0" + str(run)
        
            run_path = os.path.join(prob_path,"Run_" + run_str)
            
            data[problem][method] = {}
            
            f_pipes = os.path.join(run_path,raw_method,mode,f'{raw_method}.tpipes') if 'ALT' in method else os.path.join(run_path,raw_method,mode,f'{raw_method}.pipes')
            
            # if not os.path.exists(f_pipes):
            #     print(f"{u.RED}Missing file error:{u.OFF} Run {run} of " 
            #           + f"{problem} is missing file {os.path.basename(f_pipes)}\n{f_pipes}"
            #           + " - skipping run..")
            #     skipped_runs.append(run)
            #     break
            
            u_pop = u.load_unique_auto_pop(f_pipes) if 'AUTO' in method else u.load_unique_pop(f_pipes)
            
            d_matrix = np.zeros((len(u_pop),len(u_pop)))
            u_keys = list(u_pop.keys())
            for i in range(d_matrix.shape[0]):
                src = at.helpers.Tree.from_text(u_keys[i])
                for j in range(i+1, d_matrix.shape[0]):
                    tgt = at.helpers.Tree.from_text(u_keys[j])
                    d = at.APTED(src,tgt).compute_edit_distance()
                    d_matrix[i,j] = d
                    d_matrix[j,i] = d
            
            with open(f_dist, 'a') as f:
                f.write(f"({problem},{method},{run}):{d_matrix.shape[0]};{np.median(d_matrix)};{np.mean(d_matrix)};{np.std(d_matrix)}\n")
            
            print(f"({problem},{method},{run}):{d_matrix.shape[0]};{np.median(d_matrix)};{np.mean(d_matrix)};{np.std(d_matrix)}")
            
                
                # if raw_method == 'TPOT-BO-S' or (raw_method == 'TPOT-BO-H' and method != 'TPOT-BO-Hs'):
                #     read_data = False
                #     n_evals = 0
                #     for line in f:
                #         if 'BAYESIAN OPTIMISATION EVALS:' in line:
                #             n_evals = int(line.split(':')[-1])
                #         if f'{n_evals} BAYESIAN OPTIMISATION' in line:
                #             read_data = True
                #         if 'Best CV' in line and read_data:
                #             data[problem][run][method] = -float(line.split(":")[-1])
                            
                # if method == 'TPOT-BO-Hs':
                #     for line in f:
                #         if 'AFTER' in line and 'TPOT-BO-Hs' in line:
                #             next(f)
                #             cv_line = next(f)
                #             data[problem][run][method] = -float(cv_line.split(":")[-1])
                            
                # if raw_method == 'TPOT-BO-ALT':
                #     read_data = False
                #     final_iter = 0
                #     for line in f:
                #         if 'nITERS' in line:
                #             final_iter = int(line.split(':')[-1])
                #         if f'ITERATION {final_iter-1}' in line:
                #             read_data = True
                #         if 'Best BO CV' in line and read_data:
                #             data[problem][run][method] = -float(line.split(":")[-1])
                            
                # if raw_method == 'TPOT-BO-AUTO':
                #     read_data = False
                #     final_gen = 100
                #     for line in f:
                #         if 'nGENS' in line:
                #             final_iter = int(line.split(':')[-1])
                #         if f'GENERATION {final_gen-1}' in line:
                #             read_data = True
                #         if 'Best CV' in line and read_data:
                #             data[problem][run][method] = -float(line.split(":")[-1])
                    
            # if method not in data[problem][run]:
            #     print(f"{u.RED}Data read error:{u.OFF} {f_prog} does not contain valid data for {method} for run {run}")
            
            
        # if run in skipped_runs:
        #     continue            

# for prob, prob_d in data.items():
#     for method, method_d in prob_d.items():
#         for run,run_d in method_d.items():
#             d_matrix = np.zeros((len(run_d['u_pop']),len(run_d['u_pop'])))
#             u_keys = list(run_d['u_pop'].keys())
#             for i in range(d_matrix.shape[0]):
#                 src = at.helpers.Tree.from_text(u_keys[i])
#                 for j in range(i+1, d_matrix.shape[0]):
#                     tgt = at.helpers.Tree.from_text(u_keys[j])
#                     d = at.APTED(src,tgt).compute_edit_distance()
#                     d_matrix[i,j] = d
#                     d_matrix[j,i] = d
                    
#             print(f"no. structures: {method_d['d_matrix'].shape[0]}")
#             print(f"mean: {np.mean(method_d['d_matrix'])}")
# t_end = time.time()

# print(f"done!\nTime elapsed: {t_end-t_start:.2f} s")