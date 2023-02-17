import sys
import os
import copy
from utils import tpot_utils as u
from config.tpot_config import default_tpot_config_dict
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from utils.warning_filters import SimpleFilter


PROBLEM = 'socmob'
RESULTS_PATH = 'Results_try'
STOP_GEN = 80
SEEDS = [60]

cwd = os.getcwd()

large_mean = []
large_std = []

for seed in SEEDS:

    pipe_path = os.path.join(cwd,RESULTS_PATH,PROBLEM,'TPOT-BASE',f'Seed_{seed}','TPOT-BASE.pipes')
    pipes = u.get_progress_pop(pipe_path,stop_gen=STOP_GEN)

    cvs = np.array([v['internal_cv_score'] for v in pipes.values() if v['internal_cv_score'] > -np.inf])

    # get unique structures
    strucs = u.get_structures(pipes, config_dict=default_tpot_config_dict)
            
    # get keys by CV ranking
    struc_cvs = np.array([-strucs[k].cv for k in strucs.keys()])
    struc_cv_idxs = np.argsort(struc_cvs)        
    key_list = list(strucs.keys())

    remove_ids = []                
    # check key_list for any structures that cannot be used
    for i in range(struc_cv_idxs.shape[0]):
        if strucs[key_list[struc_cv_idxs[i]]].n_bo_params < 1:
            remove_ids.append(i)

    struc_cv_idxs = np.delete(struc_cv_idxs, remove_ids)
                    
    bo_struc_keys = [key_list[i] for i in struc_cv_idxs[:50]]

    for k in bo_struc_keys:
        valid_pipes = strucs[k].get_valid()            
        if strucs[k].mu < -1e6:
            large_mean.append(seed)
        if strucs[k].std > 1e6:
            large_std.append(seed)
        print(f"{u.CYAN}{k}{u.RED}")
        print(f"no. valid pipes: {len(valid_pipes)}")
        print(f"best cv: {-strucs[k].cv}")
        print(f"mean cv: {-strucs[k].mu}")
        print(f"std dev: {strucs[k].std}")
        print(f"median cv: {-strucs[k].median}{u.OFF}")
        if len(valid_pipes) > 1:
            strucs[k].update_stats(check_outliers=True)
            if strucs[k].n_outliers > 0:
                print(f"removing {strucs[k].n_outliers} outliers..")
                print(f"{u.GREEN}updated mean cv: {-strucs[k].mu_o}")
                print(f"updated std dev: {-strucs[k].std_o}")
                print(f"updated median cv: {-strucs[k].median_o}")

            # k_cvs = np.array([v['internal_cv_score'] for v in valid_pipes.values()]).reshape(-1,1)
            # # isof = IsolationForest(contamination='auto', behaviour='new', random_state=42)
            # with SimpleFilter('n_neighbors is greater than or equal to the number of samples', action='ignore'):
            #     lof = LocalOutlierFactor(contamination='auto')
            #     outliers = lof.fit_predict(k_cvs)
            # n_outliers = len(outliers[outliers == -1])     
            # if n_outliers > 0:
            #     print(f"removing {n_outliers} outliers..")
            #     # get indices of non outliers
            #     good_ids = np.flatnonzero(outliers == 1)
            #     good_cvs = -k_cvs[good_ids]
            #     print(f"{u.GREEN}updated mean cv: {np.mean(good_cvs)}")
            #     print(f"updated std dev: {np.std(good_cvs)}")
            #     print(f"updated median cv: {np.median(good_cvs)}")
                
        print(f"{u.OFF}")
    
    
# print(f"\nlarge mean: {set(large_mean)}\nlarge std:{set(large_std)}")