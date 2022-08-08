# -*- coding: utf-8 -*-
''' Utility functions for the TPOT hyperparameter optimisation code
'''

import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

''' Escape codes for printing coloured text
'''
WHITE = "\033[0;38;49m"
WHITE_U = "\033[4;38;49m"
GREEN = "\033[0;32;49m"
GREEN_U = "\033[4;32;49m"
RED = "\033[0;31;49m"
RED_U = "\033[4;31;49m"
YELLOW = "\033[0;33;49m"
YELLOW_U = "\033[4;33;49m"
CYAN = "\033[0;36;49m"
CYAN_U = "\033[4;36;49m"
GREY = "\033[0;37;49m"
OFF = '\x1b[0m'


class Vprint(object):
    def __init__(self, verbosity):
        self.verbosity = verbosity
    
    def v0(self, *args, **kwargs):
        print(*args, **kwargs)
    
    def v1(self, *args, **kwargs):
        if self.verbosity >= 1:
            print(*args, **kwargs)
            
    def v2(self, *args, **kwargs):
        if self.verbosity >= 2:
            print(*args, **kwargs)
    
    def v3(self, *args, **kwargs):
        if self.verbosity >= 3:
            print(*args, **kwargs)
            
    def verr(self, *args, **kwargs):
        print(f"{RED}Error:{OFF}",*args, **kwargs)
        
    def vwarn(self, *args, **kwargs):
        print(f"{RED}Warning:{OFF}",*args, **kwargs)


def is_number(string):
    ''' Check if string is a number
    '''    
    try:
        float(string)
        return True
    except ValueError:
        return False

def convert_str_param(p,config_dict):
    try:
        return float(p[1])
    except ValueError:
        if p[1] in 'TrueFalse':
            return bool(p[1])
        
        p_s = p[0].split("__")
        for k,v in config_dict.items():
            if p_s[0] in k:
                return v[p_s[1]].index(p[1])

def get_run_data(fname_prog):
    seed = n_tot_gens = tpot_stop_gen = pop_size = None
    
    # read trial data in
    with open(fname_prog, 'r') as f:
        for line in f:
            if 'SEED' in line:
                seed = int(line.split(":")[-1])
            if 'SIZE' in line:
                pop_size = int(line.split(":")[-1])
            if 'STOP' in line:
                tpot_stop_gen = int(line.split(":")[-1])
            if 'TOTAL' in line:
                n_tot_gens = int(line.split(":")[-1])
            if '*' in line:
                break
    
    return seed, n_tot_gens, tpot_stop_gen, pop_size

def get_progress_pop(fname_pipes, stop_gen):
    pop = {}
    
    with open(fname_pipes, 'r') as f:
        for line in f:
            line_s = line.split(';')
            pipe = line_s[0]
            gen = int(line_s[1])
            cv = float(line_s[2])
            op_count = len(string_to_ops(pipe))
            if gen <= stop_gen:
                pop[pipe] = {'internal_cv_score':cv,
                             'operator_count':op_count,
                             'generation':gen}
                        
    return pop


def string_to_ops(pipe_str):
    ''' Extract operators from pipeline string
    '''
    # split string to separate operators from parameters
    split_str = pipe_str.split("(")
    # operators are in all but last elements
    ops = [s.replace("input_matrix, ","") for s in split_str[0:-1]]
    
    return ops
    

def string_to_params(pipe_str):
    ''' Extract parameters from pipeline as parameter list
    '''
    params = []
    # params are in the last element - split on "," to separate params
    for split_str in pipe_str.split(","):
        if "input_matrix" in split_str:
            continue
        # remove open bracket
        param_str = split_str.split("(")[-1]
        # remove spaces and brackets and split on "="
        param_str = param_str.replace(" ",'').replace(")","").split("=")
        if not is_number(param_str[1]):
            params.append((param_str[0], param_str[1]))
        elif "." in param_str[1] or 'e' in param_str[1]:
            params.append((param_str[0], float(param_str[1])))
        else:
            params.append((param_str[0], int(param_str[1])))
        
    return params


def load_data(fpath):
    ''' Load data from file and return training and test sets

    Parameters
    ----------
    fpath : STRING
        Path to data file.

    Returns
    -------
    X_train, X_test, y_train, y_test
        Training and test data sets 
    '''
    
    header = None
    # check first line of datafile to see if there is a header
    with open(fpath, 'r') as f:
        line = f.readline()
        line_s = line.split(",")
        # if *all* fields in first line are strings, must be header 
        # (old abalone data had categorical data)
        if sum([is_number(s) for s in line_s]) == 0:
            header = 0
    
    df = pd.read_csv(fpath, header = header)

    # Identifying the variable type in each column
    var_type=[]
    for y in df.columns:
        if(df[y].dtype == np.float64 or df[y].dtype == np.int64):
              var_type.append("F")
        else:
              var_type.append("C")

    # Identifying the number of columns
    ncol = df.shape[1]

    # Renaming column headers
    df.columns=["F"+str(i) for i in range(0, ncol)]

    # Extracting the response
    y=df["F"+str(ncol-1)]

    # Dropping the response column
    df=df.drop(["F"+str(ncol-1)],axis=1)

    # Every colum of categorical variables transformed via one-hot encoding
    for i in range(0, ncol):
        if(var_type[i]=="C"):
            header="F"+str(i)
            df[header] = LabelEncoder().fit_transform(df[header].tolist())
            # # transform categorical variables to one-hot (only for GP)
            # transformed_feature = OneHotEncoder().fit_transform(
            #     df[header].values.reshape(-1,1)).toarray()
            # df_encoded = pd.DataFrame(
            #     transformed_feature, 
            #     columns = [header+"_"+str(int(j)) 
            #                 for j in range(transformed_feature.shape[1])])
            
            # Dropping the above column
            # df=df.drop([header],axis=1)

            # Adding the one hot encoded variables
            # df = pd.concat([df, df_encoded], axis=1)
            

    # These are predictor variables
    X=df

    # Splitting the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.3, random_state = 0)
    
    return X_train, X_test, y_train, y_test

def loguniform(low, high, size=None):
    if low == 0 or high == 0:
        print("Error: cannot do log of 0!")
    return np.exp(np.random.uniform(np.log(low), np.log(high), size))


def get_restricted_set(pipes, config_dict, fname_bo_res_plot):
    
    best_pipe = next(iter(pipes))
    
    best_params = string_to_params(best_pipe)
    
    param_list = [v[0] for v in best_params]
    
    rem_p = []
    
    for p in param_list:
        p_s = p.split("__")
        for k,v in config_dict.items():
            if p_s[0] in k:
                if len(v[p_s[1]]) == 1:
                    rem_p.append(p)
    
    [param_list.remove(v) for v in rem_p]
    
    n_params = len(param_list)
    
    hp_x = np.empty((0,n_params))        
    hp_y = np.empty((0,1))        
    
    for pipe,v in pipes.items():
        if v['internal_cv_score'] == -np.inf:
            continue
        pipe_params = string_to_params(pipe)
        hp_x = np.vstack((hp_x,np.array([convert_str_param(v,config_dict) 
                                         for v in pipe_params if v[0] 
                                         in param_list])))
        hp_y = np.vstack((hp_y, np.array([v['internal_cv_score']])))
    
    scores = []
    x_vals = []
    freeze_params = []
    
    while hp_x.shape[1] > 0:
        x_vals.append(hp_x.shape[1])
        regressor = LinearRegression(fit_intercept=False)
        regressor.fit(hp_x, hp_y)
        scores.append(regressor.score(hp_x, hp_y))
        worst_idx = np.argmin(np.abs(regressor.coef_))
        freeze_params.append(param_list.pop(worst_idx))
        hp_x = np.delete(hp_x, worst_idx, axis=1)
        
    freeze_idx = np.argmin(np.abs(scores-0.95*np.max(scores)))
    
    plot_data = np.hstack((np.array(x_vals).reshape(-1,1), np.array(scores).reshape(-1,1)))
    np.savetxt(fname_bo_res_plot, plot_data, delimiter=',')
    
    return freeze_params[:freeze_idx], freeze_idx, n_params