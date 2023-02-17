# -*- coding: utf-8 -*-
''' Utility functions for the TPOT hyperparameter optimisation code
'''

import sys
import numpy as np
import pandas as pd
from utils.data_structures import StructureCollection
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting, find_non_dominated
import copy
import re

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


def disp_ocba_tracking(tracking, Deltas, colours=True):
    cols = [CYAN,RED] if colours else ["",""]
    off = OFF if colours else ""
    
    out = ""
    for g in range(len(tracking)):
        title = f"{cols[g%2]}Generation {g} (Delta = {Deltas[g]}):" 
        out += f"{title}\n"
        out += f"{'='*len(title)}\n"
        max_val = max(tracking[g])
        while max_val > 0:
            out += f"      "
            for i in range(len(tracking[g])):
                track = f" * " if tracking[g][i] >= max_val else "   "
                out += track
            out += "\n"
            max_val -= 1
        n_gen_evals = np.sum([tracking[g][j] > 0 for j in range(len(tracking[g]))])
        out += "      "
        for i in range(len(tracking[g])):
            track = f"{cols[i%2]}{tracking[g][i]:^3}{off}"
            out += track
        out += f" -> {sum(tracking[g])} total, {n_gen_evals} unique\n\n"
    
                            
    # out += f"{off}\n"
    # for i in range(len(tracking)):
    #     out += f"   {i}: "
    #     for j in range(len(tracking[i])):
    #         track = f"{cols[j%2]}{tracking[i][j]:^3}{off}" if tracking[i][j] else "   "
    #         out += track
    #     n_gen_evals = np.sum([tracking[i][j] > 0 for j in range(len(tracking[i]))])
    #     out += f" {n_gen_evals:>2} (m = {m_strucs[i]})\n"

    return out

def flatten(A):
    rt = []
    for i in A:
        if isinstance(i,list): rt.extend(flatten(i))
        else: rt.append(i)
    return rt

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


def get_progress_pop(fname_pipes, stop_gen=np.inf):
    ''' call with stop_gen-1 as parameter because of initial pop generation
    '''
    
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

def update_group(group):
    group_cvs = [v['internal_cv_score'] for k,v in group['matching'].items()]
    best_idx = np.argmax(group_cvs)
    best_pipe = list(group['matching'].keys())[best_idx]
    group['best_pipe'] = best_pipe
    group['cv_mu'] = np.mean(group_cvs)
    group['cv_best'] = np.max(group_cvs)
    group['internal_cv_score'] = np.max(group_cvs)
    group['cv_worst'] = np.min(group_cvs)
    group['cv_sigma'] = np.std(group_cvs)
    # group['n_root'] = np.power(len(group['matching']),1/len(group['bo_params']))
    
    return group

def make_new_group(pipe, vals, config_dict=None):
    new_group = {
        'best_pipe' : pipe,
        'cv_mu' : vals['internal_cv_score'],
        'cv_sigma' : 0,
        'cv_best' : vals['internal_cv_score'],
        'internal_cv_score' : vals['internal_cv_score'],
        'cv_worst' :vals['internal_cv_score'],
        'matching' : {pipe: copy.deepcopy(vals)},
        'params' : string_to_params(pipe),
        'operators' : string_to_ops(pipe),
        'n_operators' : len(string_to_ops(pipe)),
        'structure' : string_to_structure(pipe),
        'bo_params' : string_to_params(pipe,config_dict=config_dict),
        'n_bo_params' : len(string_to_params(pipe,config_dict=config_dict))
        }
    return new_group

def get_structures(pipes, stop_gen=np.inf, config_dict=None):
    strucs = StructureCollection(config_dict=config_dict)
    
    for i,(p,v) in enumerate(pipes.items()):        
        if v['generation'] > stop_gen:
            continue
        
        # struc_str = string_to_bracket(p)
        strucs.add(p,v)
        
    return strucs


def get_unique_groups(pipes, stop_gen=np.inf, config_dict=None):
    strucs = {}
    
    for k,v in pipes.items():
        # struc = string_to_structure(k)
        # struc_str = str(struc)
        
        struc_str = string_to_bracket(k)

        v['group'] = struc_str
        
        # skip invalid or past stop_gen
        if v['generation'] > stop_gen:# or v['internal_cv_score'] == -np.inf or len(string_to_params(k,config_dict=config_dict)) == 0:
            continue
        
        if struc_str in strucs:
            strucs[struc_str]['matching_pipes'][k] = copy.deepcopy(v)
            strucs[struc_str]['matching_cv'].append(v['internal_cv_score'])
            strucs[struc_str]['matching_keys'].append(k)
            
            continue
        
        
        strucs[struc_str] = {'matching_pipes': 
                             {k: copy.deepcopy(v)},
                             'matching_cv': [v['internal_cv_score']],
                             'matching_keys': [k]}

    unique_groups = {}
    for s,v in strucs.items():
        unique_groups[s] = {}
        best_idx = np.argmax(v['matching_cv'])
        best_pipe = v['matching_keys'][best_idx]
        unique_groups[s]['best_pipe'] = best_pipe
        unique_groups[s]['cv_mu'] = np.mean(v['matching_cv'])
        # unique_groups[s]['bracket'] = string_to_bracket(best_pipe)
        unique_groups[s]['cv_sigma'] = np.std(v['matching_cv'])
        unique_groups[s]['cv_best'] = np.max(v['matching_cv'])
        unique_groups[s]['internal_cv_score'] = np.max(v['matching_cv'])
        unique_groups[s]['cv_worst'] = np.min(v['matching_cv'])
        unique_groups[s]['matching'] = copy.deepcopy(v['matching_pipes'])
        unique_groups[s]['params'] = string_to_params(best_pipe)
        unique_groups[s]['operators'] = string_to_ops(best_pipe)
        unique_groups[s]['n_operators'] = len(string_to_ops(best_pipe)) 
        unique_groups[s]['structure'] = string_to_structure(best_pipe)
        unique_groups[s]['bo_params'] = string_to_params(best_pipe,config_dict=config_dict)
        unique_groups[s]['n_bo_params'] = len(unique_groups[s]['bo_params'])
        # unique_groups[s]['n_root'] = np.power(len(unique_groups[s]['matching']),1/len(unique_groups[s]['bo_params']))
    
    return unique_groups


def load_unique_pop(fname_pipes, stop_gen=np.inf, config_dict=None):
    strucs = {}
    total_best_pipe = None
    total_best_cv = -np.inf
    with open(fname_pipes, 'r') as f:
        for line in f:
            line_s = line.split(';')
            pipe_str = line_s[0]
            gen = int(line_s[1])
            cv = float(line_s[-1])
            op_count = len(string_to_ops(pipe_str))
            # struc = string_to_structure(pipe_str)
            # struc_str = str(struc)
            
            struc_str = string_to_bracket(pipe_str)
            
            # skip invalid or past stop_gen
            if gen > stop_gen or cv == -np.inf:
                continue
            
            if cv > total_best_cv:
                total_best_cv = cv
                total_best_pipe = pipe_str
            
            if struc_str in strucs:
                strucs[struc_str]['matching_pipes'][pipe_str] = {'internal_cv_score':cv,
                                                                 'operator_count':op_count,
                                                                 'generation':gen}
                strucs[struc_str]['matching_cv'].append(cv)
                strucs[struc_str]['matching_keys'].append(pipe_str)
            else:            
                strucs[struc_str] = {'matching_pipes': 
                                    {pipe_str: {'internal_cv_score':cv,
                                                'operator_count':op_count,
                                                'generation':gen}},
                                    'matching_cv': [cv],
                                    'matching_keys': [pipe_str]}

    unique_pipes = {}
    for s,v in strucs.items():
        best_idx = np.argmax(v['matching_cv'])
        best_pipe = v['matching_keys'][best_idx]
        unique_pipes[s] = v['matching_pipes'][best_pipe]
        unique_pipes[s]['selected?'] = True if best_pipe == total_best_pipe else False
        # unique_pipes[best_pipe]['bracket'] = string_to_bracket(best_pipe)
        unique_pipes[s]['cv_mu'] = np.mean(v['matching_cv'])
        unique_pipes[s]['cv_sigma'] = np.std(v['matching_cv'])
        unique_pipes[s]['cv_best'] = np.max(v['matching_cv'])
        unique_pipes[s]['internal_cv_score'] = np.max(v['matching_cv'])
        unique_pipes[s]['cv_worst'] = np.min(v['matching_cv'])
        unique_pipes[s]['matching'] = v['matching_pipes']
        unique_pipes[s]['params'] = string_to_params(best_pipe)
        unique_pipes[s]['operators'] = string_to_ops(best_pipe)
        unique_pipes[s]['n_operators'] = len(string_to_ops(best_pipe))
        unique_pipes[s]['bo_params'] = string_to_params(best_pipe,config_dict=config_dict)
    
    return unique_pipes

def load_unique_auto_pop(fname_pipes, stop_gen=np.inf, config_dict=None):
    '''havent done anything with type string but probably fix that up when its standardised'''
    strucs = {}
    total_best_pipe = None
    total_best_cv = -np.inf
    with open(fname_pipes, 'r') as f:
        for line in f:
            line_s = line.split(';')
            pipe_str = line_s[0]
            gen = int(line_s[1])
            cv = float(line_s[-2])
            op_count = len(string_to_ops(pipe_str))
            # struc = string_to_structure(pipe_str)
            # struc_str = str(struc)
            
            struc_str = string_to_bracket(pipe_str)
            
            # skip invalid or past stop_gen
            if gen > stop_gen or cv == -np.inf:
                continue
            
            if cv > total_best_cv:
                total_best_cv = cv
                total_best_pipe = pipe_str
            
            if struc_str in strucs:
                strucs[struc_str]['matching_pipes'][pipe_str] = {'internal_cv_score':cv,
                                                                 'operator_count':op_count,
                                                                 'generation':gen}
                strucs[struc_str]['matching_cv'].append(cv)
                strucs[struc_str]['matching_keys'].append(pipe_str)
                
                continue
            
            
            strucs[struc_str] = {'matching_pipes': 
                                 {pipe_str: {'internal_cv_score':cv,
                                             'operator_count':op_count,
                                             'generation':gen}},
                                 'matching_cv': [cv],
                                 'matching_keys': [pipe_str]}

    unique_pipes = {}
    for s,v in strucs.items():
        best_idx = np.argmax(v['matching_cv'])
        best_pipe = v['matching_keys'][best_idx]
        unique_pipes[s] = v['matching_pipes'][best_pipe]
        unique_pipes[s]['selected?'] = True if best_pipe == total_best_pipe else False
        # unique_pipes[best_pipe]['bracket'] = string_to_bracket(best_pipe)
        unique_pipes[s]['cv_mu'] = np.mean(v['matching_cv'])
        unique_pipes[s]['cv_sigma'] = np.std(v['matching_cv'])
        unique_pipes[s]['cv_best'] = np.max(v['matching_cv'])
        unique_pipes[s]['internal_cv_score'] = np.max(v['matching_cv'])
        unique_pipes[s]['cv_worst'] = np.min(v['matching_cv'])
        unique_pipes[s]['matching'] = v['matching_pipes']
        unique_pipes[s]['params'] = string_to_params(best_pipe)
        unique_pipes[s]['operators'] = string_to_ops(best_pipe)
        unique_pipes[s]['n_operators'] = len(string_to_ops(best_pipe))
        unique_pipes[s]['bo_params'] = string_to_params(best_pipe,config_dict=config_dict)
    
    return unique_pipes


def load_bhs_pipes(fname_pipes, cutoff_pop=1):
    pop = {}
    
    with open(fname_pipes, 'r') as f:
        for line in f:
            line_s = line.split(';')
            pipe = line_s[0]
            gen = int(line_s[1])
            n_bo_pop = int(line_s[2])
            source = line_s[3]
            cv = float(line_s[4])
            op_count = len(string_to_ops(pipe))
            if n_bo_pop > cutoff_pop:
                pop[pipe] = {'internal_cv_score':cv,
                             'operator_count':op_count,
                             'generation':gen,
                             'source':source,
                             'n_bo_pop':n_bo_pop}
                        
    return pop

def load_bhd_pipes(fname_pipes, cutoff_pop=1):
    pop = {}
    
    with open(fname_pipes, 'r') as f:
        for line in f:
            line_s = line.split(';')
            pipe = line_s[0]
            struct_str = line_s[1]
            gen = int(line_s[2])
            n_bo_pop = int(line_s[3])
            source = line_s[4]
            cv = float(line_s[5])
            op_count = len(string_to_ops(pipe))
            if n_bo_pop > cutoff_pop:
                pop[pipe] = {'internal_cv_score':cv,
                             'operator_count':op_count,
                             'generation':gen,
                             'source':source,
                             'n_bo_pop':n_bo_pop,
                             'structure':struct_str}
                        
    return pop

def truncate_pop(pipes, stop_gen):
    sub_pop = {}
    for k,v in pipes.items():
        if v['generation'] <= stop_gen:
            sub_pop[k] = v

    return sub_pop


def string_to_structure(pipe_str):
    # split string and flatten to separate elements
    ps = flatten([v.split(",") for v in pipe_str.split("(")])
    ps = [v.replace(" ","").replace(")","") for v in ps]
    # remove hyper-parameter values
    ps = [v.split("=")[0] for v in ps]
    return ps

def string_to_bracket(pipe_str):
    ps = ""
    
    for p in pipe_str.replace("(","{").replace(")","}").split(" "):
        if "=" not in p:
            ps = ps + "{" + p.replace(",", "}")
        else:
            if "}" not in p:
                continue
            ps = ps + re.sub(r"[^}]", "", p)
    
    return ps

def string_to_ops(pipe_str):
    ''' Extract operators from pipeline string
    '''
    ops = string_to_structure(pipe_str)
    # remove input matrix and anything with "=" in 
    rem_idx = []
    for o in ops:
        if 'input_matrix' in o or '__' in o:
            rem_idx.append(o)
    
    [ops.remove(i) for i in rem_idx]
    
    return ops

def add_param_to_pset(tpot, p_name, p_val,verb=0):
    '''
    if new value then add to operator value list and update pset
    the operator values list gets reset with every call to fit()
    so we must add it to the operator values list, even if the
    value already exists in the pset from previous evaluations
    '''
    vprint = Vprint(verb)
    if is_number(p_val):
        for op in tpot.operators:
            if op.__name__ in p_name:
                # find parameter type in parameter list (skip index 0)
                for p_type in op.parameter_types()[0][1:]:
                    if p_type.__name__ == p_name:
                        # if value missing from p_type values list
                        if p_val not in p_type.values:
                            # update values list
                            if type(p_type.values) == np.ndarray:
                                np.append(p_type.values, p_val)
                            elif type(p_type.values) == list:
                                p_type.values.append(p_val)
                            else:
                                vprint.v2("Cannot update value list for " 
                                        + p_type.__name__)
                            # update pset new value not already used
                            if (p_type.__name__ + "=" + str(p_val) 
                                not in tpot._pset.context):
                                vprint.v2("adding " + p_type.__name__+ "="
                                        + str(p_val) + " to pset")
                                tpot._pset.addTerminal(
                                    p_val, p_type, 
                                    name=(p_type.__name__ 
                                            + "=" + str(p_val)))   


def get_matching_set(tgt, pipes):
    ''' Using tgt as the target string, get all strings representing
        pipelines that have been evaluated by TPOT so far, with matching
        structures
        stop_gen indicates maximum generation to get pipes from
    '''
    tgt_struc = string_to_structure(tgt)
    matching_set = {}
    
    for k,v in pipes.items():
        if string_to_structure(k) == tgt_struc:
            matching_set[k] = copy.deepcopy(v)
            
    return matching_set


def string_to_params(pipe_str, config_dict=None):
    ''' Extract parameters from pipeline as parameter list
        If you want bo params, pass in a config dict
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
    
    if config_dict:
        rem_p = []
        for p in params:
            p_s = p[0].split("__")
            for k,v in config_dict.items():
                if p_s[0] in k:
                    if p_s[0] == "SelectFromModel" and len(p_s) == 3:
                        for k2,v2 in v['estimator'].items():
                            if p_s[1] in k2:
                                if len(v2[p_s[2]]) == 1:
                                    rem_p.append(p)            
                    elif len(v[p_s[1]]) == 1:
                        rem_p.append(p)
                        
        [params.remove(v) for v in rem_p]
    
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

    # These are predictor variables
    X=df

    # Splitting the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.3, random_state = 0)
    
    return X_train, X_test, y_train, y_test

def get_best_structures(strucs, size=1):
    cvs = np.array([-v.cv for v in strucs.values()])
    sorted_idx = np.argsort(cvs)
    sk = list(strucs.keys())
    best_set = [sk[i] for i in sorted_idx[:size]]
    return best_set
            
def get_best(pipes, source=None, size=None):
    '''
    TODO: implement source for size > 1
    '''
    if size:
        cvs = np.array([-v['internal_cv_score'] for k,v in pipes.items()])
        sorted_idx = np.argsort(cvs)
        pk = list(pipes.keys())
        best_set = {pk[i]:copy.deepcopy(pipes[pk[i]]) for i in sorted_idx[:size]}
        return best_set
            
    else:
        best_pipe = ""
        best_cv = -1e20
        for k,v in pipes.items():
            if not source is None:
                if v['internal_cv_score'] > best_cv and source == v['source']:
                    best_pipe = k
                    best_cv = v['internal_cv_score']
            elif v['internal_cv_score'] > best_cv:
                best_pipe = k
                best_cv = v['internal_cv_score']
            
        return best_pipe,best_cv        


def get_nd_best(pipes, nd_params, source=None):
    return {}
#     '''
#     If Pareto front required, pass params to sort by as nd_params, with 
#     negative if negative required. e.g., "-n_root" for -1 * 'n_root'
#     '''

#     pareto_pipes = {}
#     nd1_mod = -1 if "-" in nd_params[0] else 1
#     nd2_mod = -1 if "-" in nd_params[1] else 1
#     if source is None:
#         points = np.hstack((np.array([nd1_mod*v[nd_params[0].strip("-")] 
#                                       for k,v in pipes.items()]).reshape(-1,1), 
#                             np.array([nd2_mod * v[nd_params[1].strip("-")] 
#                                       for k,v in pipes.items()]).reshape(-1,1)))
#     else:
#         points = np.hstack((np.array([nd1_mod*v[nd_params[0].strip("-")] 
#                                       for k,v in pipes.items() 
#                                       if source in v['source']]).reshape(-1,1), 
#                             np.array([nd2_mod * v[nd_params[1].strip("-")] 
#                                       for k,v in pipes.items() 
#                                       if source in v['source']]).reshape(-1,1)))
        
#         idxs = [0]            
#         if points.shape[0] > 1:
#             ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points=points)
#             idxs = ndf[0]
        
#         p_list = list(pipes.keys())
        
#         for i in idxs:
#             pareto_pipes[p_list[i]] = copy.deepcopy(pipes[p_list[i]])

#         return pareto_pipes

def get_max_gen(pipes):
    max_gen = 0
    for k,v in pipes.items():
        max_gen = max(max_gen, v['generation'])

    return max_gen


def loguniform(low, high, size=None):
    if low == 0 or high == 0:
        print("Error: cannot do log of 0!")
    return np.exp(np.random.uniform(np.log(low), np.log(high), size))


def get_restricted_set(pipes, config_dict):
    
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
    
    return freeze_params[:freeze_idx], freeze_idx, n_params