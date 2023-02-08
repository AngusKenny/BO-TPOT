# -*- coding: utf-8 -*-
from config.tpot_config import default_tpot_config_dict
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting, find_non_dominated
import utils.tpot_utils as u
import numpy as np

class PipeStructure(object):
    def __init__(self, pipe, config_dict=None) -> None:
        self.config_dict = config_dict
        self.pipes = {}
        self.best = pipe
        self.cv = -1e20
        self.mu = -1e20
        self.std = 0.0
        self.stderr = 0.0
        self.structure = u.string_to_bracket(pipe)
        self.params = u.string_to_params(pipe)
        self.bo_params = u.string_to_params(pipe,config_dict=config_dict)
        self.n_bo_params = len(self.bo_params)
        self.operators = u.string_to_ops(pipe)
    
    def __str__(self):
        return self.structure
    
    def __len__(self):
        return len(self.pipes)    
        
    def __getitem__(self, key):
        return self.pipes[key]
  
    def __setitem__(self, key, newvalue):
        self.pipes.add(key, newvalue)

    def __contains__(self, key):
        return key in self.pipes

    def update_stats(self):
        self.mu = np.mean([v['internal_cv_score'] for v in self.get_valid().values()])
        self.std = np.std([v['internal_cv_score'] for v in self.get_valid().values()])
        self.stderr = self.std/np.sqrt(len(self))
        return self.mu, self.std, self.stderr
    
    def add(self, pipe, data) -> None:
        if pipe not in self.pipes:
            data['structure'] = self.structure
            self.pipes[pipe] = data
            if data['internal_cv_score'] > self.cv:
                self.cv = data['internal_cv_score']
                self.best = pipe
                self.bo_params = u.string_to_params(pipe,config_dict=self.config_dict)
                self.params = u.string_to_params(pipe)
            
            self.update_stats()
    
    def get_best(self, n = 1):
        pipe_keys = list(self.pipes.keys())
        cvs = [v['internal_cv_score'] for v in self.pipes.values()]
        sorted_ids = np.argsort(cvs)
        return_pipes = []
        for i in range(n):
            return_pipes.append(pipe_keys[i % len(self)])
            
        return return_pipes
    
    def get_valid(self):
        valid_pipes = {k : v for k,v in self.pipes.items() if v['internal_cv_score'] > -np.inf}
        return valid_pipes
    
    def get_seed_samples(self):
        return [(u.string_to_params(p), v['internal_cv_score']) for p,v in self.pipes.items()]
    
    # work out what to do if we remove the last one (maybe try/catch?)
    def remove(self, pipe) -> None:
        # remove from pipe list
        self.pipes.pop(pipe)
        if self.best == pipe:
            # update best
            self.cv = -1e20
            for p,v in self.pipes.items():
                if v['internal_cv_score'] > self.cv:
                    self.cv = v['internal_cv_score']
                    self.best = p

class StructureCollection(object):    
    def __init__(self, config_dict=None) -> None:
        self.structures = {}
        self.best = ""
        self.cv = -1e20
        self.config_dict = config_dict

    def __getitem__(self, key):
        return self.structures[key]
  
    def __contains__(self, key):
        return key in self.structures
  
    # dont use this unless copying directly from another collection!
    def __setitem__(self, key, newvalue):
        self.structures[key] = newvalue

    def __len__(self):
        return len(self.structures)

    def keys(self):
        return self.structures.keys()
    
    def items(self):
        return self.structures.items()
    
    def values(self):
        return self.structures.values()

    def index(self, struc_str):
        return list(self.structures.keys()).index(struc_str)

    def get_by_index(self, idx):
        return self.structures[list(self.structures.keys())[idx]]

    def add(self, pipe_str, data):
        struc_str = u.string_to_bracket(pipe_str)
        if struc_str not in self.structures:
            self.structures[struc_str] = PipeStructure(pipe_str,self.config_dict)
        
        if pipe_str not in self.structures[struc_str]:
            self.structures[struc_str].add(pipe_str,data)
        
        if data['internal_cv_score'] > self.cv:
            self.cv = data['internal_cv_score']
            self.best = struc_str    

    def update(self, pipe_dict):
        [self.add(p,v) for p,v in pipe_dict.items()]
    
    def has_pipe(self, pipe_str):
        struc_str = u.string_to_bracket(pipe_str)
        
        if struc_str not in self.structures:
            return False
        
        return pipe_str in self.structures[struc_str]
    
    def get_nd_keys(self, size=1, f1_f2=["-cv","n_bo_params"],pareto_only=False,return_ids=False):
                
        f1_mod = -1 if "-" in f1_f2[0] else 1
        f2_mod = -1 if "-" in f1_f2[1] else 1
        
        f1_label = f1_f2[0].strip("-")
        f2_label = f1_f2[1].strip("-")
        
        F = np.array([[f1_mod*getattr(v, f1_label), f1_mod*getattr(v, f2_label)] for v in self.structures.values()])
        
        ids = find_non_dominated(F) if pareto_only else np.concatenate(NonDominatedSorting().do(F))
        
        key_list = list(self.keys())
        
        if return_ids:
            return [key_list[i] for i in ids[:size]], ids[:size]
        
        return [key_list[i] for i in ids[:size]]