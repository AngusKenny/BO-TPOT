# -*- coding: utf-8 -*-
from config.tpot_config import default_tpot_config_dict
import utils.tpot_utils as u
import numpy as np

class PipeStructure(object):
    pipes = {}
    cv = -1e20
    params = []
    bo_params = []
    operators = []
    structure = ""
    best = ""
    
    def __init__(self, pipe, data, config_dict=None) -> None:
        self.best = pipe
        self.structure = u.string_to_bracket(pipe)
        self.params = u.string_to_params(pipe)
        self.bo_params = u.string_to_params(pipe,config_dict=config_dict)
        self.operators = u.string_to_ops(pipe)
        self.add(pipe,data)
        
    def __getitem__(self, key):
        return self.pipes[key]
  
    def __setitem__(self, key, newvalue):
        self.pipes.add(key, newvalue)

    def __contains__(self, key):
        return key in self.pipes

    def add(self, pipe, data) -> None:
        data['structure'] = self.structure
        self.pipes[pipe] = data
        if data['internal_cv_score'] > self.cv:
            self.cv = data['internal_cv_score']
            self.best = pipe
    
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
    structures = {}
    config_dict = None
    
    def __init__(self, config_dict=None) -> None:
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

    def index(self, struc_str):
        return list(self.structures.keys()).index(struc_str)

    def add(self, pipe_str, data):
        '''returns 1 if new group created'''
        struc_str = u.string_to_bracket(pipe_str)
        if struc_str not in self.structures:
            self.structures[struc_str] = PipeStructure(pipe_str,data,self.config_dict)
            return 1
        else:
            self.structures[struc_str].add(pipe_str,data)
            return 0
            
    def has_pipe(self, pipe_str):
        struc_str = u.string_to_bracket(pipe_str)
        
        if struc_str not in self.structures:
            return False
        
        return pipe_str in self.structures[struc_str]