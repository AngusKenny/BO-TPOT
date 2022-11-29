# -*- coding: utf-8 -*-
import utils.tpot_utils as u
import numpy as np

class PipeStructure(object):
    pipes = {}
    cv = None
    params = []
    bo_params = []
    operators = []
    structure = ""
    
    def __init__(self, pipe, data) -> None:
        pass

    def add(self, pipe, data) -> None:
        self.pipes[pipe] = data
        self.cv = max(data['internal_cv_score'], self.cv) 
        pass
    
    def remove(self, pipe) -> None:
        
        pass

class StructureList(object):
    
    def __init__(self) -> None:
        pass

