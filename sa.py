#!/usr/bin/python3

import clustering as clt
import numpy as np

class SAHyperParams:
    def __init__(self, init_temp : float, final_temp : float, num_iter: int, alpha : int):
        self._init_temp : float = init_temp
        self._final_temp : float = final_temp
        self._num_iter : float = num_iter
        self._alpha : int = alpha

    @property
    def init_temp(self) -> float:
        return self._init_temp
    
    @property
    def final_temp(self) -> float:
        return self._final_temp
    
    @property
    def num_iter(self) -> int:
        return self._num_iter
    
    @property
    def alpha(self) -> int:
        return self._alpha

def simulated_annealing(hyper_params : SAHyperParams, temp_func : function, clusters : clt.Clusters):
    current_temp : float = hyper_params.init_temp
    iterations : int = 0

    while current_temp > hyper_params.final_temp and iterations < hyper_params.num_iter:
        current_cost = clusters.sse
        
        
