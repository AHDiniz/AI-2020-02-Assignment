#!/usr/bin/python3

import time
import numpy as np
import clustering as clt
import kmeans

class HyperParams:
    def __init__(self, num_iter : int = 0, num_best_solutions : int = 0):
        self._num_iter : int = num_iter
        self._num_best_solutions : int = num_best_solutions
    
    @property
    def num_iter(self) -> int:
        return num_iter
    
    @property
    def num_best_solutions(self) -> int:
        return num_best_solutions

def grasp(hyper_params : HyperParams, clusters : clt.Clusters) -> (clt.Clusters, float):
    
    
    return clusters
