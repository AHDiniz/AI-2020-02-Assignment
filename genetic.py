#!/usr/bin/python3

import time
import numpy as np
import clustering as clt

class HyperParams:
    def __init__(self, population_size : int, crossover_rate : float, mutation_rate : float):
        self._population_size = population_size
        self._crossover_rate = crossover_rate
        self._mutation_rate = mutation_rate
    
    @property
    def population_size(self) -> int:
        return self._population_size
    
    @property
    def crossover_rate(self) -> float:
        return self._crossover_rate
    
    @property
    def mutation_rate(self) -> float:
        return self._mutation_rate

class Population:
    def __init__(self, size : int, clusters : clt.Clusters):
        self._size = size
        self._instances : list = []
        self._clusters : clt.Clusters = clusters

def genetic(hyper_params : HyperParams, clusters : clt.Clusters) -> (clt.Clusters, float):
    return (clusters, 0)