#!/usr/bin/python3

import time
import numpy as np
import clustering as clt

class HyperParams:
    def __init__(self, num_iter : int = 0, num_best_solutions : int = 0):
        self._num_iter : int = num_iter
        self._num_best_solutions : int = num_best_solutions
    
    @property
    def num_iter(self) -> int:
        return self._num_iter
    
    @property
    def num_best_solutions(self) -> int:
        return self._num_best_solutions

def hill_climbing(clusters : clt.Clusters) -> float:

    init_time : float = time.time()
    current_time : float = 0

    while current_time - init_time < 0.1:
        clusters.disturb()
        if clusters.disturbed_sse < clusters.sse:
            clusters.accept_disturbed()
        current_time = time.time()
    
    return clusters.sse

def grasp(hyper_params : HyperParams, clusters : clt.Clusters) -> (float, float):
    
    init_time : float = time.time()
    current_time : float = 0

    clusters.initialize_state()

    result : float = 0
    best_solutions : set = set([])
    solutions_added : int = 0

    iterations : int = 0

    while iterations < hyper_params.num_iter:
        
        # Use local search to create a new solution:
        local_search = hill_climbing(clusters)
        # If can fit in the best solutions, push to it to list of best solutions
        if solutions_added < hyper_params.num_best_solutions:
            best_solutions.add(local_search)
        else:
            less_best : float = max(best_solutions)
            if local_search < less_best:
                best_solutions.remove(less_best)
                best_solutions.add(local_search)
        # Set current state to best state
        result = min(best_solutions)

        current_time = time.time()
        if current_time - init_time >= 1:
            break
        iterations += 1

    return (result, current_time - init_time)
