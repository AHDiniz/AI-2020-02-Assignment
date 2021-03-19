#!/usr/bin/python3

import math
import time
import random
import clustering as clt
import numpy as np

# Class that defines the Simulated Annealing metaheuristic hyper parameters:
class HyperParams:
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

# Metaheuristic implementation:
def simulated_annealing(hyper_params : HyperParams, clusters : clt.Clusters) -> (clt.Clusters, float):

    current_temp : float = hyper_params.init_temp
    
    init_time : float = time.time()
    elapsed_time : float = time.time()

    clusters.initialize_state()

    while current_temp > hyper_params.final_temp:

        iterations : int = 0
        while iterations < hyper_params.num_iter:
            current_cost : float = clusters.sse
            
            # Disturbing the state:
            disturbed : clt.Clusters = clusters.disturb()
            disturbed_cost : float = disturbed.sse

            # Comparing the costs of the current state and the disturbed state:
            delta : float = disturbed_cost - current_cost

            if delta < 0 or random.random() >= math.exp(-delta / current_temp):
                clusters = disturbed

            iterations += 1
            elapsed_time = time.time()

            if elapsed_time - init_time >= 1:
                break

        # Updating the temperature:
        current_temp *= hyper_params.alpha
        print("Current temperature =", current_temp)

        elapsed_time = time.time()

        if elapsed_time - init_time >= 1:
            break
    return (clusters, elapsed_time - init_time)
