#!/usr/bin/python3

import numpy as np
import scipy as sp
import clustering as clt
import datasets as ds
import sa
import grasp
import genetic

'''
For each method:
    For each problem:
        For each hyper param. config.:
            Execute metaheuristic 10 times and get avarage and avarege time execution
        Apply z-score to results
        Rank results using z-score
    Obtain avarage result, standard deviation and avarage ranking of each config.
    Obtain best config. by time and by ranking
    Obtain 5 best results and elapsed times of each config. of each method
    Obtain the ranking of each config of each method e it's avarage ranking
Return table with best config by method by mean and by ranking
'''

class TrainingResult:
    def __init__(self, name, avg_sse_list, sse_list, avg_elapsed_list, elapsed_list, z_scores, hyper_param_list):
        self._method_name : str = name
        self._avg_sse_list : list = avg_sse_list.copy()
        self._sse_list : list = sse_list.copy()
        self._avg_elapsed_list : list = avg_elapsed_list.copy()
        self._elapsed_list : list = elapsed_list.copy()
        self._z_scores : np.array = np.copy(z_scores)
        self._hyper_param_list : list = hyper_param_list.copy()
    
    @property
    def method_name(self) -> str:
        return self._method_name
    
    @property
    def avg_sse_list(self) -> list:
        return self._avg_sse_list
    
    @property
    def sse_list(self) -> list:
        return self._sse_list
    
    @property
    def avg_elapsed_list(self) -> list:
        return self._avg_elapsed_list
    
    @property
    def elapsed_list(self) -> list:
        return self._elapsed_list
    
    @property
    def hyper_param_list(self) -> list:
        return self._hyper_param_list

def training(problems : dict) -> dict:
    results : dict = {}

    for problem_name, problem in problems:
        
        sa_hyper_params_list = problem['training']['sa']
        sa_result = train_sa(sa_hyper_params_list)

        grasp_hyper_params_list = problem['training']['grasp']
        genetic_hyper_params_list = problem['training']['genetic']
        kmeans_hyper_params_list = problem['training']['kmeans']



    return results

def train_sa(hyper_param_list: list) -> list:

    avarage_sse_list : list = []
    sse_list : list = []

    avarage_elapsed_list : list = []
    elapsed_list : list = []

    for config in hyper_param_list:
        k : int = config.k
        clusters : clt.Clusters = clt.Clusters(k, config.data_set)
        hyper_params : sa.HyperParams = config.sa_hyper_params
        alpha : float = config.sa_hyper_params.alpha
        temp_func : function = lambda x : x - alpha
        
        avarage_sse : float = 0
        avarage_elapsed : float = 0

        for i in range(0, 9):
            (result, elapsed_time) = sa.simulated_annealing(hyper_params, temp_func, clusters)
            sse_list.add(result.sse)
            elapsed_list.add(elapsed_time)
            avarage_sse += result.sse
            avarage_elapsed += elapsed_time
        
        avarage_sse_list.add(avarage_sse / 10)
        avarage_elapsed_list.add(avarage_elapsed / 10)

    z_scores : np.array = scipy.stats.zscore(avarage_sse_list)

    result : TrainingResult = TrainingResult(avarage_sse_list, sse_list, avarage_elapsed_list, elapsed_list, z_scores, hyper_param_list)

    return result
