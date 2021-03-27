#!/usr/bin/python3.8

import scipy as sp
import numpy as np
import clustering as clt
import datasets as ds
import math
import sa
import grasp
import genetic
import kmeans
import training

'''
For each problem:
    For each method with best config.:
        Execute method 20 times and get avarage result and avarage elapsed time
    Execute k-means 20 times and get avarage result and avarage elapsed time
    Apply z-score to avarage results
    Rank results
Get standard avarage, standard deviation, corresponding times and rankings of each method
Make statistical test by pairs:
    Paired t
    Wilcoxon
Paired table with results
Return best method in general by avarage result and by avarage ranking
'''

class TestingAnalysis:
    def __init__(self, results : list):
        self._results : list = results
        self._avg_sse : float = 0
        self._avg_elapsed : float = 0
        self._avg_deviation : float = 0
        self._avg_zscore : float = 0
        for result in results:
            self._avg_sse += result.avg_sse
            self._avg_zscore += result.avg_zscore
        self._avg_sse /= len(results)
        self._avg_zscore /= len(results)
        n : int = 0
        for result in results:
            for sse in result.sse_list:
                self._avg_deviation += (sse - self._avg_sse) ** 2
                n += 1
        self._avg_deviation /= n
        self._avg_deviation = math.sqrt(self._avg_deviation)
    
    @property
    def avg_sse(self) -> float:
        return self._avg_sse
    
    @property
    def avg_elapsed(self) -> float:
        return self._avg_elapsed
    
    @property
    def avg_deviation(self) -> float:
        return self._avg_deviation

    @property
    def testing_results(self) -> list:
        return self._results

class TestingResults:
    def __init__(self, sse_list : list, elapsed_list : list, avg_sse : float, avg_elapsed : float, zscores : np.array):
        self._sse_list = sse_list
        self._elapsed_list = elapsed_list
        self._avg_sse = avg_sse
        self._avg_elapsed = avg_elapsed
        self._zscores = zscores
    
    @property
    def sse_list(self) -> list:
        return self._sse_list

    @property
    def avg_sse(self) -> float:
        return self._avg_sse
    
    @property
    def avg_elapsed(self) -> float:
        return self.avg_elapsed
    
    @property
    def zscores(self) -> float:
        return self._zscores
    
    @property
    def avg_zscore(self) -> float:
        return np.mean(self._zscores)

def testing(training_result : dict, problems : dict) -> dict:
    result : dict = dict({})

    for problem_name, problem in problems:
        
        result[problem_name] = dict({})

        # Testing Simulated Annealing:
        sa_results_list = list([])
        for problem_config in problem['testing']['sa']:
            sa_results : TestingResults = test_sa(training_result[problem_name]['sa'], problem_config.k, problem_config.data_set)
            sa_results_list.append(sa_results)
        result[problem_name]['sa'] = TestingAnalysis(sa_results_list)
        
        # Testing GRASP:
        grasp_results_list = list([])
        for problem_config in problem['testing']['grasp']:
            grasp_results : TestingResults = test_sa(training_result[problem_name]['grasp'], problem_config.k, problem_config.data_set)
            grasp_results_list.append(grasp_results)
        result[problem_name]['grasp'] = TestingAnalysis(grasp_results_list)
        
        # Testing Genetic Algorithm:
        genetic_results_list = list([])
        for problem_config in problem['testing']['genetic']:
            genetic_results : TestingResults = test_sa(training_result[problem_name]['genetic'], problem_config.k, problem_config.data_set)
            genetic_results_list.append(genetic_results)
        result[problem_name]['genetic'] = TestingAnalysis(genetic_results_list)
        
        # Testing K-Means:
        kmeans_results_list = list([])
        for problem_config in problem['testing']['kmeans']:
            kmeans_results : TestingResults = test_kmeans(problem_config.k, problem_config.data_set)
            kmeans_results_list.append(kmeans_results)
        result[problem_name]['kmeans'] = TestingAnalysis(kmeans_results_list)

    return result

def test_sa(training : training.TrainingResult, k : int, data_set : np.array) -> TestingResults:
    best_config_result = training.best_five_config_sse[0]

    print("Testing the Simulated Annealing metaheuristic with k =", k)

    sse_list : list = list([])
    elapsed_list : list = list([])
    avg_sse : float = 0
    avg_elapsed : float = 0

    best_config = training.best_five_config_sse[0]

    for _ in range(20):
        (result, elapsed) = sa.simulated_annealing(best_config, clt.Clusters(k, data_set))
        avg_sse += result
        avg_elapsed += elapsed
        sse_list.append(result)
        elapsed_list.append(elapsed)
    
    avg_sse /= 20
    avg_elapsed /= 20
    z_scores : np.array = sp.stats.zscore(sse_list)

    return TestingResults(sse_list, elapsed_list, avg_sse, avg_elapsed, z_scores)

def test_grasp(training : training.TrainingResult, k : int, data_set : np.array) -> TestingResults:
    best_config_result = training.best_five_config_sse[0]

    print("Testing the GRASP metaheuristic with k =", k)

    sse_list : list = list([])
    elapsed_list : list = list([])
    avg_sse : float = 0
    avg_elapsed : float = 0

    best_config = training.best_five_config_sse[0]

    for _ in range(20):
        (result, elapsed) = grasp.grasp(best_config, clt.Clusters(k, data_set))
        avg_sse += result
        avg_elapsed += elapsed
        sse_list.append(result)
        elapsed_list.append(elapsed)
    
    avg_sse /= 20
    avg_elapsed /= 20
    z_scores : np.array = sp.stats.zscore(sse_list)

    return TestingResults(sse_list, elapsed_list, avg_sse, avg_elapsed, z_scores)

def test_genetic(training : training.TrainingResult, k : int, data_set : np.array) -> TestingResults:
    best_config_result = training.best_five_config_sse[0]

    print("Testing the Genetic Algorithm metaheuristic k =", k)

    sse_list : list = list([])
    elapsed_list : list = list([])
    avg_sse : float = 0
    avg_elapsed : float = 0

    best_config = training.best_five_config_sse[0]

    for _ in range(20):
        (result, elapsed) = genetic.genetic(best_config, clt.Clusters(k, data_set))
        avg_sse += result
        avg_elapsed += elapsed
        sse_list.append(result)
        elapsed_list.append(elapsed)
    
    avg_sse /= 20
    avg_elapsed /= 20
    z_scores : np.array = sp.stats.zscore(sse_list)

    return TestingResults(sse_list, elapsed_list, avg_sse, avg_elapsed, z_scores)

def test_kmeans(k : int, data_set : np.array) -> TestingResults:
    print("Testing the K-Means algorithm with k =", k)

    sse_list : list = list([])
    elapsed_list : list = list([])
    avg_sse : float = 0
    avg_elapsed : float = 0

    for _ in range(20):
        (result, elapsed) = kmeans.kmeans(k, data_set)
        avg_sse += result
        avg_elapsed += elapsed
        sse_list.append(result)
        elapsed_list.append(elapsed)
    
    avg_sse /= 20
    avg_elapsed /= 20
    z_scores : np.array = sp.stats.zscore(sse_list)

    return TestingResults(sse_list, elapsed_list, avg_sse, avg_elapsed, z_scores)
