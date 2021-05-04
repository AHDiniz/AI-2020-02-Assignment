#!/usr/bin/python3.8

import scipy as sp
import numpy as np
import clustering as clt
import datasets as ds
import training as tr
import math
import sa
import grasp
import genetic
import kmeans

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

class ProblemTesting:
    def __init__(self, sse_list, elapsed_list, zscore_list, problem_name):
        self._problem_name = problem_name
        self._sse_list = sse_list
        self._elapsed_list = elapsed_list
        self._zscore_list = zscore_list

        self._avg_time = np.mean(elapsed_list)
        self._avg_sse = np.mean(sse_list)

    @property
    def problem_name(self):
        return self._problem_name

    @property
    def sse_list(self):
        return self._sse_list
    
    @property
    def elapsed_list(self):
        return self._elapsed_list
    
    @property
    def zscore_list(self):
        return self._zscore_list
    
    @property
    def avg_zscore(self):
        return np.mean(self._zscore_list)
    
    @property
    def avg_sse(self):
        return self._avg_sse
    
    @property
    def avg_time(self):
        return self._avg_time
    
class MHTesting:
    def __init__(self, method_name, iris_testing_list, wine_testing_list, ionosphere_testing_list):
        self._method_name = method_name
        self._iris_testing_list = iris_testing_list
        self._wine_testing_list = wine_testing_list
        self._ionosphere_testing_list = ionosphere_testing_list

        self._sse_data, self._elapsed_data, self._zscores_data = self.calculate_statistics()

        self._avg_sse_list = list([])
        self._avg_elapsed_list = list([])
        self._avg_zscore_list = list([])

        for test in self._iris_testing_list:
            self._avg_sse_list.append(test.avg_sse)
            self._avg_elapsed_list.append(test.avg_elapsed)
            self._avg_zscore_list.append(test.avg_zscore)
        
        for test in self._wine_testing_list:
            self._avg_sse_list.append(test.avg_sse)
            self._avg_elapsed_list.append(test.avg_elapsed)
            self._avg_zscore_list.append(test.avg_zscore)
        
        for test in self._ionosphere_testing_list:
            self._avg_sse_list.append(test.avg_sse)
            self._avg_elapsed_list.append(test.avg_elapsed)
            self._avg_zscore_list.append(test.avg_zscore)

    @property
    def method_name(self):
        return self._method_name

    @property
    def sse_statistics(self):
        return self._sse_data
    
    @property
    def elapsed_statistics(self):
        return self._elapsed_data
    
    @property
    def zscores_statistics(self):
        return self._zscores_data

    @property
    def avg_sse_list(self):
        return self._avg_sse_list
    
    @property
    def avg_elapsed_list(self):
        return self._avg_elapsed_list
    
    @property
    def avg_zscore_list(self):
        return self._avg_zscore_list

    def calculate_statistics(self):
        sses = list([])
        elapsed = list([])
        zscores = list([])

        for result in self._iris_testing_list:
            sses += result.sse_list
            elapsed += result.elapsed_list
            zscores += result.zscore_list
        
        for result in self._wine_testing_list:
            sses += result.sse_list
            elapsed += result.elapsed_list
            zscores += result.zscore_list
        
        for result in self._ionosphere_testing_list:
            sses += result.sse_list
            elapsed += result.elapsed_list
            zscores += result.zscore_list

        sse_data = (np.mean(sses), np.std(sses))
        elapsed_data = (np.mean(elapsed), np.std(elapsed))
        zscores_data = (np.mean(zscores), np.std(zscores))

        return (sse_data, elapsed_data, zscores_data)

class Testing:
    def __init__(self, sa_testing, grasp_testing, genetic_testing, kmeans_testing):
        self._sa_testing = sa_testing
        self._grasp_testing = grasp_testing
        self._genetic_testing = genetic_testing
        self._kmeans_tesing = kmeans_testing

        self._ranking = [self._sa_testing, self._grasp_testing, self._genetic_testing, self._kmeans_tesing]
        self._ranking.sort(key = lambda x: x.zscores_statistics[0])
    
    @property
    def sa_testing(self):
        return self._sa_testing
    
    @property
    def grasp_testing(self):
        return self._grasp_testing
    
    @property
    def genetic_testing(self):
        return self._genetic_testing
    
    @property
    def kmeans_testing(self):
        return self._kmeans_tesing
    
    @property
    def ranking(self):
        return self._ranking

def testing(t : tr.Training, problems : dict) -> Testing:
    results_lists = dict({})

    for problem_name, problem in problems.items():
        
        sa_results_list = list([])
        grasp_results_list = list([])
        genetic_results_list = list([])
        kmeans_results_list = list([])

        # Testing Simulated Annealing:
        for problem_config in problem['testing']['sa']:
            sa_results : ProblemTesing = test_sa(t, problem_config.k, problem_config.data_set, problem_name)
            sa_results_list.append(sa_results)
        
        # Testing GRASP:
        for problem_config in problem['testing']['grasp']:
            grasp_results : ProblemTesing = test_grasp(t, problem_config.k, problem_config.data_set, problem_name)
            grasp_results_list.append(grasp_results)
        
        # Testing Genetic Algorithm:
        for problem_config in problem['testing']['genetic']:
            genetic_results : ProblemTesing = test_genetic(t, problem_config.k, problem_config.data_set, problem_name)
            genetic_results_list.append(genetic_results)
        
        # Testing K-Means:
        for problem_config in problem['testing']['kmeans']:
            kmeans_results : ProblemTesing = test_kmeans(problem_config.k, problem_config.data_set, problem_name)
            kmeans_results_list.append(kmeans_results)

        results_lists[problem_name] = {'sa': sa_results_list, 'grasp': grasp_results_list, 'genetic': genetic_results_list, 'kmeans': kmeans_results_list}

    sa_testing = MHTesting(results_lists['iris']['sa'], results_lists['wine']['sa'], results_lists['ionosphere']['sa'])
    grasp_testing = MHTesting(results_lists['iris']['grasp'], results_lists['wine']['grasp'], results_lists['ionosphere']['grasp'])
    genetic_testing = MHTesting(results_lists['iris']['genetic'], results_lists['wine']['genetic'], results_lists['ionosphere']['genetic'])
    kmeans_testing = MHTesting(results_lists['iris']['kmeans'], results_lists['wine']['kmeans'], results_lists['ionosphere']['kmeans'])

    return Testing(sa_testing, grasp_testing, genetic_testing, kmeans_testing)

def test_sa(training : tr.Training, k : int, data_set : np.array, problem_name : str) -> ProblemTesting:

    sse_list : list = list([])
    elapsed_list : list = list([])
    avg_sse : float = 0
    avg_elapsed : float = 0

    best_config = training.sa_training.best_config_zscore

    for _ in range(20):
        (result, elapsed) = sa.simulated_annealing(best_config, clt.Clusters(k, data_set))
        avg_sse += result
        avg_elapsed += elapsed
        sse_list.append(result)
        elapsed_list.append(elapsed)
    
    avg_sse /= 20
    avg_elapsed /= 20
    z_scores : np.array = sp.stats.zscore(sse_list)

    return ProblemTesting(sse_list, elapsed_list, z_scores, problem_name)

def test_grasp(training : tr.Training, k : int, data_set : np.array, problem_name : str) -> ProblemTesting:

    sse_list : list = list([])
    elapsed_list : list = list([])
    avg_sse : float = 0
    avg_elapsed : float = 0

    best_config = training.grasp_training.best_config_zscore

    for _ in range(20):
        (result, elapsed) = grasp.grasp(best_config, clt.Clusters(k, data_set))
        avg_sse += result
        avg_elapsed += elapsed
        sse_list.append(result)
        elapsed_list.append(elapsed)
    
    avg_sse /= 20
    avg_elapsed /= 20
    z_scores : np.array = sp.stats.zscore(sse_list)

    return ProblemTesting(sse_list, elapsed_list, z_scores, problem_name)

def test_genetic(training : tr.Training, k : int, data_set : np.array, problem_name : str) -> ProblemTesting:

    sse_list : list = list([])
    elapsed_list : list = list([])
    avg_sse : float = 0
    avg_elapsed : float = 0

    best_config = training.genetic_training.best_config_zscore

    for i in range(20):
        (result, elapsed) = genetic.genetic(best_config, clt.Clusters(k, data_set))
        avg_sse += result
        avg_elapsed += elapsed
        sse_list.append(result)
        elapsed_list.append(elapsed)
    
    avg_sse /= 20
    avg_elapsed /= 20
    z_scores : np.array = sp.stats.zscore(sse_list)

    return ProblemTesting(sse_list, elapsed_list, z_scores, problem_name)

def test_kmeans(k : int, data_set : np.array, problem_name : str) -> ProblemTesting:

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

    return ProblemTesting(sse_list, elapsed_list, z_scores, problem_name)
