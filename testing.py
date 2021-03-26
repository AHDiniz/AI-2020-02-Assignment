#!/usr/bin/python3

import scipy
import clustering as clt
import datasets as ds
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
    def __init__(self, sa_results : list, grasp_results : list, genetic_results : list, kmeans_results : list):
        self._sa_results : list = sa_results
        self._grasp_results : list = grasp_results
        self._genetic_results : list = genetic_results
        self._kmeans_results : list = kmeans_results
    


class TestingResults:
    def __init__(self, sse_list, elapsed_list, avg_sse, avg_elapsed):
        self._sse_list = sse_list
        self._elapsed_list = elapsed_list
        self._avg_sse = avg_sse
        self._avg_elapsed = avg_elapsed
    
    @property
    def sse_list(self):
        return self._sse_list
    
    @property
    def elapsed_list(self):
        return self._elapsed_list
    
    @property
    def avg_sse(self):
        return self._avg_sse
    
    @property
    def avg_elapsed(self):
        return self._avg_elapsed

def testing(training_result : dict, problems : dict) -> dict:
    result : dict = dict({})

    for problem_name, problem in problems:
        
        result[problem_name] = dict({})

        # Testing Simulated Annealing:
        result[problem_name]['sa'] = list([])
        for problem_config in problem['testing']['sa']:
            sa_results : TestingResults = test_sa(training_result[problem_name]['sa'], problem_config.k, problem_config.data_set)
            result[problem_name]['sa'].append(sa_results)
        
        # Testing GRASP:
        result[problem_name]['grasp'] = list([])
        for problem_config in problem['testing']['grasp']:
            grasp_results : TestingResults = test_sa(training_result[problem_name]['grasp'], problem_config.k, problem_config.data_set)
            result[problem_name]['sa'].append(grasp_results)
        
        # Testing Genetic Algorithm:
        result[problem_name]['genetic'] = list([])
        for problem_config in problem['testing']['genetic']:
            genetic_results : TestingResults = test_sa(training_result[problem_name]['genetic'], problem_config.k, problem_config.data_set)
            result[problem_name]['sa'].append(genetic_results)

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

    return TestingResults(sse_list, elapsed_list, avg_sse, avg_elapsed)

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

    return TestingResults(sse_list, elapsed_list, avg_sse, avg_elapsed)

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

    return TestingResults(sse_list, elapsed_list, avg_sse, avg_elapsed)

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

    return TestingResults(sse_list, elapsed_list, avg_sse, avg_elapsed)
