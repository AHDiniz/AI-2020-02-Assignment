#!/usr/bin/python3.8

import scipy as sp
import numpy as np
import clustering as clt
import datasets as ds
import training as tr
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

class TestingResults:
    def __init__(self, results, times, method_name, dataset_name):
        self.__results = results
        self.__times = times
        self.__method_name = method_name
        self.__dataset_name = dataset_name

        self.__avg_result = np.mean(results)
        self.__avg_time = np.mean(times)
        self.__zscores = sp.stats.zscore(results)
        self.__standard = np.std(results)
    
    @property
    def method_name(self):
        return self.__method_name
    
    @property
    def dataset_name(self):
        return self.__dataset_name
    
    @property
    def avg_result(self):
        return self.__avg_result
    
    @property
    def avg_time(self):
        return self.__avg_time
    
    @property
    def zscores(self):
        return self.__zscores
    
    @property
    def standard_deviation(self):
        return self.__standard

def testing(meta_heuristic, dataset : ds.Dataset, training_result : tr.TrainingResult, method_name : str, dataset_name : str) -> TestingResults:
    results = []
    times = []
    for k in dataset.testing_ks:
        config = training_result.best_config_zscore
        for _ in range(20):
            clusters = clt.Clusters(k, dataset.points_data)
            result, time = meta_heuristic(config, clusters)
            result.append(result)
            times.append(time)
    return TestingResults(results, times, method_name, dataset_name)

def test_kmeans(dataset : ds.Dataset, dataset_name : str) -> TestingResults:
    results = []
    times = []
    for k in dataset.testing_ks:
        result, time = kmeans.kmeans(k, dataset.points_data)
        results.append(result)
        times.append(time)
    return TestingResults(results, times, "k-Means", dataset_name)
