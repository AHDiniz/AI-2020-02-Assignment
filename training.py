#!/usr/bin/python3.8

import heapq
import numpy as np
import scipy as sp
import clustering as clt
import datasets as ds

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
    Obtain the ranking of each config for each method and it's avarage ranking
Return table with best config for each method by mean and by ranking
'''

class TrainingResult:
    def __init__(self, training_data_list : list):

        self.__training_data_list = training_data_list

        max_data_time : TrainingData = max(training_data_list, key = lambda item : item.avg_time)
        max_data_result : TrainingData = max(training_data_list, key = lambda item : item.avg_result)

        self.__best_config_time = (max_data_time.config, max_data_time.avg_time)
        self.__best_config_zscore = (max_data_result.config, max_data_result.avg_zscore)

        self.__five_best_times = [(x.config, x.avg_time) for x in heapq.nsmallest(5, training_data_list, key = lambda item : item.avg_time)]
        self.__five_best_results = [(x.config, x.avg_result) for x in heapq.nsmallest(5, training_data_list, key = lambda item : item.avg_result)]
    
    @property
    def best_config_time(self):
        return self.__best_config_time
    
    @property
    def best_config_zscore(self):
        return self.__best_config_zscore
    
    @property
    def five_best_configs_time(self):
        return self.__five_best_times
    
    @property
    def five_best_configs_result(self):
        return self.__five_best_results
    
    @property
    def training_data_list(self):
        return self.__training_data_list

class TrainingData:
    def __init__(self, method_name, dataset_name, config, avg_results_list, avg_time_list, zscore_list, results, times):
        self.__method_name = method_name
        self.__dataset_name = dataset_name
        self.__config = config
        self.__avg_results_list = avg_results_list
        self.__avg_time_list = avg_time_list
        self.__zscore_list = zscore_list
        self.__results = results
        self.__times = times

        self.__avg_result = np.mean(avg_results_list)
        self.__avg_time = np.mean(avg_time_list)
        self.__avg_zscore = np.mean(zscore_list)
    
    @property
    def method_name(self):
        return self.__method_name
    
    @property
    def dataset_name(self):
        return self.__dataset_name
    
    @property
    def config(self):
        return self.__config
    
    @property
    def avg_results_list(self):
        return self.__avg_results_list
    
    @property
    def avg_time_list(self):
        return self.__avg_time_list
    
    @property
    def zscore_list(self):
        return self.__zscore_list
    
    @property
    def avg_result(self):
        return self.__avg_result
    
    @property
    def avg_time(self):
        return self.__avg_time
    
    @property
    def avg_zscore(self):
        return self.__avg_zscore
    
    @property
    def results(self):
        return self.__results
    
    @property
    def times(self):
        return self.__times

def training(meta_heuristic, dataset : ds.Dataset, hyper_param_list : list, method_name : str, dataset_name : str) -> TrainingResult:
    data = []
    for config in hyper_param_list:
        avg_results_list = []
        avg_time_list = []
        zscore_list = []
        results = []
        times = []
        for k in dataset.training_ks:
            results_list = []
            time_list = []
            for _ in range(10):
                clusters = clt.Clusters(k, dataset.points_data)
                result, time = meta_heuristic(config, clusters)
                results_list.append(result)
                results.append(result)
                time_list.append(time)
                times.append(time)
            avg_results_list.append(np.mean(results_list))
            avg_time_list.append(np.mean(time_list))
            zscore_list.append(sp.stats.zscore(results_list))
        data.append(TrainingData(method_name, dataset_name, config, avg_results_list, avg_time_list, zscore_list, results, times))
    return TrainingResult(data)
