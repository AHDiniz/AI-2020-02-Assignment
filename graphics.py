#!/usr/bin/python3.8

from copy import copy
import seaborn as sns
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import training as tr
import testing as ts
import sa
import grasp
import genetic

# Training graphics:

# Five best configurations per method:
def training_five_best_configuration(training_results : dict):
    best_configs_time = {'SA': [], 'GRASP': [], 'Genetic': []}
    best_configs_result = {'SA': [], 'GRASP': [], 'Genetic': []}
    
    for dataset_name, methods in training_results:
        for method_name, method_result in methods:
            for config in method_result.five_best_configs_time:
                if len(best_configs_time[method_name]) < 5:
                    best_configs_time.append(config)
                else:
                    not_best_config = min(best_configs_time[method_name], key = lambda x : x[1])
                    if config[1] < not_best_config[1]:
                        best_configs_time[method_name].remove(not_best_config)
                        best_configs_time[method_name].append(config)
            
            for config in method_result.five_best_configs_result:
                if len(best_configs_result[method_name]) < 5:
                    best_configs_result.append(config)
                else:
                    not_best_config = min(best_configs_result[method_name], key = lambda x : x[1])
                    if config[1] < not_best_config[1]:
                        best_configs_result[method_name].remove(not_best_config)
                        best_configs_result[method_name].append(config)
    
    print("Five best hyperparameter configurations by time:")
    for method_name, config_list in best_configs_time:
        print(method_name)
        for config, avg_time in config_list:
            if method_name == 'SA':
                print("Avarage Time =", avg_time, ", Configuration = ( Initial Temperature =", config.init_temp, ", Number of Iterations =", config.num_iter, ", Alpha =", config.alpha, ")")
            elif method_name == 'GRASP':
                print("Avarage Time =", avg_time, ", Configuration = ( Number of Iterations =", config.num_iter, ", Number of Best Solutions =", config.num_best_solutions, ")")
            elif method_name == 'Genetic':
                print("Avarage Time =", avg_time, ", Configuration = ( Population Size =", config.population_size, ", Crossover Rate =", config.crossover_rate, "Mutation Rate =", config.mutation_rate, ")")
    
    print("Five best hyperparameter configurations by result:")
    for method_name, config_list in best_configs_result:
        print(method_name)
        for config, avg_result in config_list:
            if method_name == 'SA':
                print("Avarage Time =", avg_time, ", Configuration = ( Initial Temperature =", config.init_temp, ", Number of Iterations =", config.num_iter, ", Alpha =", config.alpha, ")")
            elif method_name == 'GRASP':
                print("Avarage Time =", avg_time, ", Configuration = ( Number of Iterations =", config.num_iter, ", Number of Best Solutions =", config.num_best_solutions, ")")
            elif method_name == 'Genetic':
                print("Avarage Time =", avg_time, ", Configuration = ( Population Size =", config.population_size, ", Crossover Rate =", config.crossover_rate, "Mutation Rate =", config.mutation_rate, ")")

# Results boxplots per method:
def training_results_boxplots(training_results : dict):
    results = {'SA': {'Iris': None, 'Wine': None}, 'GRASP': {'Iris': None, 'Wine': None}, 'Genetic': {'Iris': None, 'Wine': None}}

    

# Times boxplots per method:
def training_times_boxplots(training_results : dict):
    pass

# Avarage ranking of each method in each dataset:
def training_avarage_ranking(training_results : dict):
    pass

# Best configuration of each method by time:
def best_config_by_time(training_results : dict):
    pass

# Best configuration of each method by avarage ranking:
def best_config_by_result(training_results : dict):
    pass

# Testing graphics:

# Standard avarage, standard deviation and mean of elapsed times of each method for each dataset:
def testing_statistical_data(testing_result : ts.TestingResults):
    pass

# Boxplot of each method for elapsed times:
def testing_times_boxplots(testing_result : ts.TestingResults):
    pass

# Boxplot of each method for results:
def testing_results_boxplots(testing_result : ts.TestingResults):
    pass

# Table ranking each method for each dataset:
def testing_ranking_methods_by_dataset(testing_result : ts.TestingResults):
    pass

# Avarage ranking of each problem:
def testing_problem_ranking(testing_result : ts.TestingResults):
    pass

# Paired table of each method (t-paired and wilcoxon):
def testing_paired_tests(testing_result : ts.TestingResults):
    pass

# Best method by avarage result:
def testing_best_method_result(testing_result : ts.TestingResults):
    pass

# Best method by avarage elapsed time:  
def testing_best_method_time(testing_result : ts.TestingResults):
    pass
