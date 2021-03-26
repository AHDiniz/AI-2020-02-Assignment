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
    def __init__(self, name, avg_sse_list, sse_list, avg_elapsed_list, elapsed_list, z_scores, hyper_param_list, std_list):
        self._method_name : str = name
        self._avg_sse_list : list = avg_sse_list.copy()
        self._sse_list : list = sse_list.copy()
        self._avg_elapsed_list : list = avg_elapsed_list.copy()
        self._elapsed_list : list = elapsed_list.copy()
        self._z_scores : np.array = np.copy(z_scores)
        self._hyper_param_list : list = hyper_param_list.copy()
        self._std_list : list = std_list.copy()
        self._population_list : list = None
    
    # The population result of the genetic algorithm implementation:
    def set_population_list(self, population_list : list):
        self._population_list = population_list
    def get_population_list(self) -> list:
        return self._population_list

    # Name of the method used as metaheuristic:
    @property
    def method_name(self) -> str:
        return self._method_name
    
    # List of each configuration avarage sse:
    @property
    def avg_sse_list(self) -> list:
        return self._avg_sse_list
    
    # List of sse's:
    @property
    def sse_list(self) -> list:
        return self._sse_list
    
    # List of each configuration avarage elapsed time:
    @property
    def avg_elapsed_list(self) -> list:
        return self._avg_elapsed_list
    
    # List of elapsed times:
    @property
    def elapsed_list(self) -> list:
        return self._elapsed_list
    
    # Returning the list of hyperparameter configurations:
    @property
    def hyper_param_list(self):
        return self._hyper_param_list
    
    # Returning hyperparameter configuration with the best avarage elapsed time:
    @property
    def best_config_time(self):
        return hyper_param_list[self._avg_elapsed_list.index(min(self._avg_elapsed_list))]
    
    # Returing hyperparameter configuration with the best zscore:
    @property
    def best_config_zscore(self):
        return hyper_param_list[np.where(self._z_scores == np.amin(self._z_scores))]
    
    # Returning the metaheuristic's avarage score:
    @property
    def avarage_zscore(self):
        return np.mean(self._z_scores)

    # Getting the five best configurations regarding avarage elapsed time:
    @property
    def best_five_config_time(self) -> list:
        best_configs : list = []
        best_indeces : list = []

        for i in range(5):
            min_time : float = np.inf
            min_index : int = 0
            config = None

            for j in range(len(hyper_param_list)):
                if self._avg_elapsed_list[j] < min_time and not j in best_indeces:
                    min_index = j
                    min_time = self._avg_elapsed_list[j]
                    config = hyper_param_list[j]
            
            best_configs.append(config)
            best_indeces.append(min_index)
        
        return best_configs
    
    # Getting the five best configurations regarding avarage sse:
    @property
    def best_five_config_sse(self) -> list:
        best_configs : list = []
        best_indeces : list = []

        for i in range(5):
            min_sse : float = np.inf
            min_index : int = 0
            config = None

            for j in range(len(hyper_param_list)):
                if self._avg_sse_list[j] < min_sse and not j in best_indeces:
                    min_index = j
                    min_sse = self._avg_sse_list[j]
                    config = hyper_param_list[j]
            
            best_configs.append(config)
            best_indeces.append(min_index)
        
        return best_configs
    
    population_list = property(get_population_list, set_population_list)

def training(problems : dict) -> dict:
    results : dict = {}

    for problem_name, problem in problems.items():

        print("Starting the training procedure with the", problem_name, "dataset.")
        
        sa_hyper_params_list = problem['training']['sa']
        sa_result = train_sa(sa_hyper_params_list)

        grasp_hyper_params_list = problem['training']['grasp']
        grasp_result = train_grasp(grasp_hyper_params_list)

        genetic_hyper_params_list = problem['training']['genetic']
        genetic_result = train_genetic(genetic_hyper_params_list)

        results[problem_name] = {'sa': sa_result, 'grasp': grasp_result, 'genetic': genetic_result}

    return results

def train_sa(hyper_param_list: list) -> TrainingResult:

    print("Training the Simulated Annealing metaheuristic...")

    avarage_sse_list : list = []
    sse_lists : list = []
    std_list : list = []

    avarage_elapsed_list : list = []
    elapsed_list : list = []

    for config in hyper_param_list:
        k : int = config.k
        hyper_params : sa.HyperParams = config.sa_hyper_params
        
        avarage_sse : float = 0
        avarage_elapsed : float = 0
        sse_list : list = []

        for i in range(10):
            clusters : clt.Clusters = clt.Clusters(k, config.data_set)
            (result, elapsed_time) = sa.simulated_annealing(hyper_params, clusters)
            sse_list.append(result)
            elapsed_list.append(elapsed_time)
            avarage_sse += result
            avarage_elapsed += elapsed_time
        
        std_list.append(np.std(sse_list))
        
        sse_lists.append(sse_list)
        avarage_sse_list.append(avarage_sse / 10)
        avarage_elapsed_list.append(avarage_elapsed / 10)

        print("Avarage result =", avarage_sse / 10)

    z_scores : np.array = sp.stats.zscore(avarage_sse_list)

    result : TrainingResult = TrainingResult("Simulated Annealing", avarage_sse_list, sse_list, avarage_elapsed_list, elapsed_list, z_scores, hyper_param_list, std_list)

    return result

def train_grasp(hyper_param_list : list) -> TrainingResult:

    print("Training the GRASP metaheuristic...")

    avarage_sse_list : list = []
    sse_lists : list = []
    std_list : list = []

    avarage_elapsed_list : list = []
    elapsed_list : list = []

    for config in hyper_param_list:
        k : int = config.k
        hyper_params : grasp.HyperParams = config.grasp_hyper_params
        
        avarage_sse : float = 0
        avarage_elapsed : float = 0
        sse_list : list = []

        for i in range(10):
            clusters : clt.Clusters = clt.Clusters(k, config.data_set)
            (result, elapsed_time) = grasp.grasp(hyper_params, clusters)
            sse_list.append(result)
            elapsed_list.append(elapsed_time)
            avarage_sse += result
            avarage_elapsed += elapsed_time
        
        std_list.append(np.std(sse_list))
        
        print("Avarage result =", avarage_sse / 10)

        sse_lists.append(sse_list)
        avarage_sse_list.append(avarage_sse / 10)
        avarage_elapsed_list.append(avarage_elapsed / 10)

    z_scores : np.array = sp.stats.zscore(avarage_sse_list)

    result : TrainingResult = TrainingResult("GRASP", avarage_sse_list, sse_list, avarage_elapsed_list, elapsed_list, z_scores, hyper_param_list, std_list)

    return result

def train_genetic(hyper_param_list : list) -> TrainingResult:

    print("Training the Genetic Algorithm metaheuristic...")

    avarage_sse_list : list = []
    sse_lists : list = []
    std_list : list = []

    avarage_elapsed_list : list = []
    elapsed_list : list = []
    population_list : list = []

    for config in hyper_param_list:
        k : int = config.k
        clusters : clt.Clusters = clt.Clusters(k, config.data_set)
        hyper_params : genetic.HyperParams = config.genetic_hyper_params
        
        avarage_sse : float = 0
        avarage_elapsed : float = 0
        sse_list : list = []

        population : genetic.Population = genetic.Population(k, hyper_params.population_size, clusters)
        population_list.append(population)
        
        for i in range(10):
            (result, elapsed_time) = genetic.genetic(hyper_params, clusters, population)
            sse_list.append(result)
            elapsed_list.append(elapsed_time)
            avarage_sse += result
            avarage_elapsed += elapsed_time
        
        std_list.append(np.std(sse_list))
        
        print("Avarage result =", avarage_sse / 10)

        sse_lists.append(sse_list)
        avarage_sse_list.append(avarage_sse / 10)
        avarage_elapsed_list.append(avarage_elapsed / 10)

    z_scores : np.array = sp.stats.zscore(avarage_sse_list)

    result : TrainingResult = TrainingResult("Genetic Algorithm", avarage_sse_list, sse_list, avarage_elapsed_list, elapsed_list, z_scores, hyper_param_list, std_list)
    result.population_list = population_list

    return result
