#!/usr/bin/python3.8

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
    Obtain the ranking of each config for each method and it's avarage ranking
Return table with best config for each method by mean and by ranking
'''

# The data acquired by analysing the training results:
class TrainingAnalysis:
    def __init__(self, sa_results_list : list, grasp_results_list : list, genetic_results_list : list):
        self._sa_results_list : list = sa_results_list
        self._grasp_results_list : list = grasp_results_list
        self._genetic_results_list : list = genetic_results_list
    
    @property
    def sa_results_list(self) -> list:
        return self._sa_results_list
    
    @property
    def grasp_results_list(self) -> list:
        return self._grasp_results_list
    
    @property
    def genentic_results_list(self) -> list:
        return self._genetic_results_list

    @property
    def five_best_config_sse(self) -> (list, list, list):
        five_best_sa : list = []
        five_best_grasp : list = []
        five_best_genetic : list = []
        
        # Getting the five best hyperparameter configuration for Simulated Annealing:
        for sa_result in self._sa_results_list:
            config_result_tuple : tuple = sa_result.best_config_avg_result
            
            if (len(five_best_sa) < 5):
                five_best_sa.append(config_result_tuple)
            else:
                less_best : tuple = max(five_best_sa, key = (lambda x : x[1]))
                if less_best[1] > config_result_tuple[1]:
                    five_best_sa.remove(less_best)
                    five_best_sa.append(config_result_tuple)
        
        # Getting the five best hyperparameter configuration for GRASP:
        for grasp_result in self._grasp_results_list:
            config_result_tuple : tuple = grasp_result.best_config_avg_result
            
            if (len(five_best_grasp) < 5):
                five_best_grasp.append(config_result_tuple)
            else:
                less_best : tuple = max(five_best_grasp, key = (lambda x : x[1]))
                if less_best[1] > config_result_tuple[1]:
                    five_best_grasp.remove(less_best)
                    five_best_grasp.append(config_result_tuple)

        # Getting the five best hyperparameter configuration for Genetic Algorithm:
        for grasp_result in self._grasp_results_list:
            config_result_tuple : tuple = grasp_result.best_config_avg_result
            
            if (len(five_best_grasp) < 5):
                five_best_grasp.append(config_result_tuple)
            else:
                less_best : tuple = max(five_best_grasp, key = (lambda x : x[1]))
                if less_best[1] > config_result_tuple[1]:
                    five_best_grasp.remove(less_best)
                    five_best_grasp.append(config_result_tuple)

        return (five_best_sa, five_best_grasp, five_best_genetic)
    
    @property
    def five_best_config_time(self) -> (list, list, list):
        five_best_sa : list = []
        five_best_grasp : list = []
        five_best_genetic : list = []
        
        # Getting the five best hyperparameter configuration for Simulated Annealing:
        for sa_result in self._sa_results_list:
            config_result_tuple : tuple = sa_result.best_config_time
            
            if (len(five_best_sa) < 5):
                five_best_sa.append(config_result_tuple)
            else:
                less_best : tuple = max(five_best_sa, key = (lambda x : x[2]))
                if less_best[2] > config_result_tuple[2]:
                    five_best_sa.remove(less_best)
                    five_best_sa.append(config_result_tuple)
        
        # Getting the five best hyperparameter configuration for GRASP:
        for grasp_result in self._grasp_results_list:
            config_result_tuple : tuple = grasp_result.best_config_time
            
            if (len(five_best_grasp) < 5):
                five_best_grasp.append(config_result_tuple)
            else:
                less_best : tuple = max(five_best_grasp, key = (lambda x : x[2]))
                if less_best[2] > config_result_tuple[2]:
                    five_best_grasp.remove(less_best)
                    five_best_grasp.append(config_result_tuple)

        # Getting the five best hyperparameter configuration for Genetic Algorithm:
        for grasp_result in self._grasp_results_list:
            config_result_tuple : tuple = grasp_result.best_config_time
            
            if (len(five_best_grasp) < 5):
                five_best_grasp.append(config_result_tuple)
            else:
                less_best : tuple = max(five_best_grasp, key = (lambda x : x[2]))
                if less_best[2] > config_result_tuple[2]:
                    five_best_grasp.remove(less_best)
                    five_best_grasp.append(config_result_tuple)

        return (five_best_sa, five_best_grasp, five_best_genetic)
    
    @property
    def five_best_config_zscore(self) -> (list, list, list):
        five_best_sa : list = []
        five_best_grasp : list = []
        five_best_genetic : list = []
        
        # Getting the five best hyperparameter configuration for Simulated Annealing:
        for sa_result in self._sa_results_list:
            config_result_tuple : tuple = sa_result.best_config_zscore
            
            if (len(five_best_sa) < 5):
                five_best_sa.append(config_result_tuple)
            else:
                less_best : tuple = max(five_best_sa, key = (lambda x : x[2]))
                if less_best[2] > config_result_tuple[2]:
                    five_best_sa.remove(less_best)
                    five_best_sa.append(config_result_tuple)
        
        # Getting the five best hyperparameter configuration for GRASP:
        for grasp_result in self._grasp_results_list:
            config_result_tuple : tuple = grasp_result.best_config_zscore
            
            if (len(five_best_grasp) < 5):
                five_best_grasp.append(config_result_tuple)
            else:
                less_best : tuple = max(five_best_grasp, key = (lambda x : x[2]))
                if less_best[2] > config_result_tuple[2]:
                    five_best_grasp.remove(less_best)
                    five_best_grasp.append(config_result_tuple)

        # Getting the five best hyperparameter configuration for Genetic Algorithm:
        for grasp_result in self._grasp_results_list:
            config_result_tuple : tuple = grasp_result.best_config_zscore
            
            if (len(five_best_grasp) < 5):
                five_best_grasp.append(config_result_tuple)
            else:
                less_best : tuple = max(five_best_grasp, key = (lambda x : x[2]))
                if less_best[2] > config_result_tuple[2]:
                    five_best_grasp.remove(less_best)
                    five_best_grasp.append(config_result_tuple)

        return (five_best_sa, five_best_grasp, five_best_genetic)
    
    @property
    def best_config_time(self) -> (tuple, tuple, tuple):
        sa_best : tuple = None
        grasp_best : tuple = None
        genetic_best : tuple = None
        
        for best_result in self.five_best_config_time[0]:
            if sa_best == None or best_result[2] < sa_best[2]:
                sa_best = best_result

        for best_result in self.five_best_config_time[1]:
            if grasp_best == None or best_result[2] < grasp_best[2]:
                grasp_best = best_result
        
        for best_result in self.five_best_config_time[2]:
            if grasp_best == None or best_result[2] < grasp_best[2]:
                grasp_best = best_result

        return (sa_best, grasp_best, genetic_best)
    
    @property
    def best_config_sse(self) -> (tuple, tuple, tuple):
        sa_best : tuple = None
        grasp_best : tuple = None
        genetic_best : tuple = None
        
        for best_result in self.five_best_config_sse[0]:
            if sa_best == None or best_result[2] < sa_best[2]:
                sa_best = best_result

        for best_result in self.five_best_config_sse[1]:
            if grasp_best == None or best_result[2] < grasp_best[2]:
                grasp_best = best_result
        
        for best_result in self.five_best_config_sse[2]:
            if grasp_best == None or best_result[2] < grasp_best[2]:
                grasp_best = best_result

        return (sa_best, grasp_best, genetic_best)

    @property
    def best_config_zscore(self) -> (tuple, tuple, tuple):
        sa_best : tuple = None
        grasp_best : tuple = None
        genetic_best : tuple = None
        
        for best_result in self.five_best_config_zscore[0]:
            if sa_best == None or best_result[2] < sa_best[2]:
                sa_best = best_result

        for best_result in self.five_best_config_zscore[1]:
            if grasp_best == None or best_result[2] < grasp_best[2]:
                grasp_best = best_result
        
        for best_result in self.five_best_config_zscore[2]:
            if grasp_best == None or best_result[2] < grasp_best[2]:
                grasp_best = best_result

        return (sa_best, grasp_best, genetic_best)

    @property
    def method_statistic_data(self) -> (tuple, tuple, tuple):
        sses_sa : list = list([])
        sses_grasp : list = list([])
        sses_genetic : list = list([])
        z_scores_sa : np.array = None
        z_scores_grasp : np.array = None
        z_scores_genetic : np.array = None
        for sa_result in self._sa_results_list:
            z_scores_sa = np.contatenate(z_scores_sa, sa_result.z_scores)
        for grasp_result in self._grasp_results_list:
            z_scores_grasp = np.contatenate(z_scores_grasp, grasp_result.z_scores)
        for genetic_result in self._genetic_results_list:
            z_scores_genetic = np.contatenate(z_scores_genetic, genetic_result.z_scores)
        sa_data = (np.mean(z_scores_sa), np.std(z_scores_sa))
        grasp_data = (np.mean(z_scores_grasp), np.std(z_scores_grasp))
        genetic_data = (np.mean(z_scores_genetic), np.std(z_scores_genetic))
        return (sa_data, grasp_data, genetic_data)

# The results of training for each metaheuristic applied:
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
    
    # Z-Scores:
    @property
    def z_scores(self) -> np.array:
        return self._z_scores
    
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
    def best_config_time(self) -> tuple:
        min_time : float = min(self._avg_elapsed_list)
        return (self._method_name, self._hyper_param_list[self._avg_elapsed_list.index(min_time)], min_time)
    
    # Returing hyperparameter configuration with the best zscore:
    @property
    def best_config_zscore(self):
        min_zscore : float = np.amin(self._z_scores)
        return (self._method_name, self._hyper_param_list[np.where(self._z_scores == min_zscore)], min_zscore)
    
    # Returning hyperparameter configuration with the best avarage result:
    @property
    def best_config_avg_result(self):
        min_result : float = min(self._avg_sse_list)
        return (self._method_name, self._hyper_param_list[self._avg_sse_list.index(min_result)], min_result)
    
    # Returning the metaheuristic's avarage score:
    @property
    def avarage_zscore(self):
        return np.mean(self._z_scores)
    
    population_list = property(get_population_list, set_population_list)

def training(problems : dict) -> TrainingAnalysis:
    sa_results_list : list = list([])
    grasp_results_list : list = list([])
    genetic_results_list : list = list([])

    for problem_name, problem in problems.items():

        if problem_name == 'ionosphere':
            continue
        
        sa_hyper_params_list = problem['training']['sa']
        sa_result = train_sa(sa_hyper_params_list, problem_name)
        sa_results_list.append(sa_result)

        grasp_hyper_params_list = problem['training']['grasp']
        grasp_result = train_grasp(grasp_hyper_params_list, problem_name)
        grasp_results_list.append(grasp_result)

        genetic_hyper_params_list = problem['training']['genetic']
        genetic_result = train_genetic(genetic_hyper_params_list, problem_name)
        genetic_results_list.append(genetic_result)

    training_analysis : TrainingAnalysis = TrainingAnalysis(sa_results_list, grasp_results_list, genetic_results_list)
    return training_analysis

def train_sa(hyper_param_list: list, problem_name : str) -> TrainingResult:

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

    z_scores : np.array = sp.stats.zscore(avarage_sse_list)

    result : TrainingResult = TrainingResult("Simulated Annealing " + problem_name, avarage_sse_list, sse_list, avarage_elapsed_list, elapsed_list, z_scores, hyper_param_list, std_list)

    return result

def train_grasp(hyper_param_list : list, problem_name : str) -> TrainingResult:

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
        
        sse_lists.append(sse_list)
        avarage_sse_list.append(avarage_sse / 10)
        avarage_elapsed_list.append(avarage_elapsed / 10)

    z_scores : np.array = sp.stats.zscore(avarage_sse_list)

    result : TrainingResult = TrainingResult("GRASP " + problem_name, avarage_sse_list, sse_list, avarage_elapsed_list, elapsed_list, z_scores, hyper_param_list, std_list)

    return result

def train_genetic(hyper_param_list : list, problem_name : str) -> TrainingResult:

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
        
        sse_lists.append(sse_list)
        avarage_sse_list.append(avarage_sse / 10)
        avarage_elapsed_list.append(avarage_elapsed / 10)

    z_scores : np.array = sp.stats.zscore(avarage_sse_list)

    result : TrainingResult = TrainingResult("Genetic Algorithm " + problem_name, avarage_sse_list, sse_list, avarage_elapsed_list, elapsed_list, z_scores, hyper_param_list, std_list)
    result.population_list = population_list

    return result
