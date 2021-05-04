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

class ProblemTraining:
    def __init__(self, problem_name, hp_list, sse_lists, elapsed_lists, zscores_list, avg_sse_list, avg_elapsed_list):
        self._problem_name = problem_name
        self._hp_list = hp_list
        self._sse_lists = sse_lists
        self._elapsed_lists = elapsed_lists
        self._zscores_list = zscores_list
        self._avg_sse_list = avg_sse_list
        self._avg_elapsed_list = avg_elapsed_list
    
    @property
    def problem_name(self) -> str:
        return self._problem_name
    
    @property
    def hyper_param_list(self) -> list:
        return self._hp_list
    
    @property
    def sse_lists(self) -> list:
        return self._sse_lists
    
    @property
    def elapsed_lists(self) -> list:
        return self._elapsed_lists
    
    @property
    def zscores_list(self) -> list:
        return self._zscores_list
    
    @property
    def avg_sse_list(self) -> list:
        return self._avg_sse_list
    
    @property
    def avg_elapsed_list(self) -> list:
        return self._avg_elapsed_list

class MHTraining:
    def __init__(self, iris_training : ProblemTraining, wine_training : ProblemTraining, method_name : str):
        self._iris_training : ProblemTraining = iris_training
        self._wine_training : ProblemTraining = wine_training
        self._method_name : str = method_name

        self._five_best_config_time, self._five_best_config_sse, self._five_best_config_zscore = self.set_five_best_config()

        self._best_config_time, self._best_config_sse, self._best_config_zscore = self.set_best_config()

        self._sse_stats, self._time_stats, self._zscore_stats = self.calculate_statistics()
    
    @property
    def method_name(self):
        return self._method_name

    @property
    def iris_training(self):
        return self._iris_training
    
    @property
    def wine_training(self):
        return self._wine_training
    
    @property
    def five_best_config_time(self):
        return self._five_best_config_time
    
    @property
    def five_best_config_sse(self):
        return self._five_best_config_sse
    
    @property
    def five_best_config_zscore(self):
        return self._five_best_config_zscore
    
    @property
    def best_config_time(self):
        return self._best_config_time
    
    @property
    def best_config_sse(self):
        return self._best_config_sse
    
    @property
    def best_config_zscore(self):
        return self._best_config_zscore
    
    @property
    def sse_stats(self):
        return self._sse_stats
    
    @property
    def time_stats(self):
        return self._time_stats
    
    @property
    def zscore_stats(self):
        return self._zscore_stats
    
    def set_five_best_config(self) -> (list, list, list):
        five_time : list = list([])
        five_time_indices : list = list([])
        five_sse : list = list([])
        five_sse_indices : list = list([])
        five_zscore : list = list([])
        five_zscore_indices : list = list([])

        for i in range(len(self._iris_training.hyper_param_list)):
            config = self._iris_training.hyper_param_list[i]
            name = self.method_name + " " + self._iris_training.problem_name
            time = self._iris_training.avg_elapsed_list[i]
            sse = self._iris_training.avg_sse_list[i]
            zscore = self._iris_training.zscores_list[i]
            
            element_time = (config, name, time)
            element_sse = (config, name, sse)
            element_zscore = (config, name, zscore)
            
            if len(five_time) < 5:
                five_time.append(element_time)
                five_time_indices.append(i)
            else:
                max_t = max(five_time, key = lambda x : x[2])
                if time < max_t[2]:
                    five_time.remove(max_t)
                    five_time.append(element_time)
            
            if len(five_sse) < 5:
                five_sse.append(element_sse)
                five_sse_indices.append(i)
            else:
                max_sse = max(five_sse, key = lambda x : x[2])
                if sse < max_sse[2]:
                    five_sse.remove(max_sse)
                    five_sse.append(element_sse)

            if len(five_zscore) < 5:
                five_zscore.append(element_sse)
                five_zscore_indices.append(i)
            else:
                max_zscore = max(five_zscore, key = lambda x : x[2])
                if sse < max_zscore[2]:
                    five_zscore.remove(max_zscore)
                    five_zscore.append(element_zscore)
        
        for i in range(len(self._wine_training.hyper_param_list)):
            config = self._wine_training.hyper_param_list[i]
            name = self.method_name + " " + self._wine_training.problem_name
            time = self._wine_training.avg_elapsed_list[i]
            sse = self._wine_training.avg_sse_list[i]
            zscore = self._wine_training.zscores_list[i]
            
            element_time = (config, name, time)
            element_sse = (config, name, sse)
            element_zscore = (config, name, zscore)
            
            if len(five_time) < 5:
                five_time.append(element_time)
                five_time_indices.append(i)
            else:
                max_t = max(five_time, key = lambda x : x[2])
                if time < max_t[2]:
                    five_time.remove(max_t)
                    five_time.append(element_time)
            
            if len(five_sse) < 5:
                five_sse.append(element_sse)
                five_sse_indices.append(i)
            else:
                max_sse = max(five_sse, key = lambda x : x[2])
                if sse < max_sse[2]:
                    five_sse.remove(max_sse)
                    five_sse.append(element_sse)

            if len(five_zscore) < 5:
                five_zscore.append(element_zscore)
                five_zscore_indices.append(i)
            else:
                max_zscore= max(five_zscore, key = lambda x : x[2])
                if sse < max_zscore[2]:
                    five_zscore.remove(max_zscore)
                    five_zscore.append(element_zscore)

        return (five_time, five_sse, five_zscore)

    def set_best_config(self) -> (tuple, tuple, tuple):
        best_config_time : tuple = (None, None, None)
        best_config_sse : tuple = (None, None, None)
        best_config_zscore : tuple = (None, None, None)

        for i in range(len(self._iris_training.hyper_param_list)):
            config = self._iris_training.hyper_param_list[i]
            name = self.method_name + self._iris_training.problem_name
            time = self._iris_training.avg_elapsed_list[i]
            sse = self._iris_training.avg_sse_list[i]
            zscore = self._iris_training.zscores_list[i]

            if best_config_sse[2] == None or sse < best_config_sse[2]:
                best_config_sse = (config, name, sse)
            
            if best_config_time[2] == None or sse < best_config_time[2]:
                best_config_time = (config, name, time)
            
            if best_config_zscore[2] == None or sse < best_config_zscore[2]:
                best_config_zscore = (config, name, zscore)

        for i in range(len(self._wine_training.hyper_param_list)):
            config = self._wine_training.hyper_param_list[i]
            name = self.method_name + " " + self._wine_training.problem_name
            time = self._wine_training.avg_elapsed_list[i]
            sse = self._wine_training.avg_sse_list[i]
            zscore = self._wine_training.zscores_list[i]

            if best_config_sse[2] == None or sse < best_config_sse[2]:
                best_config_sse = (config, name, sse)
            
            if best_config_time[2] == None or sse < best_config_time[2]:
                best_config_time = (config, name, time)
            
            if best_config_zscore[2] == None or sse < best_config_zscore[2]:
                best_config_zscore = (config, name, zscore)

        return (best_config_time, best_config_sse, best_config_zscore)
    
    def calculate_statistics(self) -> (tuple, tuple, tuple):
        sses : list = self._iris_training.avg_sse_list + self._wine_training.avg_sse_list
        times : list = self._iris_training.avg_elapsed_list + self._wine_training.avg_elapsed_list
        zscores : list = self._iris_training.zscores_list + self._wine_training.zscores_list

        sse_data = (np.mean(sses), np.std(sses))
        time_data = (np.mean(times), np.std(times))
        zscore_data = (np.mean(zscores), np.std(zscores))

        return (sse_data, time_data, zscore_data)

class Training:
    def __init__(self, sa_training, grasp_training, genetic_training):
        self._sa_training = sa_training
        self._grasp_training = grasp_training
        self._genetic_training = genetic_training

        self._ranking = [self._sa_training, self._grasp_training, self._genetic_training]
        self._ranking.sort(key = lambda x: x.zscore_stats[0])
    
    @property
    def sa_training(self):
        return self._sa_training
    
    @property
    def grasp_training(self):
        return self._grasp_training
    
    @property
    def genetic_training(self):
        return self._genetic_training
    
    @property
    def ranking(self):
        return self._ranking

def training(problems : dict) -> Training:
    iris_sa = None
    wine_sa = None

    iris_grasp = None
    wine_grasp = None

    iris_genetic = None
    wine_genetic = None

    for problem_name, problem in problems.items():

        if problem_name == 'ionosphere':
            continue
        
        sa_hyper_params_list = problem['training']['sa']
        sa_result : ProblemTraining = train_sa(sa_hyper_params_list, problem_name)
        if problem_name == 'iris':
            iris_sa : ProblemTraining = sa_result
        elif problem_name == 'wine':
            wine_sa : ProblemTraining = sa_result

        grasp_hyper_params_list = problem['training']['grasp']
        grasp_result : ProblemTraining = train_grasp(grasp_hyper_params_list, problem_name)
        if problem_name == 'iris':
            iris_grasp : ProblemTraining = grasp_result
        elif problem_name == 'wine':
            wine_grasp : ProblemTraining = grasp_result

        genetic_hyper_params_list = problem['training']['genetic']
        genetic_result : ProblemTraining = train_genetic(genetic_hyper_params_list, problem_name)
        if problem_name == 'iris':
            iris_genetic : ProblemTraining = genetic_result
        elif problem_name == 'wine':
            wine_genetic : ProblemTraining = genetic_result

    sa_result : MHTraining = MHTraining(iris_sa, wine_sa, "Simulated Annealing")
    grasp_result : MHTraining = MHTraining(iris_grasp, wine_grasp, "GRASP")
    genetic_result : MHTraining = MHTraining(iris_genetic, wine_genetic, "Genetic Algorithm")

    return Training(sa_result, grasp_result, genetic_result)

def train_sa(hyper_param_list: list, problem_name : str) -> ProblemTraining:

    avarage_sse_list : list = []
    sse_lists : list = []
    std_list : list = []

    avarage_elapsed_list : list = []
    elapsed_lists : list = []

    for config in hyper_param_list:
        k : int = config.k
        hyper_params : sa.HyperParams = config.sa_hyper_params
        
        avarage_sse : float = 0
        avarage_elapsed : float = 0
        sse_list : list = []
        elapsed_list : list = []

        for i in range(10):
            clusters : clt.Clusters = clt.Clusters(k, config.data_set)
            (result, elapsed_time) = sa.simulated_annealing(hyper_params, clusters)
            sse_list.append(result)
            elapsed_list.append(elapsed_time)
            avarage_sse += result
            avarage_elapsed += elapsed_time
        
        std_list.append(np.std(sse_list))
        
        sse_lists.append(sse_list)
        elapsed_lists.append(elapsed_list)
        avarage_sse_list.append(avarage_sse / 10)
        avarage_elapsed_list.append(avarage_elapsed / 10)

    z_scores : np.array = sp.stats.zscore(avarage_sse_list)

    problem_training = ProblemTraining(problem_name, hyper_param_list, sse_lists, elapsed_lists, z_scores, avarage_sse_list, avarage_elapsed_list)

    return problem_training

def train_grasp(hyper_param_list : list, problem_name : str) -> ProblemTraining:

    avarage_sse_list : list = []
    sse_lists : list = []
    std_list : list = []

    avarage_elapsed_list : list = []
    elapsed_lists : list = []

    for config in hyper_param_list:
        k : int = config.k
        hyper_params : grasp.HyperParams = config.grasp_hyper_params
        
        avarage_sse : float = 0
        avarage_elapsed : float = 0
        sse_list : list = []
        elapsed_list : list = []

        for i in range(10):
            clusters : clt.Clusters = clt.Clusters(k, config.data_set)
            (result, elapsed_time) = grasp.grasp(hyper_params, clusters)
            sse_list.append(result)
            elapsed_list.append(elapsed_time)
            avarage_sse += result
            avarage_elapsed += elapsed_time
        
        std_list.append(np.std(sse_list))
        
        sse_lists.append(sse_list)
        elapsed_lists.append(elapsed_list)
        avarage_sse_list.append(avarage_sse / 10)
        avarage_elapsed_list.append(avarage_elapsed / 10)

    z_scores : np.array = sp.stats.zscore(avarage_sse_list)

    problem_training = ProblemTraining(problem_name, hyper_param_list, sse_lists, elapsed_lists, z_scores, avarage_sse_list, avarage_elapsed_list)

    return problem_training

def train_genetic(hyper_param_list : list, problem_name : str) -> ProblemTraining:

    avarage_sse_list : list = []
    sse_lists : list = []
    std_list : list = []

    avarage_elapsed_list : list = []
    elapsed_lists : list = []

    for config in hyper_param_list:
        k : int = config.k
        clusters : clt.Clusters = clt.Clusters(k, config.data_set)
        hyper_params : genetic.HyperParams = config.genetic_hyper_params
        
        avarage_sse : float = 0
        avarage_elapsed : float = 0
        sse_list : list = []
        elapsed_list : list = []

        population : genetic.Population = genetic.Population(k, hyper_params.population_size, clusters)
        
        for i in range(10):
            (result, elapsed_time) = genetic.genetic(hyper_params, clusters, population)
            sse_list.append(result)
            elapsed_list.append(elapsed_time)
            avarage_sse += result
            avarage_elapsed += elapsed_time
        
        std_list.append(np.std(sse_list))
        
        sse_lists.append(sse_list)
        elapsed_lists.append(elapsed_list)
        avarage_sse_list.append(avarage_sse / 10)
        avarage_elapsed_list.append(avarage_elapsed / 10)

    z_scores : np.array = sp.stats.zscore(avarage_sse_list)

    problem_training = ProblemTraining(problem_name, hyper_param_list, sse_lists, elapsed_lists, z_scores, avarage_sse_list, avarage_elapsed_list)

    return problem_training
