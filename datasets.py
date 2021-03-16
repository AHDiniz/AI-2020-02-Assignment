#!/usr/bin/python3

import numpy as np
import seaborn as sns
import sa
import grasp
import genetic
import clustering as clt

class Problem:
    def __init__(self, problem_name : str = "", data_set : np.array, k : int):
        self._data_set = np.copy(data_set)
        self._k = k
    
    @property
    def data_set(self) -> np.array:
        return self._data_set
    
    @property
    def k(self) -> int:
        return self._k

    @sa_hyper_params.setter
    def sa_hyper_params(self, hyper_params : sa.HyperParams):
        self._sa_hyper_params = hyper_params
    
    @property
    def sa_hyper_params(self):
        return self._sa_hyper_params
    
    @grasp_hyper_params.setter
    def grasp_hyper_params(self, hyper_params : grasp.HyperParams):
        self._grasp_hyper_params = hyper_params
    
    @property
    def grasp_hyper_params(self):
        return self._grasp_hyper_params
    
    @genetic_hyper_params.setter
    def genetic_hyper_params(self, hyper_params : genetic.HyperParams):
        self._genetic_hyper_params = hyper_params
    
    @property
    def genetic_hyper_params(self):
        return self._genetic_hyper_params
    
def load_problems() -> dict:
    problems = dict([])
    problems['iris'] = {'training': {'sa': [], 'grasp': [], 'genetic': [], 'kmeans': []}, 'testing': {'sa': [], 'grasp': [], 'genetic': [], 'kmeans': []}}
    problems['wine'] = {'training': {'sa': [], 'grasp': [], 'genetic': [], 'kmeans': []}, 'testing': {'sa': [], 'grasp': [], 'genetic': [], 'kmeans': []}}
    problems['ionosphere'] = {'training': {'sa': [], 'grasp': [], 'genetic': [], 'kmeans': []}, 'testing': {'sa': [], 'grasp': [], 'genetic': [], 'kmeans': []}}

    # Metaheuristic hyperparameters data:
    sa_hyper_params : list = []
    sa_init_temps : list = [500, 100, 50]
    sa_alphas : list = [.95,.85,.7]
    sa_num_iters : list = [350,500]
    for init_temp in sa_init_temps:
        for alpha in sa_alphas:
            for num_iter in sa_num_iters:
                sa_hyper_params.add(sa.HyperParams(init_temp, 0, num_iter, alpha))

    grasp_hyper_params : list = []
    grasp_num_iters : list = [20,50,100,200,350,500]
    grasp_num_bests : list = [5,10,15]
    for num_iter in grasp_num_iters:
        for num_best in grasp_num_bests:
            grasp_hyper_params.add(grasp.HyperParams(num_iter, num_best))

    genetic_hyper_params : list = []
    genetic_pop_sizes : list = [10,30,50]
    genetic_cross_rates : list = [.75,.85,.95]
    genetic_mut_rates : list = [.1,.2]
    for pop_size in genetic_pop_sizes:
        for cross_rate in genetic_cross_rates:
            for mut_rate in genetic_mut_rates:
                genetic_hyper_params.add(genetic.HyperParams(pop_size, cross_rate, mut_rate))

    # Creating clustering data for the iris seaborn dataset:
    iris_ks_training : list = [3,7,10,13,22]
    iris_ks_test : list = [2,4,8,11,15,17,23,28,32,50]

    iris_ds = sns.load_dataset('iris')

    iris_points_data : np.array = iris_ds[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]

    for k in iris_ks_training:
        for sa in sa_hyper_params:
            p = problems['iris']['training']['sa'].add(Problem(iris_points_data, k))
            p.sa_hyper_params = sa
        for grasp in grasp_hyper_params:
            p = problems['iris']['training']['grasp'].add(Problem(iris_points_data, k))
            p.grasp_hyper_params = grasp
        for genetic in genetic_hyper_params:
            p = problems['iris']['training']['genetic'].add(Problem(iris_points_data, k))
            p.genetic_hyper_params = genetic
        problems['iris']['training']['kmeans'].add(Problem(iris_points_data, k))
        
    
    for k in iris_ks_testing:
        for sa in sa_hyper_params:
            p = problems['iris']['testing']['sa'].add(Problem(iris_points_data, k))
            p.sa_hyper_params = sa
        for grasp in grasp_hyper_params:
            p = problems['iris']['testing']['grasp'].add(Problem(iris_points_data, k))
            p.grasp_hyper_params = grasp
        for genetic in genetic_hyper_params:
            p = problems['iris']['testing']['genetic'].add(Problem(iris_points_data, k))
            p.genetic_hyper_params = genetic
        problems['iris']['testing']['kmeans'].add(Problem(iris_points_data, k))

    return problems
