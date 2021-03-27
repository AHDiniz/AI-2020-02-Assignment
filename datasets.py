#!/usr/bin/python3.8

import numpy as np
import seaborn as sns
from sklearn import datasets
import pandas as ps
import sa as sa
import grasp as grasp
import genetic as genetic
import clustering as clt

class Problem:
    def __init__(self, problem_name : str, data_set : np.array, k : int):
        self._data_set = np.copy(data_set)
        self._k = k
        self._sa_hyper_params = None
        self._grasp_hyper_params = None
        self._genetic_hyper_params = None
    
    @property
    def data_set(self) -> np.array:
        return self._data_set
    
    @property
    def k(self) -> int:
        return self._k

    def set_sa_hyper_params(self, hyper_params : sa.HyperParams):
        self._sa_hyper_params = hyper_params
    
    @property
    def sa_hyper_params(self):
        return self._sa_hyper_params
    
    def set_grasp_hyper_params(self, hyper_params : grasp.HyperParams):
        self._grasp_hyper_params = hyper_params
    
    @property
    def grasp_hyper_params(self):
        return self._grasp_hyper_params
    
    def set_genetic_hyper_params(self, hyper_params : genetic.HyperParams):
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
                sa_hyper_params.append(sa.HyperParams(init_temp, 0, num_iter, alpha))

    grasp_hyper_params : list = []
    grasp_num_iters : list = [20,50,100,200,350,500]
    grasp_num_bests : list = [5,10,15]
    for num_iter in grasp_num_iters:
        for num_best in grasp_num_bests:
            grasp_hyper_params.append(grasp.HyperParams(num_iter, num_best))

    genetic_hyper_params : list = []
    genetic_pop_sizes : list = [10,30,50]
    genetic_cross_rates : list = [.75,.85,.95]
    genetic_mut_rates : list = [.1,.2]
    for pop_size in genetic_pop_sizes:
        for cross_rate in genetic_cross_rates:
            for mut_rate in genetic_mut_rates:
                genetic_hyper_params.append(genetic.HyperParams(pop_size, cross_rate, mut_rate))

    # Creating clustering data for the iris seaborn dataset:
    iris_ks_training : list = [3,7,10,13,22]
    iris_ks_test : list = [2,4,8,11,15,17,23,28,32,50]

    iris_ds = sns.load_dataset('iris')

    iris_points_data : np.array = iris_ds[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]

    for k in iris_ks_training:
        for sa_hps in sa_hyper_params:
            p = Problem("sa", iris_points_data, k)
            p.set_sa_hyper_params(sa_hps)
            problems['iris']['training']['sa'].append(p)
        for grasp_hps in grasp_hyper_params:
            p = Problem("grasp", iris_points_data, k)
            p.set_grasp_hyper_params(grasp_hps)
            problems['iris']['training']['grasp'].append(p)
        for genetic_hps in genetic_hyper_params:
            p = Problem("genetic", iris_points_data, k)
            p.set_genetic_hyper_params(genetic_hps)
            problems['iris']['training']['genetic'].append(p)
        problems['iris']['training']['kmeans'].append(Problem("kmeans", iris_points_data, k))
    
    for k in iris_ks_test:
        for sa_hps in sa_hyper_params:
            p = Problem("sa", iris_points_data, k)
            p.set_sa_hyper_params(sa_hps)
            problems['iris']['testing']['sa'].append(p)
        for grasp_hps in grasp_hyper_params:
            p = Problem("grasp", iris_points_data, k)
            p.set_grasp_hyper_params(grasp_hps)
            problems['iris']['testing']['grasp'].append(p)
        for genetic_hps in genetic_hyper_params:
            p = Problem("genetic", iris_points_data, k)
            p.set_genetic_hyper_params(genetic_hps)
            problems['iris']['testing']['genetic'].append(p)
        problems['iris']['testing']['kmeans'].append(Problem("kmeans", iris_points_data, k))

    # Creating clustering data for the wine data set:
    wine_ks_training : list = [2,6,9,11,13]
    wine_ks_test : list = [3,5,13,15,20,23,25,30,41,45]

    wine_ds = datasets.load_wine()
    wine_points_data : np.array = wine_ds['data']

    for k in wine_ks_training:
        for sa_hps in sa_hyper_params:
            p = Problem("sa", wine_points_data, k)
            p.set_sa_hyper_params(sa_hps)
            problems['wine']['training']['sa'].append(p)
        for grasp_hps in grasp_hyper_params:
            p = Problem("grasp", wine_points_data, k)
            p.set_grasp_hyper_params(grasp_hps)
            problems['wine']['training']['grasp'].append(p)
        for genetic_hps in genetic_hyper_params:
            p = Problem("genetic", iris_points_data, k)
            p.set_genetic_hyper_params(genetic_hps)
            problems['wine']['training']['genetic'].append(p)
        problems['wine']['training']['kmeans'].append(Problem("kmeans", wine_points_data, k))
    
    for k in wine_ks_test:
        for sa_hps in sa_hyper_params:
            p = Problem("sa", wine_points_data, k)
            p.set_sa_hyper_params(sa_hps)
            problems['wine']['testing']['sa'].append(p)
        for grasp_hps in grasp_hyper_params:
            p = Problem("grasp", wine_points_data, k)
            p.set_grasp_hyper_params(grasp_hps)
            problems['wine']['testing']['grasp'].append(p)
        for genetic_hps in genetic_hyper_params:
            p = Problem("genetic", wine_points_data, k)
            p.set_genetic_hyper_params(genetic_hps)
            problems['wine']['testing']['genetic'].append(p)
        problems['wine']['testing']['kmeans'].append(Problem("kmeans", wine_points_data, k))
    
    # Creating clustering data for the ionosphere dataset:
    iono_ks_test : list = [2,3,5,10,15,20,25,30,40,50]
    iono_ds = ps.read_csv('ionosphere.data')
    iono_points_data = iono_ds.to_numpy()
    iono_points_data = np.delete(iono_points_data, iono_points_data.shape[1] - 1, 1)

    for k in iono_ks_test:
        for sa_hps in sa_hyper_params:
            p = Problem("sa", iono_points_data, k)
            p.set_sa_hyper_params(sa_hps)
            problems['iono']['testing']['sa'].append(p)
        for grasp_hps in grasp_hyper_params:
            p = Problem("grasp", iono_points_data, k)
            p.set_grasp_hyper_params(grasp_hps)
            problems['iono']['testing']['grasp'].append(p)
        for genetic_hps in genetic_hyper_params:
            p = Problem("genetic", iono_points_data, k)
            p.set_genetic_hyper_params(genetic_hps)
            problems['iono']['testing']['genetic'].append(p)
        problems['iono']['testing']['kmeans'].append(Problem("kmeans", iono_points_data, k))

    return problems
