#!/usr/bin/python3.8

import datasets as ds
import training as tr
import testing as ts
import graphics as gr
import sa
import grasp
import genetic

datasets = ds.load_datasets()

sa_hyper_param_list = [
    sa.HyperParams(500, 0, 350, .95),
    sa.HyperParams(500, 0, 350, .85),
    sa.HyperParams(500, 0, 350, .7),
    sa.HyperParams(500, 0, 500, .95),
    sa.HyperParams(500, 0, 500, .85),
    sa.HyperParams(500, 0, 500, .7),
    sa.HyperParams(100, 0, 350, .95),
    sa.HyperParams(100, 0, 350, .85),
    sa.HyperParams(100, 0, 350, .7),
    sa.HyperParams(100, 0, 500, .95),
    sa.HyperParams(100, 0, 500, .85),
    sa.HyperParams(100, 0, 500, .7),
    sa.HyperParams(50, 0, 350, .95),
    sa.HyperParams(50, 0, 350, .85),
    sa.HyperParams(50, 0, 350, .7),
    sa.HyperParams(50, 0, 500, .95),
    sa.HyperParams(50, 0, 500, .85),
    sa.HyperParams(50, 0, 500, .7)
]

grasp_hyper_param_list = [
    grasp.HyperParams(20, 5),
    grasp.HyperParams(20, 10),
    grasp.HyperParams(20, 15),
    grasp.HyperParams(50, 5),
    grasp.HyperParams(50, 10),
    grasp.HyperParams(50, 15),
    grasp.HyperParams(100, 5),
    grasp.HyperParams(100, 10),
    grasp.HyperParams(100, 15),
    grasp.HyperParams(200, 5),
    grasp.HyperParams(200, 10),
    grasp.HyperParams(200, 15),
    grasp.HyperParams(350, 5),
    grasp.HyperParams(350, 10),
    grasp.HyperParams(350, 15),
    grasp.HyperParams(500, 5),
    grasp.HyperParams(500, 10),
    grasp.HyperParams(500, 15)
]

genetic_hyper_param_list = [
    genetic.HyperParams(10, .75, .1),
    genetic.HyperParams(10, .75, .2),
    genetic.HyperParams(10, .85, .1),
    genetic.HyperParams(10, .85, .2),
    genetic.HyperParams(10, .95, .1),
    genetic.HyperParams(10, .95, .2),
    genetic.HyperParams(30, .75, .1),
    genetic.HyperParams(30, .75, .2),
    genetic.HyperParams(30, .85, .1),
    genetic.HyperParams(30, .85, .2),
    genetic.HyperParams(30, .95, .1),
    genetic.HyperParams(30, .95, .2),
    genetic.HyperParams(50, .75, .1),
    genetic.HyperParams(50, .75, .2),
    genetic.HyperParams(50, .85, .1),
    genetic.HyperParams(50, .85, .2),
    genetic.HyperParams(50, .95, .1),
    genetic.HyperParams(50, .95, .2)
]

methods = {'SA': sa.simulated_annealing, 'GRASP': grasp.grasp, 'Genetic': genetic.genetic}

hyper_params = {'SA': sa_hyper_param_list, 'GRASP': grasp_hyper_param_list, 'Genetic': genetic_hyper_param_list}

training_results = {
    'Iris': {'SA': None, 'GRASP': None, 'Genetic': None},
    'Wine': {'SA': None, 'GRASP': None, 'Genetic': None}
}

for dataset_name, dataset in datasets.items():
    if not dataset_name == 'Ionosphere':
        for method_name, method in methods.items():
            training_results[dataset_name][method_name] = tr.training(method, dataset, hyper_params[method_name], method_name, dataset_name)

testing_results = {
    'Iris': {'SA': None, 'GRASP': None, 'Genetic': None},
    'Wine': {'SA': None, 'GRASP': None, 'Genetic': None},
    'Ionosphere': {'SA': None, 'GRASP': None, 'Genetic': None}
}

for dataset_name, dataset in datasets.items():
    for method_name, method in methods.items():
        testing_results[dataset_name][method_name] = ts.testing(method, dataset, training_results[dataset_name][method_name], method_name, dataset_name)
