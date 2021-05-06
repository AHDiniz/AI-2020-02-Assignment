#!/usr/bin/python3.8

import numpy as np
import seaborn as sns
from sklearn import datasets
import pandas as ps
import sa as sa
import grasp as grasp
import genetic as genetic
import clustering as clt

class Dataset:
    def __init__(self, points_data : np.array, training_ks : list, testing_ks : list):
        self.__points_data = points_data
        self.__training_ks = training_ks
        self.__testing_ks = testing_ks
    
    @property
    def points_data(self):
        return self.__points_data
    
    @property
    def training_ks(self):
        return self.__training_ks
    
    @property
    def testing_ks(self):
        return self.__testing_ks

def load_datasets() -> dict:
    result = {'Iris': None, 'Wine': None, 'Ionosphere': None}

    result['Iris'] = Dataset(datasets.load_iris()['data'], [3,7,10,13,22], [2,4,8,11,15,17,23,28,32,50])
    result['Wine'] = Dataset(datasets.load_wine()['data'], [2,6,9,11,13], [3,5,13,15,20,23,25,30,41,45])

    ionosphere_ds = ps.read_csv('ionosphere.data')
    ionosphere_data = ionosphere_ds.to_numpy()
    ionosphere_data = np.delete(ionosphere_data, ionosphere_data.shape[1] - 1, 1)

    result['Ionosphere'] = Dataset(ionosphere_data, None, [2,3,5,10,15,20,25,30,40,50])

    return result
