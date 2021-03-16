#!/usr/bin/python3

import numpy as np
import pandas as ps
import matplotlib as mplot
import seaborn as sns
import clustering as clt

# Importing the Iris seaborn dataset and creating all the training and test scenarios for it:

train_ks_iris = [3,7,10,13,22]
test_ks_iris = [2,4,8,11,15,17,23,28,32,50]

iris_ds = sns.load_dataset("iris")
iris_data = iris_ds[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]

training_iris_sa = []
for i in train_ks_iris:
    training_iris_sa.add(clt.Clusters(i, iris_data, (1, 4)))

testing_iris_sa = []
for i in test_ks_iris:
    testing_iris_sa.add(clt.Clusters(i, iris_data, (1, 4)))


sa_params_list : list = [
    sa.HyperParams(500, 0, 350, 0.95),
    sa.HyperParams(100, 0, 350, 0.95),
    sa.HyperParams(50,  0, 350, 0.95),
    sa.HyperParams(500, 0, 500, 0.95),
    sa.HyperParams(100, 0, 500, 0.95),
    sa.HyperParams(50,  0, 500, 0.95),
    sa.HyperParams(500, 0, 350, 0.85),
    sa.HyperParams(100, 0, 350, 0.85),
    sa.HyperParams(50,  0, 350, 0.85),
    sa.HyperParams(500, 0, 500, 0.85),
    sa.HyperParams(100, 0, 500, 0.85),
    sa.HyperParams(50,  0, 500, 0.85),
    sa.HyperParams(500, 0, 350, 0.7),
    sa.HyperParams(100, 0, 350, 0.7),
    sa.HyperParams(50,  0, 350, 0.7),
    sa.HyperParams(500, 0, 500, 0.7),
    sa.HyperParams(100, 0, 500, 0.7),
    sa.HyperParams(50,  0, 500, 0.7)]
