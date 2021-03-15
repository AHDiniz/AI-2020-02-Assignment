#!/usr/bin/python3

import seaborn as sns
import clustering as clt

# Importing the Iris seaborn dataset and creating all the training and test scenarios for it:

train_ks_iris = [3,7,10,13,22]
test_ks_iris = [2,4,8,11,15,17,23,28,32,50]

iris_ds = sns.load_dataset("iris")

training_iris = []
for i in train_ks_iris:
    training_iris.add(clt.Clusters(i))

testing_iris = []
for i in test_ks_iris:
    testing_iris.add(clt.Clusters(i))
