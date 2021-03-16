#!/usr/bin/python3

import scipy
import clustering as clt
import datasets as ds
import kmeans
import sa
import grasp
import genetic
import training

'''
For each problem:
    For each method with best config.:
        Execute method 20 times and get avarage result and avarage elapsed time
    Execute k-means 20 times and get avarage result and avarage elapsed time
    Apply z-score to avarage results
    Rank results
Get standard avarage, standard deviation, corresponding times and rankings of each method
Make statistical test by pairs:
    Paired t
    Wilcoxon
Paired table with results
Return best method in general by avarage result and by avarage ranking
'''
