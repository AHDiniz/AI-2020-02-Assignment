#!/usr/bin/python3

import scipy
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
