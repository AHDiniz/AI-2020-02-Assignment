#!/usr/bin/python3

import scipy
import clustering as clt
import datasets as ds
import training as tr

# Training the AI:

results = tr.training(ds.load_problems())

# Testing the AI:
