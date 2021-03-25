#!/usr/bin/python3

import datasets as ds
import training as tr

# Training the AI:

results = tr.training(ds.load_problems())

# Testing the AI:
