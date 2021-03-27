#!/usr/bin/python3.8

import datasets as ds
import training as tr
import testing as ts
import graphics as gr

# Training the AI:
problems : dict = ds.load_problems()
training_results : dict = tr.training(problems)

gr.training_best_config_table(training_results)

# Testing the AI:
testing_results : dict = ts.testing(training_results, problems)
