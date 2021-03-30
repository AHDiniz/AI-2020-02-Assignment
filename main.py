#!/usr/bin/python3.8

import datasets as ds
import training as tr
import testing as ts
import graphics as gr

# Training the AI:
problems : dict = ds.load_problems()
training_analysis : tr.TrainingAnalysis = tr.training(problems)

gr.training_avg_sse_boxplots(training_analysis)
gr.training_time_boxplots(training_analysis)
gr.five_best_train_config_table(training_analysis)
gr.training_best_config_table(training_analysis)
gr.training_ranking_table(training_analysis)

# Testing the AI:

