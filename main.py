#!/usr/bin/python3.8

import datasets as ds
import training as tr
import testing as ts
import graphics as gr

# Training the AI:
problems : dict = ds.load_problems()
training : tr.Training = tr.training(problems)

# Testing the AI:
testing : ts.Testing = ts.testing(training, problems)

gr.five_best_train_config_table(training)
gr.training_ranking_table(training)
gr.training_best_config_table(training)
gr.test_best_method(testing)
gr.test_statistic_data_table(testing)
gr.test_paired_test(testing)
gr.test_wilcoxon_test(testing)
gr.testing_avg_sse_boxplots(testing)
gr.testing_time_boxplots(testing)
gr.training_avg_sse_boxplots(training)
gr.training_time_boxplots(training)
