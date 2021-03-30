#!/usr/bin/python3.8

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import training as tr

def five_best_train_config_table(training_analysis : tr.TrainingAnalysis):
    five_best_sa : list = training_analysis.five_best_config_sse[0]
    five_best_grasp : list = training_analysis.five_best_config_sse[1]
    five_best_genetic : list = training_analysis.five_best_config_sse[2]

    print("Five best hyperparameter configuration for each method:")

    print("\nSimulated Annealing:")
    for sa_best in five_best_sa:
        print(sa_best[0], "\nInit Temp:", sa_best[1].sa_hyper_params.init_temp, "\nNum. Iter.:", sa_best[1].sa_hyper_params.num_iter,"\nAlpha:", sa_best[1].sa_hyper_params.alpha, "\nSSE:", sa_best[2])
    
    print("\nGRASP:")
    for grasp_best in five_best_grasp:
        print(grasp_best[0], "\nNum. Iter.:", grasp_best[1].grasp_hyper_params.num_iter, "\nNum. Best Solutions:", grasp_best[1].grasp_hyper_params.num_best_solutions, "\nSSE:", grasp_best[2])
    
    print("\nGenetic Algorithm:")
    for genetic_best in five_best_genetic:
        print(genetic_best[0], "\nPopulation Size:", genetic_best[1].genetic_hyper_params.population_size, "\nCrossover Rate:", genetic_best[1].genetic_hyper_params.crossover_rate, "\nMutation Rate:", genetic_best[1].genetic_hyper_params.mutation_rate, "\nSSE:", genetic_best[2])
    print("\n")

def training_time_boxplots(training_analysis : tr.TrainingAnalysis):
    sa_time : dict = dict({})
    grasp_time : dict = dict({})
    genetic_time : dict = dict({})
    for sa_result in training_analysis.sa_results_list:
        sa_time[problem_name] = sa_result.elapsed_list
    for grasp_result in training_analysis.grasp_results_list:
        grasp_time[problem_name] = grasp_result.elapsed_list
    for genetic_result in training_analysis.genetic_results_list:
        genetic_time[problem_name] = genetic_result.elapsed_list
    sa_time_df : pd.DataFrame = pd.DataFrame(sa_time, columns = ['iris','wine'])
    grasp_time_df : pd.DataFrame = pd.DataFrame(grasp_time, columns = ['iris','wine'])
    genetic_time_df : pd.DataFrame = pd.DataFrame(genetic_time, columns = ['iris','wine'])

    sa_boxplot = sns.boxplot(x = "variable", y = "values", data = pd.melt(sa_time_df))
    plt.savefig("graphics/sa_training_time.png")

    grasp_boxplot = sns.boxplot(x = "variable", y = "values", data = pd.melt(grasp_time_df))
    plt.savefig("graphics/grasp_training_time.png")

    genetic_boxplot = sns.boxplot(x = "variable", y = "values", data = pd.melt(genetic_time_df))
    plt.savefig("graphics/genetic_training_time.png")

def training_avg_sse_boxplots(training_analysis : tr.TrainingAnalysis):
    sa_sse : dict = dict({})
    grasp_sse : dict = dict({})
    genetic_sse : dict = dict({})
    for sa_result in training_analysis.sa_results_list:
        sa_sse[problem_name] = sa_result.sse_list
    for grasp_result in training_analysis.grasp_results_list:
        grasp_sse[problem_name] = grasp_result.sse_list
    for genetic_result in training_analysis.genetic_results_list:
        genetic_sse[problem_name] = genetic_result.sse_list
    sa_sse_df : pd.DataFrame = pd.DataFrame(sa_sse, columns = ['iris','wine'])
    grasp_sse_df : pd.DataFrame = pd.DataFrame(grasp_sse, columns = ['iris','wine'])
    genetic_sse_df : pd.DataFrame = pd.DataFrame(genetic_sse, columns = ['iris','wine'])

    sa_boxplot = sns.boxplot(x = "variable", y = "values", data = pd.melt(sa_sse_df))
    plt.savefig("graphics/sa_training_sse.png")

    grasp_boxplot = sns.boxplot(x = "variable", y = "values", data = pd.melt(grasp_sse_df))
    plt.savefig("graphics/grasp_training_sse.png")

    genetic_boxplot = sns.boxplot(x = "variable", y = "values", data = pd.melt(genetic_sse_df))
    plt.savefig("graphics/genetic_training_sse.png")

def training_ranking_table(training_analysis : tr.TrainingAnalysis):
    print("Hyperparameter configuration avarage z-score ranking for each problem in training")

    for problem_name, result in training_analysis.training_results.items():
        print(result.method_name, result.avarage_zscore)
    
    print("\n")

def training_best_config_table(training_analysis : tr.TrainingAnalysis):
    print("Best hyperparameter configuration by avarage ranking for each method")

    for best_time in training_analysis.best_config_zscore:
        print(best_time)
    
    print("\nBest hyperparameter configuration by avarage result for each method")

    for best_result in training_analysis.best_config_sse:
        print(best_result)


def test_statistic_data_table(testing_result : dict):
    return None

def testing_time_boxplots(testing_result : dict):
    return None

def testing_avg_sse_boxplots(testing_result : dict):
    return None


