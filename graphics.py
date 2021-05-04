#!/usr/bin/python3.8

import seaborn as sns
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import training as tr
import testing as ts

def five_best_train_config_table(t : tr.Training):
    five_best_sa : list = t.sa_training.five_best_config_sse
    five_best_grasp : list = t.grasp_training.five_best_config_sse
    five_best_genetic : list = t.genetic_training.five_best_config_sse

    print("Five best hyperparameter configuration for each method:")

    print("\nSimulated Annealing:")
    for sa_best in five_best_sa:
        print(sa_best[1], "\nInit Temp:", sa_best[0].init_temp, "\nNum. Iter.:", sa_best[0].num_iter,"\nAlpha:", sa_best[1].sa_hyper_params.alpha, "\nSSE:", sa_best[2])
    
    print("\nGRASP:")
    for grasp_best in five_best_grasp:
        print(grasp_best[1], "\nNum. Iter.:", grasp_best[0].num_iter, "\nNum. Best Solutions:", grasp_best[0].num_best_solutions, "\nSSE:", grasp_best[2])
    
    print("\nGenetic Algorithm:")
    for genetic_best in five_best_genetic:
        print(genetic_best[1], "\nPopulation Size:", genetic_best[0].population_size, "\nCrossover Rate:", genetic_best[0].crossover_rate, "\nMutation Rate:", genetic_best[0].mutation_rate, "\nSSE:", genetic_best[2])
    print("\n")

def training_time_boxplots(t : tr.Training):
    sa_time : dict = dict({})
    grasp_time : dict = dict({})
    genetic_time : dict = dict({})
    
    sa_time['iris'] = t.sa_training.iris_training.avg_elapsed_list
    sa_time['wine'] = t.sa_training.wine_training.avg_elapsed_list

    grasp_time['iris'] = t.grasp_training.iris_training.avg_elapsed_list
    grasp_time['wine'] = t.grasp_training.wine_training.avg_elapsed_list

    genetic_time['iris'] = t.genetic_training.iris_training.avg_elapsed_list
    genetic_time['wine'] = t.genetic_training.wine_training.avg_elapsed_list

    sa_time_df : pd.DataFrame = pd.DataFrame(sa_time, columns = ['iris','wine'])
    grasp_time_df : pd.DataFrame = pd.DataFrame(grasp_time, columns = ['iris','wine'])
    genetic_time_df : pd.DataFrame = pd.DataFrame(genetic_time, columns = ['iris','wine'])

    sa_boxplot = sns.boxplot(x = "variable", y = "values", data = pd.melt(sa_time_df))
    plt.savefig("graphics/sa_training_time.png")

    grasp_boxplot = sns.boxplot(x = "variable", y = "values", data = pd.melt(grasp_time_df))
    plt.savefig("graphics/grasp_training_time.png")

    genetic_boxplot = sns.boxplot(x = "variable", y = "values", data = pd.melt(genetic_time_df))
    plt.savefig("graphics/genetic_training_time.png")

def training_avg_sse_boxplots(t : tr.Training):
    sa_sse : dict = dict({})
    grasp_sse : dict = dict({})
    genetic_sse : dict = dict({})
    
    sa_sse['iris'] = t.sa_training.iris_training.avg_sse_list
    sa_sse['wine'] = t.sa_training.wine_training.avg_sse_list

    grasp_sse['iris'] = t.grasp_training.iris_training.avg_sse_list
    grasp_sse['wine'] = t.grasp_training.wine_training.avg_sse_list

    genetic_sse['iris'] = t.genetic_training.iris_training.avg_sse_list
    genetic_sse['wine'] = t.genetic_training.wine_training.avg_sse_list

    sa_sse_df : pd.DataFrame = pd.DataFrame(sa_sse, columns = ['iris','wine'])
    grasp_sse_df : pd.DataFrame = pd.DataFrame(grasp_sse, columns = ['iris','wine'])
    genetic_sse_df : pd.DataFrame = pd.DataFrame(genetic_sse, columns = ['iris','wine'])

    sa_boxplot = sns.boxplot(data = pd.melt(sa_sse_df))
    plt.savefig("graphics/sa_training_time.png")

    grasp_boxplot = sns.boxplot(data = pd.melt(grasp_sse_df))
    plt.savefig("graphics/grasp_training_time.png")

    genetic_boxplot = sns.boxplot(data = pd.melt(genetic_sse_df))
    plt.savefig("graphics/genetic_training_time.png")

def training_ranking_table(t : tr.Training):
    print("Metaheuristic ranking based on avarage z-score in training")

    for result in t.ranking:
        print(result[0].method_name, result[1])
    
    print("\n")

def training_best_config_table(t : tr.Training):
    print("Best hyperparameter configuration by avarage zscore for each method")

    config_sa, _, zscore_sa = t.sa_training.best_config_zscore
    print("Simulated Annealing", "Initial temperature =", config_sa.init_temp, "Alpha =", config_sa.alpha, "Number of iterations =", config_sa.num_iter, "z-score =", zscore_sa)

    config_grasp, _, zscore_grasp = t.grasp_training.best_config_zscore
    print("GRASP", "Number of iterations =", config_grasp.num_iter, "Number of best solutions =", config_grasp.num_best_solutions, "z-score =", zscore_grasp)

    config_genetic, _, zscore_genetic = t.genetic_training.best_config_zscore
    print("Genetic Algorithm", "Crossover rate =", config_genetic.crossover_rate, "Mutation rate =", config_genetic.mutation_rate, "z-score =", zscore_genetic)
    
    print("\nBest hyperparameter configuration by avarage result for each method")

    config_sa, _, sse_sa = t.sa_training.best_config_sse
    print("Simulated Annealing", "Initial temperature =", config_sa.init_temp, "Alpha =", config_sa.alpha, "Number of iterations =", config_sa.num_iter, "SSE =", sse_sa)

    config_grasp, _, sse_grasp = t.grasp_training.best_config_sse
    print("GRASP", "Number of iterations =", config_grasp.num_iter, "Number of best solutions =", config_grasp.num_best_solutions, "SSE =", sse_grasp)

    config_genetic, _, sse_genetic = t.genetic_training.best_config_sse
    print("Genetic Algorithm", "Crossover rate =", config_genetic.crossover_rate, "Mutation rate =", config_genetic.mutation_rate, "SSE =", sse_genetic)

    print("\nBest hyperparameter configuration by avarage elapsed time for each method")

    config_sa, _, time_sa = t.sa_training.best_config_time
    print("Simulated Annealing", "Initial temperature =", config_sa.init_temp, "Alpha =", config_sa.alpha, "Number of iterations =", config_sa.num_iter, "time =", time_sa)

    config_grasp, _, time_grasp = t.grasp_training.best_config_time
    print("GRASP", "Number of iterations =", config_grasp.num_iter, "Number of best solutions =", config_grasp.num_best_solutions, "time =", time_grasp)

    config_genetic, _, time_genetic = t.genetic_training.best_config_time
    print("Genetic Algorithm", "Crossover rate =", config_genetic.crossover_rate, "Mutation rate =", config_genetic.mutation_rate, "time =", time_genetic)







def test_best_method(t : ts.Testing):
    best = t.ranking[0]

    print("Best method to solve the Clustering Problem:")
    print(best.method_name, "Avarage Z-Score =", best.avg_zscore)

def test_wilcoxon_test(t : ts.Testing):
    print("Wilcoxon test table")
    
    methods = [t.sa_testing, t.grasp_testing, t.genetic_testing, t.kmeans_testing]

    for i in range(len(methods)):
        v1 = methods[i].avg_zscore_list
        for j in range(i + 1, len(methods)):
            v2 = methods[j].avg_zscore_list
            s,p = sp.stats.wilcoxon(v1, v2)
            print(methods[i].method_name, methods[j].method_name, s, p)

def test_paired_test(t : ts.Testing):
    print("Paired test table")
    
    methods = [t.sa_testing, t.grasp_testing, t.genetic_testing, t.kmeans_testing]

    for i in range(len(methods)):
        v1 = methods[i].avg_zscore_list
        for j in range(i + 1, len(methods)):
            v2 = methods[j].avg_zscore_list
            s,p = sp.stats.ttest_rel(v1, v2)
            print(methods[i].method_name, methods[j].method_name, s, p)

def test_statistic_data_table(t : ts.Testing):
    print("Simulated Annealing testing statistic data.")
    print("Avarage SSE =",t.sa_testing.sse_statistics[0],"SSE Standard Deviation =",t.sa_testing.sse_statistics[1])

    print("\nGRASP testing statistic data.")
    print("Avarage SSE =",t.grasp_testing.sse_statistics[0],"SSE Standard Deviation =",t.grasp_testing.sse_statistics[1])

    print("\nGenetic Algorithm testing statistic data.")
    print("Avarage SSE =",t.genetic_testing.sse_statistics[0],"SSE Standard Deviation =",t.genetic_testing.sse_statistics[1])

    print("\nK-Means testing statistic data.")
    print("Avarage SSE =",t.kmeans_testing.sse_statistics[0],"SSE Standard Deviation =",t.kmeans_testing.sse_statistics[1])

def testing_time_boxplots(t : ts.Testing):
    elapsed = dict({})

    elapsed['sa'] = t.sa_testing.avg_elapsed_list
    elapsed['grasp'] = t.grasp_testing.avg_elapsed_list
    elapsed['genetic'] = t.genetic_testing.avg_elapsed_list
    elapsed['kmeans'] = t.kmeans_tesing.avg_elapsed_list

    elapsed_df = pd.DataFrame(elapsed, columns = ['sa','grasp','genetic','kmeans'])
    boxplot = sns.boxplot(data = pd.melt(elapsed_df))
    plt.savefig("graphics/testing_sse.png")

def testing_avg_sse_boxplots(t : ts.Testing):
    sses = dict({})

    sses['sa'] = t.sa_testing.avg_sse_list
    sses['grasp'] = t.grasp_testing.avg_sse_list
    sses['genetic'] = t.genetic_testing.avg_sse_list
    sses['kmeans'] = t.kmeans_tesing.avg_sse_list

    sses_df = pd.DataFrame(sses, columns = ['sa','grasp','genetic','kmeans'])
    boxplot = sns.boxplot(data = pd.melt(sses_df))
    plt.savefig("graphics/testing_sse.png")
