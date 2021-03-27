#!/usr/bin/python3.8

import seaborn as sns
import matplotlib.pyplot as plt

def five_best_train_config_table(training_result):
    five_best_sa : list = []
    five_best_grasp : list = []
    five_best_genetic : list = []
    for problem_name, result in training_result:
        # Simulated Annealing:
        for sa_result in result['sa']:
            if len(five_best_sa) < 5:
                five_best_sa.append(sa_result)
            else:
                for appended in five_best_sa:
                    if sa_result[1] < appended[1]:
                        five_best_sa.remove(appended)
                        five_best_sa.append(sa_result)
                        break
        
        # GRASP:
        for grasp_result in result['grasp']:
            if len(five_best_grasp) < 5:
                five_best_grasp.append(grasp_result)
            else:
                for appended in five_best_grasp:
                    if grasp_result[1] < appended[1]:
                        five_best_sa.remove(appended)
                        five_best_sa.append(grasp_result)
                        break
        
        # Genetic Algorithm:
        for genetic_result in result['genetic']:
            if len(five_best_genetic) < 5:
                five_best_genetic.append(genetic_result)
            else:
                for appended in five_best_genetic:
                    if genetic_result[1] < appended[1]:
                        five_best_sa.remove(appended)
                        five_best_sa.append(genetic_result)
                        break
    
    # Printing the results of the 
    for sa_best in five_best_sa:
        print(sa_best[2],
            "\nInit Temp: " + sa_best[0].init_temp,
            "\nNum. Iter.: " + sa_best[0].num_iter,
            "\nAlpha: " + sa_best[0].alpha,
            "\nSSE: " + sa_best[1])
    
    for grasp_best in five_best_grasp:
        print(grasp_best[2],
            "\nNum. Iter.: " + grasp_best[0].num_iter,
            "\nNum. Best Solutions: " + grasp_best[0].num_best_solutions,
            "\nSSE: " + grasp_best[1])
    
    for genetic_best in five_best_genetic:
        print(genetic_best[2],
            "\nPopulation Size: " + genetic_best[0].population_size,
            "\nCrossover Rate: " + genetic_best[0].crossover_rate,
            "\nMutation Rate: " + genetic_best[0].mutation_rate,
            "\nSSE: " + genetic_best[1])

def training_time_boxplots(training_result):
    return None

def training_avg_sse_boxplots(training_result):
    return None

def training_ranking_table(training_result):
    return None

def training_best_config_table(training_result):
    return None


def test_statistic_data_table(testing_result):
    return None

def testing_time_boxplots(testing_result):
    return None

def testing_avg_sse_boxplots(testing_result):
    return None


