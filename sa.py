#!/usr/bin/python3

def simulated_annealing(initial_temp, final_temp, temp_var_func, num_iter, clusters):
    current_temp = initial_temp
    eq_temp = equilibrium_temp(initial_temp, final_temp)
    while current_temp > final_temp:
        while not current_temp == eq_temp:
            next_state = disturb(clusters)
            delta = cost(next_state) - cost(clusters)
            if delta <= 0:
                clusters = next_state
            else:
                clusters = next_state if check_update_state(next_state) else clusters
        current_temp = temp_var_func(current_temp)
    return clusters

def check_update_state(next_state):
    return False

def equilibrium_temp(initial_temp, final_temp):
    return 0

def disturb_state(clusters):
    return clusters

def cost(clusters):
    return 0
