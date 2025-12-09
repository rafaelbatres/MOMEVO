"""
==============================================================================
Project:      MOMEVO  
Filename:     test-ultraseven-with-memory.py  
Description:  The MOMEVO algorithm  
Author:       Rafael Batres  
Created:      March 14, 2025 
Last Updated: March 16, 2025
Version:      1.5
==============================================================================

Change Log:
------------------------------------------------------------------------------
Date          Version     Author         Description
------------  ---------   ------------   --------------------------------------
March-15-2025    1.0      Rafael Batres  Implemented a memory allocation method 
                                         for the decision and objective spaces.
                                         Implemmented a memory mechanism to avoid
                                         simulating already simulated solutions.
March-15-2025    1.1      Rafael Batres  Implemented a space reduction approach
                                         every epoch.
March-16-2025    1.2      Rafael Batres  Implemented a memory allocation scheme
                                         for the Pareto solutions.
March-17-2025    1.3      Rafael Batres  Solved the problem of incorrectly saving
                                         the surrogate Pareto decision results,
                                         instead of the real Pareto values.
                 1.4      Rafael Batres  Now each solution in the result files is
                                         separated by a comma.
                 1.5      Rafael Batres  Added number of Pareto solutions in the
                                         results files.
                 1.6      Rafael Batres  Changed y_f2 = np.array(obj_func_values[:, 0])
                                         to y_f2 = np.array(obj_func_values[:, 1])
                 1.7      Rafael Batres  Instead of the random solution, introduced
                                         a crossover between random solution and 
                                         selected_x
                 1.8      Rafael Batres  Probability of 0.07 for crossover, and 
                                         0.01 for mutation (mutation of selected_x)
------------------------------------------------------------------------------
"""

import gc
import numpy as np
import pandas as pd
from datetime import datetime
from functools import wraps
import time
from warnings import catch_warnings, simplefilter
from itertools import repeat
from numpy import vstack, asarray
import matplotlib.pyplot as plt
import math
import random
import copy
import os
from math import pi

# Sampling
from skopt.space import Space
from skopt.sampler import Sobol 
from skopt.sampler import Lhs
from scipy.spatial.distance import cdist
import skopt.space

# Surrogate
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.exceptions import ConvergenceWarning
import warnings

# Ensemble model
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR


import deap.benchmarks as db

from utils.target_space import TargetSpace

# Platypus
from Platypus.platypus.algorithms import NSGAIII
from Platypus.platypus import Problem, Real
from Platypus.platypus import core   

#from utils.hv import *
from utils.hv import HyperVolume


# ZDT1 configuration
num_variables = 30
initial_lower_bound = [0] * num_variables
initial_upper_bound = [1] * num_variables
PB = np.asarray([[0, 1]] * num_variables)
benchmark = 'zdt1.txt'


# Samples for initial sampling plan
n_samples = 64

Xsamples_history_size = 128
initial_population_size = 100 

max_iter = 29 # 32

# runs the algorithm number_of_epochs times
number_of_epochs = 10 #10

number_of_runs = 1
max_simulation_times = 468 # 468

NObj = 2



# db benchmarks 
def objective(x):
    return np.asarray(db.zdt1(x))
    
# objective function for the surrogate model evaluation
def spea2_obj_func(x):
    f = np.zeros((x.shape[0], 2))

    i = 0
    for sol in x:
        f[i, 0] = surrogate_f1(sol)
        f[i, 1] = surrogate_f2(sol)
        i += 1
    
    return f


def surrogate_f1(x):
    global ensemble_model_f1

    with catch_warnings():
        simplefilter("ignore")
        
        if isinstance(x, np.ndarray):
            surr_prediction = ensemble_model_f1.predict([x.tolist()])
        else:
            surr_prediction = ensemble_model_f1.predict([x])
        surr_prediction_arr = asarray(surr_prediction)
      
        surr_prediction = (surr_prediction_arr.T)
        ensemble_result = float(surr_prediction)
        
        return ensemble_result

def surrogate_f2(x):
    global ensemble_model_f2

    with catch_warnings():
        simplefilter("ignore")

        if isinstance(x, np.ndarray):
            surr_prediction = ensemble_model_f2.predict([x.tolist()])
        else:
            surr_prediction = ensemble_model_f2.predict([x])
            
        surr_prediction_arr = asarray(surr_prediction)
      
        surr_prediction = (surr_prediction_arr.T)
        ensemble_result = float(surr_prediction)
        
        return ensemble_result
    
def platypus_obj_func(x):
    f1 = surrogate_f1(x)
    f2 = surrogate_f2(x)
    return [f1, f2]

#-----------

class Root:
    """
        This class is Abstract class for all other class to inherit
    """

    def __init__(self, pareto_front=None, reference_front=None):
        """
        :param pareto_front: list/tuple or 2d-array (matrix) of non-dominated front (pareto front obtained from your test case)
        :param reference_front: list/tuple or 2d-array (matrix) of True pareto-front or your appropriate front you wish to be reference front
        """
        self.messages = []
        self.flag = True
        self.n_objs = 0
        # When creating object, you can pass pareto front with different size, or even None. It wont effect the program
        # But when you calling the function, if you pass None or front with different size --> this flag will be triggered

        self.pareto_front = self.check_convert_front(pareto_front)
        self.reference_front = self.check_convert_front(reference_front)

    def check_convert_front(self, front=None, converted_type="float64"):
        if front is None:
            return None
        else:
            if type(front) in [list, tuple]:
                front_temp = np.array(front)
                if type(front_temp[0]) is not np.ndarray:
                    self.messages.append("Some points in your front have different size. Please check again")
                    self.flag = False
                    return None
                else:
                    front_temp = front_temp.astype(converted_type)
                    check_none = np.isnan(front_temp).any()
                    check_infinite = np.isfinite(front_temp).all()
                    if check_none or not check_infinite:
                        self.messages.append("Some points in your front contain None/Infinite value. Please check again")
                        self.flag = False
                        return None
                    else:
                        return front_temp
            if type(front) is np.ndarray:
                return front

    def print_messages(self):
        for msg in self.messages:
            print(msg)

    def check_hypervolume_point(self, hv_point=None):
        if hv_point is None:
            self.messages.append("Need Hypervolume point to calculate Volume. Please set its values")
            self.print_messages()
            exit(0)

    def dominates(self, fit_a, fit_b):
        return all(fit_a <= fit_b) and any(fit_a < fit_b)

    def find_dominates_list(self, fit_matrix=None):
        size = len(fit_matrix)
        list_dominated = np.zeros(size)  # 0: non-dominated, 1: dominated by someone
        for i in range(0, size):
            list_dominated[i] = 0
            for j in range(0, i):
                if any(fit_matrix[i] != fit_matrix[j]):
                    if self.dominates(fit_matrix[i], fit_matrix[j]):
                        list_dominated[j] = 1
                    elif self.dominates(fit_matrix[j], fit_matrix[i]):
                        list_dominated[i] = 1
                        break
                else:
                    list_dominated[j] = 1
                    list_dominated[i] = 1
        return list_dominated

    def get_pareto_front_reference_front(self, pareto_front=None, reference_front=None, metric=None):
        reference_front = self.check_convert_front(reference_front)
        pareto_front = self.check_convert_front(pareto_front)
        if self.reference_front is None:
            if reference_front is None:
                self.messages.append(f'To calculate {metric} you need Reference front')
                self.print_messages()
                exit(0)
            else:
                if self.pareto_front is None:
                    if pareto_front is None:
                        self.messages.append(f'To calculate {metric} you need Pareto front obtained from yor test case')
                        self.print_messages()
                        exit(0)
                    else:
                        return pareto_front, reference_front
                else:
                    if pareto_front is None:
                        return self.pareto_front, reference_front
                    else:
                        return pareto_front, reference_front
        else:
            if reference_front is None:
                if self.pareto_front is None:
                    if pareto_front is None:
                        self.messages.append(f'To calculate {metric} you need Pareto front obtained from yor test case')
                        self.print_messages()
                        exit(0)
                    else:
                        return pareto_front, self.reference_front
                else:
                    if pareto_front is None:
                        return self.pareto_front, self.reference_front
                    else:
                        return pareto_front, self.reference_front
            else:
                if self.pareto_front is None:
                    if pareto_front is None:
                        self.messages.append(f'To calculate {metric} you need Pareto front obtained from yor test case')
                        self.print_messages()
                        exit(0)
                    else:
                        return pareto_front, reference_front
                else:
                    if pareto_front is None:
                        return self.pareto_front, reference_front
                    else:
                        return pareto_front, reference_front

    def find_reference_front(self, solutions=None):     # List of non-dominated solutions
        list_solutions = self.check_convert_front(solutions)
        list_dominated = self.find_dominates_list(list_solutions)
        return list_solutions[where(list_dominated == 0)]

    def find_reference_point(self, solutions=None):     # The maximum single point in all dimensions
        list_solutions = self.check_convert_front(solutions)
        return np.max(list_solutions, axis=0)

    def get_metrics_by_name(self, *func_names):
        temp = []
        for idx, func_name in enumerate(func_names):
            obj = getattr(self, func_name)
            temp.append(obj())
        return temp

    def get_metrics_by_list(self, func_name_list=None, func_para_list=None):
        temp = []
        for idx, func_name in enumerate(func_name_list):
            obj = getattr(self, func_name)
            if func_para_list is None:
                temp.append(obj())
            else:
                if len(func_name_list) != len(func_para_list):
                    print("Failed! Different length between functions and parameters")
                    exit(0)
                temp.append(obj(**func_para_list[idx]))
        return temp

class Metric(Root):
    """
        This class is for: Evaluating Obtained front and True Pareto front
    """

    def __init__(self, pareto_fronts=None, reference_fronts=None, **kwargs):
        super().__init__(pareto_fronts, reference_fronts)
        ## Other parameters
        for key, value in kwargs.items():
            setattr(self, key, value)

    ##
    ## Ratio: Metrics Assessing the Number of Pareto Optimal Solutions in the Set
    ##
    def error_ratio(self, pareto_front=None, reference_front=None):     ## ER function
        pareto_front, reference_front = self.get_pareto_front_reference_front(pareto_front, reference_front, "ER")
        rni = 0
        for point in pareto_front:
            list_flags = [all(point == solution) for solution in reference_front]
            if not any(list_flags):
                rni += 1
        print(f"{rni} - {len(reference_front)}")
        return rni / len(reference_front)

    def overall_non_dominated_vector_generation(self, pareto_front=None, reference_front=None):  ## ONVG function
        pareto_front, reference_front = self.get_pareto_front_reference_front(pareto_front, reference_front, "ONVG")
        rni = 0
        for point in pareto_front:
            list_flags = [all(point == solution) for solution in reference_front]
            if any(list_flags):
                rni += 1
        print(f"===={rni} - {len(reference_front)}")
        return rni

    ##
    ##  Spread : Metrics Concerning Spread of the Solutions
    ##
    def maximum_spread(self, pareto_front=None, reference_front=None):  ## MS function
        """ It addresses the range of objective function values and takes into account the proximity to the true Pareto front"""
        pareto_front, reference_front = self.get_pareto_front_reference_front(pareto_front, reference_front)
        n_objs = reference_front.shape[1]
        pf_max = np.max(pareto_front, axis=0)
        pf_min = np.min(pareto_front, axis=0)
        rf_max = np.max(reference_front, axis=0)
        rf_min = np.min(reference_front, axis=0)
        ms = 0
        for i in range(0, n_objs):
            ms += ((min(pf_max[i], rf_max[i]) - max(pf_min[i], rf_min[i])) / (rf_max[i] - rf_min[i])) ** 2
        return np.sqrt(ms / n_objs)

    ##
    ##  Closeness: Metrics Measuring the Closeness of the Solutions to the True Pareto Front
    ##
    def generational_distance(self, pareto_front=None, reference_front=None):  ## GD function
        pareto_front, reference_front = self.get_pareto_front_reference_front(pareto_front, reference_front, "GD")
        pf_size, rf_size = len(pareto_front), len(reference_front)
        gd = 0
        for i in range(pf_size):
            dist_min = min([np.linalg.norm(pareto_front[i] - reference_front[j]) for j in range(0, rf_size)])
            gd += dist_min ** 2
        return np.sqrt(gd) / pf_size

    def inverted_generational_distance(self, pareto_front=None, reference_front=None):  ## IGD function
        pareto_front, reference_front = self.get_pareto_front_reference_front(pareto_front, reference_front, "IGD")
        pf_size, rf_size = len(pareto_front), len(reference_front)
        igd = 0
        for i in range(rf_size):
            dist_min = min([np.linalg.norm(reference_front[i] - point) for point in pareto_front])
            igd += dist_min ** 2
        return np.sqrt(igd) / rf_size

    def maximum_pareto_front_error(self, pareto_front=None, reference_front=None):  ## MPFE function
        pareto_front, reference_front = self.get_pareto_front_reference_front(pareto_front, reference_front, "IGD")
        pf_size, rf_size = len(pareto_front), len(reference_front)
        mpfe_list = zeros(pf_size)
        for i in range(pf_size):
            dist_min = min([np.linalg.norm(reference_front[i] - point) for point in pareto_front])
            mpfe_list[i] = dist_min
        return max(mpfe_list)

    ##
    ## Distribution: Metrics Focusing on Distribution of the Solutions
    ##
    def spacing(self, pareto_front=None, reference_front=None):  ## S function
        pareto_front, reference_front = self.get_pareto_front_reference_front(pareto_front, reference_front)
        pf_size = len(pareto_front)
        rf_size = len(reference_front)
        size = pf_size
        if rf_size < pf_size:
            size = rf_size
        dist_min_list = zeros(size)
        for i in range(size):
            dist_min = min([np.linalg.norm(pareto_front[i] - reference_front[j]) for j in range(size) if i != j])
            dist_min_list[i] = dist_min
        dist_mean = np.mean(dist_min_list)
        spacing = np.sqrt(sum((dist_min_list - dist_mean) ** 2) / size)
        return spacing

    def spacing_to_extend(self, pareto_front=None, reference_front=None):  ## STE function
        pareto_front, reference_front = self.get_pareto_front_reference_front(pareto_front, reference_front)
        pf_size = len(pareto_front)
        dist_min_list = zeros(pf_size)
        for i in range(pf_size):
            dist_min = min([np.linalg.norm(pareto_front[i] - reference_front[j]) for j in range(pf_size) if i != j])
            dist_min_list[i] = dist_min
        dist_mean = mean(dist_min_list)
        spacing = sum((dist_min_list - dist_mean) ** 2) / (pf_size - 1)

        f_max = np.max(pareto_front, axis=0)
        f_min = np.min(pareto_front, axis=0)
        extent = sum(abs(f_max - f_min))
        ste = spacing / extent
        return ste

    # Function: Hypervolume
    def hv_indicator(self, solution = [], n_objs = 3, ref_point = [], normalize = False):
        if (solution.shape[1] > n_objs):
            sol = solution[:,-n_objs:]
        elif (solution.shape[1] == n_objs):
            sol = np.copy(solution)
        if (normalize == True):
            z_min     = np.min(sol, axis = 0)
            z_max     = np.max(sol, axis = 0)
            sol       = np.clip((sol - z_min)/(z_max - z_min + 0.000000001), 0, 1)
            ref_point = [1]*n_objs
        if (len(ref_point) == 0):
            ref_point = [np.max(sol[:,j]) for j in range(0, sol.shape[1])]
        else:
            for j in range(0, len(ref_point)):
                if (ref_point[j] < np.max(sol[:,j])):
                    print('Reference Point is Invalid: Outside Boundary')
                    print('Correcting Position', j, '; Reference Point Value', ref_point[j], 'was changed to', np.max(sol[:,j]))
                    print('')
                    ref_point[j] = np.max(sol[:,j])
        hv = HyperVolume(ref_point)
        volume = hv.compute(sol)
        return volume

    # Ratio: Metrics Assessing the Number of Pareto Optimal Solutions in the Set
    ER = error_ratio
    ONVG = overall_non_dominated_vector_generation

    # Spread : Metrics Concerning Spread of the Solutions
    MS = maximum_spread

    # Closeness: Metrics Measuring the Closeness of the Solutions to the True Pareto Front
    GD = generational_distance
    IGD = inverted_generational_distance
    MPFE = maximum_pareto_front_error

    # Distribution: Metrics Focusing on Distribution of the Solutions
    S = spacing
    STE = spacing_to_extend

#-----------    
    
    
def normalize(distances):
    """Safely normalize an array while handling NaN, Inf, and zero variance cases."""
    distances = np.array(distances, dtype=np.float64)  # Ensure floating-point precision
    
    # Remove NaN values (if all are NaN, return zeros)
    if np.all(np.isnan(distances)):
        return np.zeros_like(distances)
    
    # Replace Infs with the max finite value
    finite_array = distances[np.isfinite(distances)]
    if finite_array.size > 0:
        max_finite = np.max(finite_array)
        distances[np.isinf(distances)] = max_finite
    else:
        return np.zeros_like(distances)  # If all values are Inf, return zeros

    # Compute mean and standard deviation safely
    mean = np.nanmean(distances)  # Use nanmean to ignore NaNs
    std = np.nanstd(distances, ddof=1)  # Use nanstd to ignore NaNs

    if std == 0 or np.isnan(std):  # Avoid division by zero or NaN cases
        return np.zeros_like(distances)

    return (distances - mean) / (std + 1e-10)

def MinimalDistance(X, Y):
    """Compute the minimal Euclidean distance from X to the closest point in Y."""
    min_dist = float('inf')
    for y in Y:
        dist = np.linalg.norm(np.array(X) - np.array(y))
        if dist < min_dist:
            min_dist = dist
    return min_dist

def CrowdingDistance(front):
    """Compute the crowding distance for a population based on decision variables."""
    num_solutions = len(front)
    num_objectives = len(front[0])

    distances = np.zeros(num_solutions)

    for m in range(num_objectives):
        sorted_indices = np.argsort([sol[m] for sol in front])
        distances[sorted_indices[0]] = distances[sorted_indices[-1]] = float('inf')

        min_val = front[sorted_indices[0]][m]
        max_val = front[sorted_indices[-1]][m]

        if max_val == min_val:
            continue  # Avoid division by zero

        for i in range(1, num_solutions - 1):
            distances[sorted_indices[i]] += (
                front[sorted_indices[i + 1]][m] - front[sorted_indices[i - 1]][m]
            ) / (max_val - min_val)

    return distances

def MultiObjectiveDifference(candidate_objs, previous_objs):
    """
    Compute the minimal Euclidean distance between each candidate's two-objective values
    and previously evaluated solutions.
    """
    num_candidates = len(candidate_objs)
    
    differences = np.zeros(num_candidates)

    for i in range(num_candidates):
        min_diff = float('inf')
        for prev in previous_objs:
            diff = np.linalg.norm(np.array(candidate_objs[i]) - np.array(prev))
            min_diff = min(min_diff, diff)
        differences[i] = min_diff

    return differences

def SelectCandidate(x_seed, solution_set, nsgaIII_sols, chroms_obj_record):
    """
    Selects a candidate balancing:
    - Minimal distance in decision space.
    - Crowding distance in decision space.
    - Objective value difference in objective space.
    """
    
    # Perform random selection with a specified probability
    if np.random.rand() < 0.1:
        selected_index = np.random.randint(len(x_seed))
        return x_seed[selected_index]

    # Compute minimal distances in decision space
    min_distances = np.array([MinimalDistance(x, solution_set) for x in x_seed])

    # Compute crowding distances in decision space
    crowding_distances = CrowdingDistance(x_seed)

    # Compute multi-objective differences in objective space
    obj_differences = MultiObjectiveDifference(nsgaIII_sols, chroms_obj_record)

    # **Normalize all computed metrics to prevent numerical issues**
    min_distances = normalize(min_distances)
    crowding_distances = normalize(crowding_distances)
    obj_differences = normalize(obj_differences)

    # Weighted combination of scores (50% minimal distance, 50% objective difference)
    final_scores = 0.5 * min_distances + 0.5 * obj_differences
    selected_index = np.argmax(final_scores)
       
    return x_seed[selected_index]

def is_dominated(solution, solutions):
    epsilon=1e-9
    for other in solutions:
        if all(o <= s + epsilon for o, s in zip(other, solution)) and any(o < s - epsilon for o, s in zip(other, solution)):
            return True
    return False

def find_pareto_front(decision_solutions, objective_solutions):
    """
    Extract the Pareto front from a list of multi-objective solutions and return
    both the Pareto-optimal objective values and their corresponding decision variables.
    """
    
    decision_solutions = list(map(list, decision_solutions))
    objective_solutions = list(map(list, objective_solutions))
    
    # Ensure unique solutions
    unique_obj_solutions = list(map(list, set(map(tuple, objective_solutions))))
    
    # Find Pareto-optimal objective values
    pareto_front = [sol for sol in unique_obj_solutions if not is_dominated(sol, unique_obj_solutions)]
    
    # Find corresponding decision variables
    pareto_decision_solutions = [
        decision_solutions[objective_solutions.index(sol)] for sol in pareto_front
    ]
    
    return np.array(pareto_front), np.array(pareto_decision_solutions)
    
def generate_random_solution(lower_bound, upper_bound):
    return [random.uniform(lower, upper) for lower, upper in zip(lower_bound, upper_bound)]
    
    
def lhs_pruning(Xsamples, y_f, num_samples):
    """
    Perform LHS-based pruning to maintain a small but representative dataset.
    
    Parameters:
        Xsamples (np.array): Decision space samples.
        y_f1 (np.array): Objective function 1 values.
        y_f2 (np.array): Objective function 2 values.
        num_samples (int): Number of representative samples to retain.
    
    Returns:
        pruned_X (np.array): Pruned decision space samples.
        pruned_y_f1 (np.array): Pruned objective function 1 values.
        pruned_y_f2 (np.array): Pruned objective function 2 values.
    """
    # Normalize the objective space
    y_min, y_max = y_f.min(axis=0), y_f.max(axis=0)
    y_norm = (y_f - y_min) / (y_max - y_min)

    # Define proper LHS sampling space using skopt.space.Real
    space = skopt.space.Space([skopt.space.Real(0.0, 1.0) for _ in range(y_f.shape[1])])  # Dynamically create space based on number of objectives

    # Generate LHS samples in the objective space
    lhs_sampler = Sobol(randomize=False)
    lhs_samples = lhs_sampler.generate(space.dimensions, num_samples)

    # Select the closest points from the original dataset
    selected_indices = []
    for lhs_point in lhs_samples:
        distances = cdist([lhs_point], y_norm)
        closest_idx = np.argmin(distances)
        selected_indices.append(closest_idx)

    selected_indices = list(set(selected_indices))  # Remove duplicates

    # Prune the samples based on the selected indices
    pruned_X = Xsamples[selected_indices]
    pruned_y_f = y_f[selected_indices]
    return pruned_X, pruned_y_f
    
def is_similar(sol1, sol2, tol):
    # Ensure both solutions have the same length
    if len(sol1) != len(sol2):
        raise ValueError("Solutions must have the same length")
    
    # Iterate over the pairs of elements from sol1 and sol2
    for a, b in zip(sol1, sol2):
        # Calculate the relative difference and compare with tolerance
        if abs(a - b) / max(abs(a), abs(b), 1) >= tol:
            return False
    
    # If no differences exceed the tolerance, return True
    return True

def found_in_memory(solution, solutions):
    """
    Check if a solution has already been simulated.
    """     
    found_in_memory = False
    similar_idx = -1

    for idx, sol in enumerate(solutions):
        found_in_memory = is_similar(solution, sol, 1E-6)
        if found_in_memory:
            similar_idx = idx
            break
    return found_in_memory
    
def gauss_mutation(solution):
    size = len(solution)
    mu = 0.0 * size
    # Parameters obtained via experimentation based on LHS
    sigma = 0.32531251 
    prob = 0.629609672  

    mu = repeat(mu, size)
    sigma = repeat(sigma, size)

    mutated_sol = solution[:]
    for i, m, s in zip(range(size), mu, sigma):
        if random.random() < prob:
            mutated_sol[i] += random.gauss(m, s)
            
    for i in range(len(mutated_sol)):
        if mutated_sol[i] < lower_bound[i]:
            mutated_sol[i] = lower_bound[i]
        if mutated_sol[i] > upper_bound[i]:
            mutated_sol[i] = upper_bound[i]
    return mutated_sol

def ensure_capacity(pareto_length):
    global pareto_decision_vars, pareto_front

    current_capacity = pareto_decision_vars.shape[0]

    if pareto_length >= current_capacity:
        # Increase capacity (double current size)
        new_capacity = max(pareto_length + 1, current_capacity * 2)

        # Create larger arrays
        new_pareto_decision_vars = np.empty((new_capacity, num_variables))
        new_pareto_front = np.empty((new_capacity, NObj))

        # Copy old data
        new_pareto_decision_vars[:pareto_length] = pareto_decision_vars[:pareto_length]
        new_pareto_front[:pareto_length] = pareto_front[:pareto_length]

        # Assign new arrays
        pareto_decision_vars = new_pareto_decision_vars
        pareto_front = new_pareto_front
        
        
def ensure_capacity(pareto_length):
    global pareto_decision_vars, pareto_front
    
    current_capacity = pareto_decision_vars.shape[0]

    if pareto_length >= current_capacity:
        new_capacity = max(pareto_length + 1, int(current_capacity * 1.5))  # Smarter growth

        pareto_decision_vars.resize((new_capacity, num_variables), refcheck=False)
        pareto_front.resize((new_capacity, NObj), refcheck=False)
        
def round_up_iters(n):
    return math.ceil(n / 100) * 100

def crossover(dad_chromosome, mom_chromosome, lower_bound, upper_bound):
    mu = 0.9 
    sigma = 0.1
    
    son = [(dad_chromosome[i] + mom_chromosome[i])/2.0 for i in range(len(dad_chromosome))]
    
    son = [son[i]+sigma*(random.random()) if random.random() <= mu else son[i] for i in range(len(son))]
    son = [son[i]-sigma*(random.random()) if random.random() <= mu else son[i] for i in range(len(son))]
    
    for i in range(len(son)):
        if son[i] < lower_bound[i]:
            son[i] = lower_bound[i]
        if son[i] > upper_bound[i]:
            son[i] = upper_bound[i]

    return son


run_best_solution_array = []
run_best_front_array = []
run_gd_array = []
run_front_error_array = []
run_max_spread_array = []
run_igd_array = []
run_spacing_array = []
run_front_hypervolume_array = []
run_reference_hypervolume_array = []
run_comp_time_array = []
run_act_time_array = []
run_simulations_array = []
run_number_of_pareto_sols = []

for run in range(number_of_runs):

    for i in range(5):
        # Use time as a changing value for the seed
        random.seed(time.time() + i)

    total_simulation_times = 0
    lower_bound = initial_lower_bound
    upper_bound = initial_upper_bound
    population_size = initial_population_size
    
    
    space = Space([(0.0, 1.0),]*num_variables)
    dimensions = space.dimensions

    Xmemory = []
    y_f1_memory = []
    y_f2_memory = []

    # Sobol sampling 
    sobol = Sobol(randomize=False)
    Xsamples = sobol.generate(space.dimensions, n_samples)

    for sample in Xsamples:
        for sample_var in range(num_variables):
            sample[sample_var] *= (upper_bound[sample_var] - lower_bound[sample_var])

    for sample in Xsamples:
        for sample_var in range(num_variables):
            sample[sample_var] += lower_bound[sample_var]
            
    
    tspace = TargetSpace(objective, 2, PB,
                         [],
                         None,
                         False)

    if max_simulation_times > tspace._n_alloc_rows:
        tspace._allocate(max_simulation_times)
                         



    obj_func_values = np.zeros((len(Xsamples), NObj)) 
    for i in range(0, len(Xsamples)):
        print("Simulation started ...")
        obj_func_values[i] = objective(Xsamples[i])
        total_simulation_times += 1
        
    for i in range(len(Xsamples)):
        dummy = tspace.observe_point_and_response(np.array(Xsamples[i]), obj_func_values[i])
        
        

    # Initial surrogate
    y_f1 = np.array(obj_func_values[:, 0])
    y_f2 = np.array(obj_func_values[:, 1])

    y_f1_memory = y_f1[:]
    y_f2_memory = y_f2[:]

    # Kernel parameters
    l = 0.1
    sigma_f = 2
    kernel = ConstantKernel(constant_value=1,constant_value_bounds =(1e3,1e6)) * RBF(length_scale=100.0, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=1, 
        noise_level_bounds=(1e-15, 1e+1))

    
    level0_f1 = list()
    level0_f1.append(('svr_f1', SVR(kernel=1.41**2 * RBF(length_scale=0.1))))
    level0_f1.append(('gb_f1', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)))
    
    # define the meta learner model
    level1_f1 = LinearRegression()
    
    # define the stacking ensemble
    #ensemble_model_f1 = StackingRegressor(estimators=level0_f1, final_estimator=level1_f1)
    # Gaussian process with a larger number of max iters
    ensemble_model_f1 = GaussianProcessRegressor(kernel=Matern(nu=1.5), n_restarts_optimizer=100)
    ensemble_model_f1.fit(Xsamples, y_f1.ravel())
            

    model_f1_score = ensemble_model_f1.score(Xsamples, y_f1)

                                
    print("model_f1_score: ", round(model_f1_score,6))
    
    level0_f2 = list()
    level0_f2.append(('svr_f2', SVR(kernel=1.41**2 * RBF(length_scale=0.1))))
    level0_f2.append(('gb_f2', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)))

    # define the meta learner model
    level1_f2 = LinearRegression()
    
    # define the stacking ensemble
    #ensemble_model_f2 = StackingRegressor(estimators=level0_f2, final_estimator=level1_f2, passthrough=False)
    ensemble_model_f2 = GaussianProcessRegressor(kernel=Matern(nu=1.5), n_restarts_optimizer=100)
    ensemble_model_f2.fit(Xsamples, y_f2.ravel())
    
    model_f2_score = ensemble_model_f2.score(Xsamples, y_f2)
    print("model_f2_score: ", round(model_f2_score,6))
        
    # For recording data of each epoch
    best_solution_array = []
    best_front_array = []
    gd_array = []
    front_error_array = []
    max_spread_array = []
    igd_array = []
    spacing_array = []
    front_hypervolume_array = []
    reference_hypervolume_array = []
    comp_time_array = []
    act_time_array  = []
    simulations_array = []
    number_of_pareto_sols = []

    cross1_times_total_array = []
    cross2_times_total_array = []
    cross3_times_total_array = []
    cross4_times_total_array = []
    mut_times_total_array = []

    #total_simulation_times = 0
    total_comp_time = 0
    total_act_time = 0


    #obj_space = (tspace.f).tolist()
    #pareto_front = [sol for sol in obj_space if not is_dominated(sol, obj_space)]

    pareto_length = 0
    pareto_front = None
    pareto_decision_vars = None
    
    initial_capacity = 0  # Adjust based on problem size
    pareto_decision_vars = np.empty((initial_capacity, num_variables))
    pareto_front = np.empty((initial_capacity, NObj))

    
    epoch = 0
    while total_simulation_times <= max_simulation_times:
        #population_size += 5
            
        cross1_times_total = 0
        cross2_times_total = 0
        cross3_times_total = 0
        cross4_times_total = 0
        mut_times_total = 0

        # Convergence file headers
        data = []
        #now = datetime.now()
        #date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
        

        simulation_times = 0
        new_sols = []
        pareto = []
        #pareto_front = []
        pareto_mut = []
        iter_no = 0
        evaluator = Metric()
        referencePoint = [1, 1]
        #hv = HyperVolume(referencePoint)
        current_volume = 0
        volume = 0
        best_vol = -55555555555555555555555555.55
        #start_act_time = current_milli_time()
        start_act_time = time.perf_counter()
        
        initial_upper_bound = upper_bound
        initial_lower_bound = lower_bound

        
        start_time = time.process_time()
        while(iter_no < max_iter):
            cross1_times = 0
            cross2_times = 0
            cross3_times = 0
            cross4_times = 0
            mut_times = 0
                       
            
            pl_problem = Problem(num_variables, 2)
            pl_problem.types[:] = [Real(lower, upper) for lower, upper in zip(lower_bound, upper_bound)]
            pl_problem.function = platypus_obj_func # surrogate objective functions 
            
            algorithm = NSGAIII(problem= pl_problem, divisions_outer=20)
            algorithm.run(round_up_iters(200*num_variables+600)) # 1200
            
            # Surrogate pareto front generated with NSGAIII
            nondominated_solutions = core.nondominated(algorithm.result)                                                                        
            pop = [pl_sol.variables for pl_sol in nondominated_solutions]
            Population = np.asarray(pop)


            front = [pl_sol.objectives for pl_sol in nondominated_solutions]
            front = np.array(front)
                      
            # Select a candidate solution from the surrogate Pareto front
            selected_x = SelectCandidate(pop, tspace.x, front, tspace.f)

            sol_found = False
            sol_found = found_in_memory(selected_x, tspace.x)
            if sol_found:
                print("Solution found in memory")
                selected_x = gauss_mutation(selected_x)
                
            
            response = objective(selected_x)
            total_simulation_times += 1
            simulation_times += 1
            dummy = tspace.observe_point_and_response(np.array(selected_x), objective(selected_x))
            print("Added: ", dummy, response)
            
            if random.random() < 0.07:
                new_point = generate_random_solution(lower_bound, upper_bound)

                sol_found = False
                sol_found = found_in_memory(new_point, tspace.x)
                if sol_found:
                    print("Solution found in memory")
                    new_point = gauss_mutation(new_point)
                new_point = crossover(selected_x, new_point, lower_bound, upper_bound)
                response = objective(new_point)
                total_simulation_times += 1
                simulation_times += 1
                dummy = tspace.observe_point_and_response(np.array(new_point), objective(new_point))
                print("New point with crossover")
                print("Added: ", dummy, response)

            new_pareto_front, new_pareto_decision_vars = find_pareto_front(tspace.x, tspace.f)
            pareto_length = new_pareto_front.shape[0]
            
            if random.random() < 0.01:
                new_point = gauss_mutation(selected_x)
                response = objective(new_point)
                total_simulation_times += 1
                simulation_times += 1
                dummy = tspace.observe_point_and_response(np.array(new_point), objective(new_point))
                print("New point mutation")
                print("Added: ", dummy, response)


            # Allocate memory for storing the Pareto solutions
            ensure_capacity(pareto_length)

            pareto_decision_vars[:pareto_length] = new_pareto_decision_vars
            pareto_front[:pareto_length] = new_pareto_front
            
            #y_Pareto, x_Pareto = tspace.ParetoSet()
            
            del ensemble_model_f1
            ensemble_model_f1 = GaussianProcessRegressor(kernel=Matern(nu=1.5), alpha = 0.0001, n_restarts_optimizer=5)
            del ensemble_model_f2
            ensemble_model_f2 = GaussianProcessRegressor(kernel=Matern(nu=1.5), alpha = 0.0001, n_restarts_optimizer=5)
            
            iter_no = iter_no + 1
            
            print("Run: ", run)
            print("Epoch: ", epoch)
            print("Iteration: ", iter_no)
            print("----------------------------------------------------")
            

            ensemble_model_f1.fit(tspace.x, tspace.f[:, 0])
            ensemble_model_f2.fit(tspace.x, tspace.f[:, 1])
 
            print("Size of solution space: ", len(tspace.x))
            
            print("Size of Pareto front: ", pareto_length)
            
            if len(pareto_decision_vars) > 10:
                # Reducing the search space based on Pareto front
                d = np.array(upper_bound) - np.array(lower_bound) 
                alpha = 0.7 
                
                max_values = np.array(pareto_decision_vars).max(axis = 0)
                max_values = np.array(max_values) + alpha*d/2.0
                min_values = np.array(pareto_decision_vars).min(axis = 0)
                min_values = np.array(min_values) - alpha*d/2.0
                #print("max and min values: ", max_values, min_values)
                for var_index in range(num_variables):
                    upper_bound[var_index] = min(max_values[var_index], upper_bound[var_index])
                    lower_bound[var_index] = max(min_values[var_index], lower_bound[var_index])

                #print("New boundaries: ", upper_bound, lower_bound)
       
            
            if total_simulation_times >= max_simulation_times:
                print("total simulation times:", total_simulation_times)
                break


        # Stop timer 
        end_time = time.perf_counter()
        elapsed_time = (end_time - start_act_time) * 1000.0 # time in miliseconds

        #ms = (process_time() - start_time) * 1000.0
        ms = elapsed_time

        model_f1_score = ensemble_model_f1.score(tspace.x, tspace.f[:, 0])
        model_f2_score = ensemble_model_f2.score(tspace.x, tspace.f[:, 1]) 
        print("*** Ensemble Model score f1:", model_f1_score)
        print("*** Ensemble Model score f2:", model_f2_score)

        # Remove extra unused rows (as part of the memory allocation process) after the loop ends
        front = pareto_front[:pareto_length]
                    # Remove extra unused rows (as part of the memory allocation process) after the loop ends
        front = pareto_front[:pareto_length]
        best_x = pareto_decision_vars[:pareto_length]
            
        print("Front: ", front)
        
        # Force garbage collection
        gc.collect()
        


        with open(benchmark) as fp:
            reference = np.array([list(map(float, line.strip().split())) for line in fp])    



        front_array = evaluator.check_convert_front(front)
        reference_array = evaluator.check_convert_front(reference)
        gd = evaluator.generational_distance(front_array, reference_array)
        print("Generational distance: ", gd)
        error = evaluator.error_ratio(front_array, reference_array)
        print("Error: ", error)
        max_spread = evaluator.maximum_spread(front_array, reference_array)
        print("Maximum spread:", max_spread)
        igd = evaluator.inverted_generational_distance(front_array, reference_array)
        print("IGD:", igd)
        #spacing_measure = evaluator.spacing(front_array, reference_array)
        spacing_measure = 0
        print("Spacing:", spacing_measure)
        hyp_vol = evaluator.hv_indicator(solution = front_array, n_objs = 2, ref_point = referencePoint, normalize = True)
        print("Hypervolume: ", hyp_vol)
        ref_vol = evaluator.hv_indicator(solution = reference_array, n_objs = 2, ref_point = referencePoint, normalize = True)
        print("Reference hypervolume: ", ref_vol)

        best_x_as_list = [arr.tolist() if isinstance(arr, np.ndarray) else arr for arr in best_x]                            
        best_solution_array.append(best_x_as_list)
        #best_solution_array.append(best_x)
        front = [
            [float(element) if isinstance(element, np.floating) else element for element in inner_arr]
            for inner_arr in front
            ]
        best_front_array.append(front)
        gd_array.append(gd)
        front_error_array.append(error)
        max_spread_array.append(max_spread)
        igd_array.append(igd)
        spacing_array.append(spacing_measure)
        front_hypervolume_array.append(hyp_vol)
        reference_hypervolume_array.append(ref_vol)
        comp_time_array.append(ms)
        act_time_array.append(elapsed_time)
        simulations_array.append(simulation_times)
        number_of_pareto_sols.append(pareto_length)

        cross1_times_total_array.append(cross1_times_total)
        cross2_times_total_array.append(cross2_times_total)
        cross3_times_total_array.append(cross3_times_total)
        cross4_times_total_array.append(cross4_times_total)
        mut_times_total_array.append(mut_times_total)
        
        #total_simulation_times += simulation_times
        total_comp_time += ms
        total_act_time += elapsed_time
        epoch += 1
        

    print("front: ", front)
    
    best_x_as_list = [arr.tolist() if isinstance(arr, np.ndarray) else arr for arr in best_x]
    run_best_solution_array.append(best_x_as_list)
    front = [
        [float(element) if isinstance(element, np.floating) else element for element in inner_arr]
        for inner_arr in front
        ]
    run_best_front_array.append(front)        
    run_gd_array.append(gd)
    run_front_error_array.append(error)
    run_max_spread_array.append(max_spread)
    run_igd_array.append(igd)
    run_spacing_array.append(spacing_measure)
    run_front_hypervolume_array.append(hyp_vol)
    run_reference_hypervolume_array.append(ref_vol)
    run_comp_time_array.append(total_comp_time)
    run_act_time_array.append(total_act_time)
    run_simulations_array.append(total_simulation_times)
    run_number_of_pareto_sols.append(pareto_length)
    



    print("Total number of simulation times: ", total_simulation_times)
    function1_values = []
    function2_values = []
    

    function1_vals = [row[0] for row in front]
    function2_vals = [row[1] for row in front]
    
    del front
    del best_x
    # Force garbage collection
    gc.collect()

    now = datetime.now()
    date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
    fig=plt.figure()
    plt.xlabel('Function 1', fontsize=15)
    plt.ylabel('Function 2', fontsize=15)
    #plt.scatter(real_y1, real_y2, s=10, c='b', marker="s", label='real optimum')
    plt.xlabel('Function 1', fontsize=15)
    plt.ylabel('Function 2', fontsize=15)
    #plt.plot(real_y1, real_y2, label='real optimum')
    plt.scatter(reference[:, 0], reference[:, 1], c = 'black', s = 2,  marker = 's', label='real optimum')
    plt.scatter(function1_vals, function2_vals, c = 'red',   s = 25, marker = 'o', label='mo-mevo optimum')
    plt.legend(loc = 'upper right')
    #plt.show()
    plot_file_name = "run_" + str(run) + "_pareto_" + date_time + ".png"
    fig.savefig(plot_file_name,dpi=300)
    fig.clear()
    
    avs,afvs  = optimize(spea2_obj_func,num_variables,lower_bound,upper_bound,150,100,100,0.1,prnt_msg=0,savedata=0)
    ax1 = plt.figure()
    plt.xlabel('Function 1', fontsize = 12)
    plt.ylabel('Function 2', fontsize = 12)
    plt.scatter(afvs[:,0], afvs[:,1], c = 'red',   s = 25, marker = 'o', label = 'SPEA-2 with surrogate')
    plt.scatter(reference[:, 0], reference[:, 1], c = 'black', s = 2,  marker = 's', label='real optimum')
    plt.legend(loc = 'upper right')
    plot_file_name = "run_" + str(run) + "_surrogate_pareto_" + date_time + ".png"
    ax1.savefig(plot_file_name,dpi=150)
    ax1.clear() 

    # save the results of all the epochs
    now = datetime.now()
    date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
    df = pd.DataFrame()
    df['Solution'] = pd.Series(best_solution_array)
    df['Pareto front'] = pd.Series(best_front_array)
    df['GD'] = pd.Series(gd_array)
    df['Error'] = pd.Series(front_error_array)
    df['Max spread'] = pd.Series(max_spread_array)
    df['IGD'] = pd.Series(igd_array)
    df['Spacing'] = pd.Series(spacing_array)
    df['Hypervolume'] = pd.Series(front_hypervolume_array)
    df['Ref hypervolume'] = pd.Series(reference_hypervolume_array)
    df['Comp time'] = pd.Series(comp_time_array)
    df['Elapsed time'] = pd.Series(act_time_array)
    df['Simulations'] = pd.Series(simulations_array)
    df['Pareto size'] = pd.Series(number_of_pareto_sols)
    df['Cross1'] = pd.Series(cross1_times_total_array)
    df['Cross2'] = pd.Series(cross2_times_total_array)
    df['Cross3'] = pd.Series(cross3_times_total_array)
    df['Cross4'] = pd.Series(cross4_times_total_array)
    df['Mut'] = pd.Series(mut_times_total_array)
    epochs_file_name = "run_" + str(run) + "_epoch_" + date_time + ".csv"
    df.to_csv(epochs_file_name)
    
    del ensemble_model_f1
    del ensemble_model_f2
    # Force garbage collection
    gc.collect()

# Save the results of all the runs
now = datetime.now()
date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
df = pd.DataFrame()
df['Solution'] = pd.Series(run_best_solution_array)
df['Pareto front'] = pd.Series(run_best_front_array)
df['GD'] = pd.Series(run_gd_array)
df['Error'] = pd.Series(run_front_error_array)
df['Max spread'] = pd.Series(run_max_spread_array)
df['IGD'] = pd.Series(run_igd_array)
df['Spacing'] = pd.Series(run_spacing_array)
df['Hypervolume'] = pd.Series(run_front_hypervolume_array)
df['Ref hypervolume'] = pd.Series(run_reference_hypervolume_array)
df['Comp time'] = pd.Series(run_comp_time_array)
df['Elapsed time'] = pd.Series(run_act_time_array)
df['Simulations'] = pd.Series(run_simulations_array)
df['Pareto size'] = pd.Series(run_number_of_pareto_sols)
runs_file_name = "runs_" + date_time + ".csv"
df.to_csv(runs_file_name)