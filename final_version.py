#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: AHeckman118, JerryWu1996
"""

import numpy as np
import copy as cp
import matplotlib.pyplot as plt

'''
compares the two estimator functions at a given table size nxk with 
probability of flipping bits e and std dev sigma.
'''
def estimator_compare(n, k, e, sigma):
    #generate the table and perturb it for RR
    table = np.random.randint(2, size = (n,k))
    noisy_table = noise_generator(n, k, table, e)
    #initialize the sum arrays
    true_sums = []
    rr_sums = []
    lap_sums = []
    #laplace noise to perturb the sums
    lap = np.random.laplace(0, sigma, size=k)
    for j in range(k):
        true_sums += [column_sum(table, j)]
        rr_sums += [column_sum(noisy_table, j)]
        lap_sums += [column_sum(table, j)+int(lap[0])]
    #generate guesses for rr and laplace estimators
    rr_guess = rr_estimator(rr_sums, n, e)
    lap_guess = lap_estimator(lap_sums, sigma)
    #calculate the proportional error and return the average
    rr_err = []
    lap_err = []
    for i in range(k):
        rr_err += [(abs(rr_guess[i] - true_sums[i]))/n]
        lap_err += [(abs(lap_guess[i] - true_sums[i]))/n]
    return [np.mean(rr_err), np.mean(lap_err)]

'''
compare how the tracing attacks perform on each mechanism using a nxk table,
bit flipping probability e, over m guesses
'''
def tracing_compare(n, k, e, m):
    #list of sigmas, could be expanded for better granularity.
    sigma = [.5, 5, 20, 100, 500, 1000, 5000, 10000]
    #generate table and perturb it for RR
    table = np.random.randint(2, size = (n,k))
    noisy_table = noise_generator(n, k, table, e)
    #set sigma such that the accuracy of each estimation is around the same.
    i = 1
    errors = estimator_compare(n, k, e, sigma[0])
    while(errors[0] >= errors[1]):
        errors = estimator_compare(n, k, e, sigma[i])
        i += 1
    sigma = sigma[i]
    #get vector of laplace noisy sums
    lap = np.random.laplace(0, sigma, size=k)
    lap_sums = []
    for i in range(k):
        lap_sums += [column_sum(table, i)+int(lap[0])]
    #result = 1 if the attacker thinks the data is in the dataset and 0 if not.
    rr_result = 0
    lap_result = 0
    #set the thresholds at which the attacker will say data is in the dataset.
    rr_thresh = n/k**2
    lap_thresh = sigma + lap_attacker(lap_sums, get_guess(table))
    #initialize the performance statistics.
    rr_true_positive = 0
    rr_false_positive = 0
    rr_true_negative = 0
    rr_false_negative = 0
    lap_true_positive = 0
    lap_false_positive = 0
    lap_true_negative = 0
    lap_false_negative = 0
    #generate m guesses and run them through each attacker.  Label the result accordingly
    for i in range(m):
        guess = get_guess(table)
        conf = lap_attacker(lap_sums, guess)
        if (rr_attacker(noisy_table, guess) <= rr_thresh):
            rr_result = 1
        if (conf <= lap_thresh):
            lap_result = 1
        if(rr_result == 1 and guess.tolist() in table.tolist()):
            rr_true_positive += 1
        elif(rr_result == 1 and guess.tolist() not in table.tolist()):
            rr_false_positive += 1
        elif(rr_result == 0 and guess.tolist() not in table.tolist()):
            rr_true_negative += 1
        elif(rr_result == 0 and guess.tolist() in table.tolist()):
            rr_false_negative += 1
        if(lap_result == 1 and guess.tolist() in table.tolist()):
            lap_true_positive += 1
        elif(lap_result == 1 and guess.tolist() not in table.tolist()):
            lap_false_positive += 1
        elif(lap_result == 0 and guess.tolist() not in table.tolist()):
            lap_true_negative += 1
        elif(lap_result == 0 and guess.tolist() in table.tolist()):
            lap_false_negative += 1
        rr_result = 0
        lap_result = 0
    #calculate tpr and fpr for each mechanism
    rr_tpr = rr_true_positive/(rr_true_positive + rr_false_negative)
    rr_fpr = rr_false_positive/(rr_false_positive + rr_true_negative)
    lap_tpr = lap_true_positive/(lap_true_positive + lap_false_negative)
    lap_fpr = lap_false_positive/(lap_false_positive + lap_true_negative)
    return [rr_tpr, rr_fpr, lap_tpr, lap_fpr]

'''
run tracing_compare() on m nxk tables and plot the 
average success at different levels of accuracy
'''
def plot_tracers(n, k, m):
    #set different levels of e.  cannot include .5
    e = [.05, .1, .15, .2, .25, .3, .35, .4]
    #initialize arrays for each stat
    rr_tprs = []
    rr_fprs = []
    lap_tprs = []
    lap_fprs = []
    for i in range(len(e)):
        results = []
        arr = []
        #get the average of m runs of each level of e.
        for j in range(m):
            arr += [tracing_compare(n, k, e[i], 10)]
        results = np.mean(arr, axis=0)
        arr = []
        rr_tprs += [results[0]]
        rr_fprs += [results[1]]
        lap_tprs += [results[2]]
        lap_fprs += [results[3]]
    #plot the results
    plt.plot(e, np.divide(rr_tprs, rr_fprs), 'r')
    plt.plot(e, np.divide(lap_tprs, lap_fprs), 'b')
    plt.ylim(0)
    plt.xlabel("e")
    plt.ylabel("tpr/fpr for tracing attacks")
    plt.show()

'''
either return an entry in table, or generate a randomized row using the same 
method that constructed table.
'''
def get_guess(table):
    if np.random.randint(2) == 1:
        guess = table[np.random.randint(len(table))]
        return guess
    else:
        fake = np.random.randint(2, size = (len(table),len(table[0])))
        guess = fake[np.random.randint(len(fake))]
        return guess

'''
returns the minimum diffference between the given guess and every row in table
'''
def rr_attacker(table, guess):
    n = len(table)
    dist = []
    for i in range(n):
        dist += [difference(guess, table[i])]
    return min(dist)

'''
returns the inner product of two vectors.  This is a separate function because
there could be better methods of performing a good attack on a laplace noisy
data set.
'''
def lap_attacker(sums, guess):
    x = np.inner(sums, guess)
    return x

'''
plots the error in estimation as e and sigma increase in an nxk table
'''
def estimator_plot(n, k):
    #initialize values, create the values for e and sigma
    e = [0, .1, .2, .3, .4]
    sig = [0, .5, 10, 100, 500]
    rr_accs = []
    lap_accs = []
    for i in range(len(e)):
        #compare the estimator for each value of e and sigma
        results = estimator_compare(n, k, e[i], sig[i])
        rr_accs += [results[0]]
        lap_accs += [results[1]]
    #plot the results
    plt.plot(e, rr_accs)
    plt.title("Estimator Accuracy as Noise Increases")
    plt.xlabel("% chance of flipping bit")
    plt.ylabel("Error")
    plt.show()
    plt.plot(sig, lap_accs)
    plt.xlabel("Sigma for laplace noise")
    plt.ylabel("Error")
    plt.show()

'''
create a noisy table using RR
'''
def noise_generator(n, k, base_table, e):
    table = cp.deepcopy(base_table)
    for i in range(len(table)):
        for j in range(len(table[i])):
            flip = np.random.random()
            if (flip < e):
                table[i][j] = bit_flip(table[i][j])
    return table

'''
flip a bit(probably a numpy function that does this)
'''
def bit_flip(i):
    if (i == 0):
        return 1
    else:
        return 0

'''
finds the sum for a column(definitely a numpy function for this)
'''
def column_sum(table, y):
    freq = 0
    for i in range(len(table)):
        freq += table[i][y]
    return freq

'''
estimates true column sums for a table of length n with a bit flipping
probability e, using the noisy sums
'''
def rr_estimator(sums, n, e):
    guess = []
    for i in range(len(sums)):
        guess += [(sums[i]+n*e)/(1-2*e)]
    return guess

'''
takes sigma and generates laplace noise, which it uses to try and counteract
the noise added by our mechanism to get an estimation.
'''
def lap_estimator(lap_sums, sigma):
    lap = np.random.laplace(0, sigma, size=len(lap_sums))
    for i in range(len(lap_sums)):
        lap_sums[i] = lap_sums[i] - lap[i]
    return lap_sums

'''
finds how different two rows are in terms of how many locations they differ in.
eg: [1, 0, 1] and [0, 1, 1] would have difference() = 2.
'''
def difference(x, y):
    diff = 0
    for i in range(len(x)):
        if x[i] != y[i]:
            diff += 1
    return diff
