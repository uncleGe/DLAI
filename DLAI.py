"""
1st attempt at optimizing black-box function (f) using
OpenAI's Evolution Strategies (ES) to evolve a
Dynamically Learning Artificial Intelligence (DLAI) to run DLA,
where the parameter distribution is a gaussian of fixed standard deviation.
"""
import numpy as np
import time

global max_counter
global offset
global nrules
global start
global time_elapsed

start = time.time()
np.random.seed(0)                 # psuedo-randomizer seed
nrules = 100                     # number of scheduling rules to simulate

solution = np.random.rand(nrules) # the correct weights/order of the rules
w = np.random.rand(nrules)        # a psuedo-random list of weights

# we don't need to recompare things, so do i-1
# you can offset the reward with 'offset' so depth still sorta makes sense
# e.g. for 50 rules: would be 2500 total comparisons; optimized it's 1176 iterations, so offset is 1324
# TODO: this should be a method
max_counter = 0
for i in range(nrules):
    for j in range(i-1):
        max_counter += 1
offset = (nrules**2) - max_counter
print('max counter: %d' % max_counter)
print('offset: %d' % offset)

# hyperparameters
npop = 100            # population size
max_generations = 50001
sigma = 0.01          # noise in standard deviations
alpha = 0.0005        # learning rate

def test(w, s):
    correct_comparison_counter = 0
    incorrect_comparison_counter = 0   
    wrong_i = 0
    wrong_j = 0
    for i in range(nrules):
        for j in range(i-1):
            if w[i] > w[j] and solution[i] > solution[j]:
                correct_comparison_counter += 1
            elif w[i] < w[j] and solution[i] < solution[j]:
                correct_comparison_counter += 1
            elif w[i] == w[j] and solution[i] == solution[j]:
                correct_comparison_counter += 1
            else:
                incorrect_comparison_counter += 1
                wrong_i = i
                wrong_j = j
                w_i = w[i]
                w_j = w[j]
                s_i = s[i]
                s_j = s[j]
                
                print('wrong i: %d, wrong j: %d\nw[i]: %f, w[j]: %f\ns[i]: %f, s[j]: %f' % 
                      (wrong_i, wrong_j, w_i, w_j, s_i, s_j))
            
    print('correct: %d, incorrect: %d\n' % 
                      (correct_comparison_counter, incorrect_comparison_counter))

def DLA(w):
    # finds the nrules that are in the right order compared to all other rules
    # not actually the depth from top
    # so it's an all or nothing solution because rules aren't prioritized
    correct_comparison_counter = 0    
    for i in range(nrules):
        for j in range(i-1):
            if w[i] > w[j] and solution[i] > solution[j]:
                correct_comparison_counter += 1
            elif w[i] < w[j] and solution[i] < solution[j]:
                correct_comparison_counter += 1
            elif w[i] == w[j] and solution[i] == solution[j]:
                correct_comparison_counter += 1
    
    reward = correct_comparison_counter
    return reward

def f(w):
    """    
    now, this is just a wrapper for DLA
    later, this will include a NN
    """
    reward = DLA(w)
    return reward

# evolve
for i in range(max_generations):
    correct_comparisons = f(w)
    
    # approx
    nrules_satisfied = (correct_comparisons + offset) / nrules        
    
    end = time.time()
    time_elapsed = (end - start) / 60
    
    # print current fitness of the most likely parameter setting
    if i % 20 == 0:
        print('iter: %d; nrules: %s; approx nrules satisfied %f\ncorrect comparison ratio: %d/%d\ntime elapsed: %f minutes\n' % 
          (i, str(nrules), nrules_satisfied, correct_comparisons, max_counter, time_elapsed))   
        
    if correct_comparisons == max_counter-1:
        test(w,solution)
        
    if correct_comparisons == max_counter:
        test(w,solution)
        print(' :::ALL RULES SATISFIED:::\n')
        print('This took %s minutes' % str(time_elapsed))
        break

    # initialize memory for a population of w's, and their rewards
    N = np.random.randn(npop, nrules) # samples from a normal distribution N(0,1)
    R = np.zeros(npop)
    
    # populate N with freaky mutants
    for mutant_i in range(npop):
        w_try = w + sigma*N[mutant_i]        # jitter w using gaussian of sigma 0.1
        R[mutant_i] = f(w_try)               # evaluate the jittered version

    # standardize the rewards to have a gaussian distribution
    A = (R - np.mean(R)) / np.std(R)
    # perform the parameter update. The matrix multiply below
    # to efficiently sum up all the rows of the noise matrix N,
    # where each row N[j] is weighted by A[j]    
    w = w + alpha/(npop*sigma) * np.dot(N.T, A)
