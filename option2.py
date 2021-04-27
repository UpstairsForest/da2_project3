import matplotlib.pyplot as plt
import numpy as np
from boost import pion_pair
from library import d_k


def data_generator(): # generates pions and decay positions
    decay_position = np.zeros(data_len)
    positive_pion_velocity = np.zeros((data_len, 3))
    neutral_pion_velocity = np.zeros((data_len, 3))
    for i in range(data_len):
        s = np.random.exponential(average_decay_length_kaon)
        decay_position[i] = s
        v, w = pion_pair()
        positive_pion_velocity[i] = v
        neutral_pion_velocity[i] = w
    return decay_position, positive_pion_velocity, neutral_pion_velocity


def number_of_detections(detector_position): # evaluates entire generated data on one position
    results_positive = []
    distance = np.zeros(data_len)
    for k in range(data_len):
        distance[k] = detector_position-decay_position[k]
        if positive_pion_velocity[k][2] <= 0 and distance[k]>0:
            results_positive.append(0)
        elif positive_pion_velocity[k][2] >= 0 and distance[k]<0:
            results_positive.append(0)
        else:
            t = abs(distance[k] / positive_pion_velocity[k][2])
            dx = t * positive_pion_velocity[k][0]
            dy = t * positive_pion_velocity[k][1]
            if dx**2+dy**2 > 4:
                results_positive.append(0)
            else:
                results_positive.append(1)
    count_positive = results_positive.count(1)

    results_neutral = []
    for k in range(data_len):
        if neutral_pion_velocity[k][2] <= 0 and detector_position - decay_position[k] > 0:
            results_positive.append(0)
        elif neutral_pion_velocity[k][2] >= 0 and detector_position - decay_position[k] < 0:
            results_neutral.append(0)
        else:
            t = abs(distance[k])/neutral_pion_velocity[k][2]
            dx = t*neutral_pion_velocity[k][0]
            dy = t * neutral_pion_velocity[k][1]
            if dx**2+dy**2 > 4:
                results_neutral.append(0)
            else:
                results_neutral.append(1)
    count_neutral = results_neutral.count(1)

    success = []
    if count_positive != 0 and count_neutral != 0:
        indices_positive = [i for i, x in enumerate(results_positive) if x == 1]
        indices_neutral = [i for i, x in enumerate(results_neutral) if x == 1]
        success = [x for x in indices_positive if x in indices_neutral]

    fails = data_len - len(success)
    return fails


'''
Up untill here only copy of main, now comes the interesting part:
'''

data_len = 1000 # how many pion pairs and corresponding decay positions we generate
iterations = 10 # how many times we generate data sets of length data_len
detector_positions = np.linspace(0, 1000, 100) # detector positions on which we check number_of_detections()

average_decay_length_kaon = d_k
average_fails_over_distances = 0*detector_positions
fails_over_distances_over_iterations = np.zeros((iterations, len(detector_positions)))

for i in range(iterations):
    decay_position, positive_pion_velocity, neutral_pion_velocity = data_generator()
    for j, d in enumerate(detector_positions):
        fails_over_distances_over_iterations[i, j] = number_of_detections(d)
    plt.plot(detector_positions, fails_over_distances_over_iterations[i], lw = 1, color = "green", alpha = 1/iterations)
average_fails_over_distances = np.average(fails_over_distances_over_iterations, axis = 0) # axis 0 is iterations
'''
fit average_fails_over_distances
define a new function from fit
minimize the new function
'''

plt.plot(detector_positions, average_fails_over_distances, lw = 1, marker = ".", color = "orange")
plt.show()
