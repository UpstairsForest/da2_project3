import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from boost import pion_pair
from average_decay_length import kaon_average_decay_length


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
    results_neutral = []
    distance = np.zeros(data_len)
    for k in range(data_len):
        distance[k] = detector_position-decay_position[k]
        if distance[k] < 0:
            results_positive.append(0)
            results_neutral.append(0)
        else:
            t_positive = distance[k] / positive_pion_velocity[k][2]
            dx_positive = t_positive * positive_pion_velocity[k][0]
            dy_positive = t_positive * positive_pion_velocity[k][1]
            if dx_positive ** 2 + dy_positive ** 2 > 4:
                results_positive.append(0)
            else:
                results_positive.append(1)

            t_neutral = distance[k] / neutral_pion_velocity[k][2]
            dx_neutral = t_neutral * neutral_pion_velocity[k][0]
            dy_neutral = t_neutral * neutral_pion_velocity[k][1]
            if dx_neutral ** 2 + dy_neutral ** 2 > 4:
                results_neutral.append(0)
            else:
                results_neutral.append(1)

    count_positive = results_positive.count(1)
    count_neutral = results_neutral.count(1)

    success = []
    if count_positive != 0 and count_neutral != 0:
        indices_positive = [i for i, x in enumerate(results_positive) if x == 1]
        indices_neutral = [i for i, x in enumerate(results_neutral) if x == 1]
        success = [x for x in indices_positive if x in indices_neutral]

    fails = data_len - len(success)
    return fails


'''
Up untill here only copy of option0, now comes the interesting part:
'''

data_len = 1000 # how many pion pairs and corresponding decay positions we generate
iterations = 10 # how many times we generate data sets of length data_len
detector_positions = np.linspace(0, 1000, 100) # detector positions on which we check number_of_detections()

average_decay_length_kaon = kaon_average_decay_length
average_fails_over_distances = 0*detector_positions
fails_over_distances_over_iterations = np.zeros((iterations, len(detector_positions)))

for i in range(iterations):
    print("iteration:", i)
    decay_position, positive_pion_velocity, neutral_pion_velocity = data_generator()
    for j, d in enumerate(detector_positions):
        fails_over_distances_over_iterations[i, j] = number_of_detections(d)
    plt.plot(detector_positions[20:35], fails_over_distances_over_iterations[i][20:35], lw = 1, color = "green", alpha = 1/iterations)
average_fails_over_distances = np.average(fails_over_distances_over_iterations, axis = 0) # axis 0 is iterations


def model_fct(x, a, b, c, d, e):
    return a*x*x*x*x + b*x*x*x + c*x*x + d*x + e     # we want the fit to be a polynomial of fourth degree


detector_positions_slice = detector_positions[20:35]                        #slicing the data for a good fit
average_fails_over_distances_slice = average_fails_over_distances[20:35]
popt, pcov = scipy.optimize.curve_fit(model_fct, detector_positions_slice, average_fails_over_distances_slice)  # fitting


def fit(x):
    return model_fct(x,popt[0], popt[1], popt[2], popt[3], popt[4])


optimal = scipy.optimize.minimize(fit,x0=300)   # optimising the fit
print("The minimum lies at", optimal["x"][0])

plt.plot(detector_positions[20:35], average_fails_over_distances[20:35], lw = 1, marker = ".", color = "orange")
plt.plot(detector_positions_slice, model_fct(detector_positions_slice, popt[0], popt[1], popt[2], popt[3], popt[4]), color = "black")
plt.title("Number of Fails versus Detector Positions")
plt.xlabel("Detector position")
plt.ylabel("Number of fails")
plt.show()
