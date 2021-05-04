import numpy as np
import scipy.optimize
from functions import data_generator, number_of_detections
import matplotlib.pyplot as plt


data_len = 1000 # how many pion pairs and corresponding decay positions we generate
iterations = 10 # how many times we generate data sets of length data_len
detector_positions = np.linspace(0, 1000, 100) # detector positions on which we check number_of_detections()

average_fails_over_distances = 0*detector_positions
fails_over_distances_over_iterations = np.zeros((iterations, len(detector_positions)))

for i in range(iterations):
    decay_position, positive_pion_velocity, neutral_pion_velocity = data_generator(data_len)
    for j, d in enumerate(detector_positions):
        fails_over_distances_over_iterations[i, j] = number_of_detections(data_len, d, decay_position, positive_pion_velocity, neutral_pion_velocity)
    plt.plot(detector_positions, fails_over_distances_over_iterations[i], lw=1, color="green",alpha=1 / iterations)
average_fails_over_distances = np.average(fails_over_distances_over_iterations, axis = 0) # axis 0 is iterations


def model_fct(x, a, b, c, d, e):
    return a*x*x*x*x + b*x*x*x + c*x*x + d*x + e     # we want the fit to be a polynomial of fourth degree


detector_positions_slice = detector_positions[20:35]                        #slicing the data for a good fit
average_fails_over_distances_slice = average_fails_over_distances[20:35]
popt, pcov = scipy.optimize.curve_fit(model_fct, detector_positions_slice, average_fails_over_distances_slice)  # fitting


def fit(x):
    return model_fct(x,popt[0], popt[1], popt[2], popt[3], popt[4])


optimal = scipy.optimize.minimize(fit,x0=300)   # optimising the fit
print("The minimum lies at", np.round(optimal["x"][0], 1), "m")

plt.plot(detector_positions, average_fails_over_distances, lw = 1, marker = ".", color = "orange")
plt.plot(detector_positions_slice, model_fct(detector_positions_slice, popt[0], popt[1], popt[2], popt[3], popt[4]), color = "black")
plt.title("Number of Fails versus Detector Positions")
plt.xlabel("Detector position")
plt.ylabel("Number of fails")
plt.show()
