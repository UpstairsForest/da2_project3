# contains only functions
import numpy as np
import scipy.optimize
from scipy.constants import c
from constants_library import v_k


def isotropic_angle_distribution(N):
    v = np.zeros((N, 3))
    for i in range(N):
        while True:
            a = 2 * np.random.rand() - 1
            b = 2 * np.random.rand() - 1
            if a * a + b * b < 1:
                break
        v[i, 0] = 2 * a * np.sqrt(1 - a * a - b * b)
        v[i, 1] = 2 * b * np.sqrt(1 - a * a - b * b)
        v[i, 2] = 1 - 2 * (a * a + b * b)
    return v  # N by 3 array

def boost(fv): # takes an array of four-vectors in kaon rest frame
    beta = v_k/c
    gamma = 1/np.sqrt(1 - beta*beta)
    LT = np.array([[gamma, 0, 0, beta * gamma * c],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [beta * gamma / c, 0, 0, gamma]])
    for i, v in enumerate(fv):
        fv[i] = LT @ v
    return fv

def momentum_to_velocity(p, m): # takes array of relativistic momenta and mass as arguments
    p_tot2 = np.sum(p * p, axis = 1)  # array of total momenta squared
    beta2 = p_tot2 / (m * m * c * c + p_tot2)
    return p * np.sqrt(1 - np.full((3, len(beta2)), beta2).T) / m # returns an array of velocities

def rot_y(r, theta): # takes array of vectors and angles as arguments
    for i, v in enumerate(r):
        r[i] = np.array([[np.cos(theta[i]), 0, np.sin(theta[i])],
              [0, 1, 0],
              [-np.sin(theta[i]), 0, np.cos(theta[i])]]) @ v
    return r

def number_of_misses(z_detector, generated):
    r_decays, v_cpi, v_npi = generated
    misses_over_distances = 0*z_detector
    for i, z in enumerate(z_detector):
        detections = r_decays[:, 2] <= z # decays behind the detector cannot be detected
        t_cpi = (z - r_decays[:, 2]) / v_cpi[:, 2]
        t_npi = (z - r_decays[:, 2]) / v_npi[:, 2]
        del_x_cpi = r_decays[:, 0] + t_cpi * v_cpi[:, 0]
        del_x_npi = r_decays[:, 0] + t_npi * v_npi[:, 0]
        del_y_cpi = r_decays[:, 1] + t_cpi * v_cpi[:, 1]
        del_y_npi = r_decays[:, 1] + t_cpi * v_npi[:, 1]
        detections *= del_x_cpi*del_x_cpi + del_y_cpi*del_y_cpi <= 4 # "and" gate
        detections *= del_x_npi*del_x_npi + del_y_npi*del_x_npi <= 4
        misses_over_distances[i] = len(r_decays)-np.sum(detections)
    return misses_over_distances

def model_fit(x, a, b, c, d, e):
    return a*x*x*x*x + b*x*x*x + c*x*x + d*x + e
