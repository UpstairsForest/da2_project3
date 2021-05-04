import numpy as np
# import matplotlib.pyplot as plt
from math import sqrt
from scipy.constants import c
from library import v_kaon as kaon_v
from library import m_pp, m_np, m_k, pion_fv, pion_E, average_decay_length_kaon


# evaluates entire generated data on one position
def number_of_detections(data_len, detector_position, decay_position, positive_pion_velocity, neutral_pion_velocity):
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

def data_generator(data_len): # generates pions and decay positions
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

def pion_pair(mode = "whatever"):
    positive_pion_fv = random_isotropic_rotation(pion_fv)
    neutral_pion_fv = -1 * positive_pion_fv  # inverting everything, so that the momenta are correct
    neutral_pion_fv[0] = m_k * c * c - positive_pion_fv[0]  # fixing energy through energy conservation

    positive_pion_velocity = momentum_to_velocity(boost(positive_pion_fv)[1:], m_pp)
    neutral_pion_velocity = momentum_to_velocity(boost(neutral_pion_fv)[1:], m_np)
    if mode != "non-divergent":
        rand_theta = np.random.normal(scale=0.001)
        positive_pion_velocity = gaussian_angle_rotation(positive_pion_velocity, rand_theta)
        neutral_pion_velocity = gaussian_angle_rotation(neutral_pion_velocity, rand_theta)

    return (positive_pion_velocity, neutral_pion_velocity)


def random_isotropic_rotation(pion_fv):  # ~Marsaglia method
    fv = 1 * pion_fv
    while True:
        a = 2 * np.random.rand() - 1
        b = 2 * np.random.rand() - 1
        if a * a + b * b < 1:
            break
    fv[1] = fv[3] * 2 * a * sqrt(1 - a * a - b * b)
    fv[2] = fv[3] * 2 * b * sqrt(1 - a * a - b * b)
    fv[3] = fv[3] * (1 - 2 * (a * a + b * b))
    return fv


def boost(fv):
    beta = kaon_v / c
    gamma = 1 / sqrt(1 - beta * beta)
    LT = np.array([[gamma, 0, 0, beta * gamma * c],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [beta * gamma / c, 0, 0, gamma]])

    return LT @ fv


def momentum_to_velocity(p, m):
    p_tot2 = np.sum(p * p)  # total momentum squared
    beta2 = p_tot2 / (m * m * c * c + p_tot2)
    return p * sqrt(1 - beta2) / m


def gaussian_angle_rotation(v, rand_theta):
    if v[0] != 0:
        current_phi = np.arctan(v[1] / v[0])
        if v[0] < 0 and v[1] < 0:
            current_phi = np.pi + current_phi
    else:
        current_phi = np.pi / 2
        if v[1] < 0:
            current_phi = np.pi + current_phi
    v = Rz(-current_phi) @ v  # rotate to xz plane

    if v[2] != 0:
        current_theta = np.arctan(v[0] / v[2])
        if v[2] < 0 and v[0] < 0:
            current_theta = np.pi + current_theta
    else:
        current_theta = np.pi / 2
        if v[0] < 0:
            current_theta = np.pi + current_theta

    return Rz(current_phi) @ Ry(rand_theta) @ v


def Rz(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]])


def Ry(phi):
    return np.array([[np.cos(phi), 0, np.sin(phi)],
                     [0, 1, 0],
                     [-np.sin(phi), 0, np.cos(phi)]])
