import numpy as np
# import matplotlib.pyplot as plt
from math import sqrt
from scipy.constants import c
from library import v_kaon as kaon_v
from library import p_pion as pion_p
from library import m_pp, m_np, m_k


def pion_pair(mode = "whatever"):
    positive_pion_fv = random_isotropic_rotation(pion_fv)
    neutral_pion_fv = -1 * positive_pion_fv  # inverting everything, so that the momenta are correct
    neutral_pion_fv[0] = m_k * c * c - positive_pion_fv[0]  # fixing energy through energy conservation

    positive_pion_velocity = momentum_to_velocity(boost(positive_pion_fv)[1:], m_pp)
    neutral_pion_velocity = momentum_to_velocity(boost(neutral_pion_fv)[1:], m_np)
    if mode != "non-divergent":
        positive_pion_velocity = gaussian_angle_rotation(positive_pion_velocity)
        neutral_pion_velocity = gaussian_angle_rotation(neutral_pion_velocity)
    return (positive_pion_velocity, neutral_pion_velocity)


def random_isotropic_rotation(pion_fv):  # ~Marsaglia method, checked, all good
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


def gaussian_angle_rotation(v):
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

    rand_theta = np.random.normal(scale=0.001)
    return Rz(current_phi) @ Ry(rand_theta) @ v


def Rz(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]])


def Ry(phi):
    return np.array([[np.cos(phi), 0, np.sin(phi)],
                     [0, 1, 0],
                     [-np.sin(phi), 0, np.cos(phi)]])


pion_E = sqrt(m_pp * c * c * m_pp * c * c + pion_p * c * pion_p * c)
pion_fv = np.array([pion_E, 0, 0, pion_p])

"""
data_len = 10000
things = np.zeros((data_len, 3))

fig, ax = plt.subplots()

for i in range(data_len):
    v = gaussian_angle_rotation(pion_pair()[0])
    things[i] = v
    ax.plot((0, v[0]), (0, v[1]), ls = "", marker=".", ms=3, color="orange", alpha=400/data_len)

ax.grid()
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("front_view")
ax.set_aspect("equal")
fig.tight_layout()
plt.show()
"""
