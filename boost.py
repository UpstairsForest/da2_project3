import numpy as np
# import matplotlib.pyplot as plt
from math import sqrt
from scipy.constants import c


def pion_pair():
    positive_pion_fv = random_isotropic_rotation(pion_fv)
    neutral_pion_fv = -1*positive_pion_fv # inverting everything, so that the momenta are correct
    neutral_pion_fv[0] = m_k*c*c - positive_pion_fv[0] # fixing energy through energy conservation
    positive_pion_velocity = boost(positive_pion_fv)[1:]*c*c/positive_pion_fv[0] # relativistic momentum
    neutral_pion_velocity = boost(neutral_pion_fv)[1:]*c*c/neutral_pion_fv[0]
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
    LT = np.array([[gamma, 0, 0, beta * gamma*c], # from appendix with *c, /c added to match the units
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [beta * gamma/c, 0, 0, gamma]])

    return LT @ fv


kaon_v = 299785945.432217 # from library
pion_v = 242011283.59450692  # from library
pion_m = 1.349768 * 1e8  # eV/c**2 for +pion from pdg
m_k = 4.93677 * 1e8
m_np = 1.349768 * 1e8

pion_p = 5.723153942711986e+16
pion_E = sqrt(pion_m * c * c * pion_m * c * c + pion_p * c * pion_p * c)  # E = sqrt( (mc**2)**2 + (p*c)**2)
pion_fv = np.array([pion_E, 0, 0, pion_p])  # four-vector parallel to z-axis

'''
data_len = 100
pions = np.zeros((data_len, 2, 3))

shit = 0

for i in range(data_len):
    v = pion_pair()
    pions[i] = v
    absolute_v = sqrt(v[0][0]**2 + v[0][1]**2 + v[0][2]**2), sqrt(v[1][0]**2 + v[1][1]**2 + v[1][2]**2)
    if absolute_v[0] >= c:
        shit +=1
    if absolute_v[1] >= c:
        shit +=1
print("ratio of pions which exceed the speed of light:", round(shit/2/data_len, 2))

fig, ax = plt.subplots(1, 1)
for v in pions:
    ax.plot((0, v[0, 2]), (0, v[0, 1]), lw=0.2, marker=".", ms=3, color="orange", alpha=0.5)
    ax.plot((0, v[1, 2]), (0, v[1, 0]), lw=0.2, marker=".", ms=3, color="green", alpha=0.5)

ax.grid()
ax.set_xlabel("v_z [m/s]")
ax.set_ylabel("v_y [m/s]")
ax.set_title("side_view")
#ax.set_aspect("equal")
fig.tight_layout()
plt.show() # +pi in orange, 0pi in green
'''