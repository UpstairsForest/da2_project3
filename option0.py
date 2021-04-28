import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from boost import pion_pair


pion_m_positive = 2.48807 * 10**(-34)   # kg, 139.57039*1e6 eV/c^2      from pdg
pion_m_neutral = 2.40618 * 10**(-34)    # kg, 134.9768*1e6   #eV/c^2    from pdg
kaon_m = 8.80059 * 10**(-34)            # kg, 493.677*1e6 eV/c^2        from pdg

pion_lifetime_positive = 2.6033*10**(-8) #seconds   from pdg
kaon_lifetime = 1.2380*10**(-8)          #seconds    from pdg

average_decay_length_pion_positive = 4.188*1e3 #meters  from script

# beta = v/c and gamma = 1/sqrt(1-beta^2)
# 4.188*1e3 = pion_l = beta * gamma * c * pion_t = pion_v / c * 1/sqrt(1-(pion_v)^2/c^2) * c * pion_t
# -> pion_v = pion_l / pion_t / sqrt(1 + (pion_l)^2 / (pion_t * c)^2)
# p = pion_m * pion_v
# kaon_v = p / kaon_m
# kaon_l = kaon_v / sqrt(1 - (kaon_v/c)^2) * tau_kaon

#p = pion_m_positive * average_decay_length_pion_positive / pion_lifetime_positive / sqrt(1 + average_decay_length_pion_positive**2 / (pion_lifetime_positive * c)**2)
#average_decay_length_kaon = p / kaon_m / sqrt(1 - (p / kaon_m / c)**2) * kaon_lifetime
# the new formulas give results that I think are not right

average_decay_length_kaon = pion_m_positive / kaon_m * kaon_lifetime / pion_lifetime_positive * average_decay_length_pion_positive

data_len = 10000
decay_position = np.zeros(data_len)
positive_pion_velocity = np.zeros((data_len, 3))
neutral_pion_velocity = np.zeros((data_len, 3))
for i in range(data_len):
    s = np.random.exponential(average_decay_length_kaon)
    decay_position[i] = s
    v, w = pion_pair()
    positive_pion_velocity[i] = v
    neutral_pion_velocity[i] = w
# print(np.mean(decay_position)) #arithmetic mean


def number_of_detections(detector_position):
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
    count_positive = results_positive.count(1) # this is how many times the detector detects the positive pion

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
    count_neutral = results_neutral.count(1) # this is how many times the detector detects the neutral pion

    success = []
    if count_positive != 0 and count_neutral != 0:
        indices_positive = [i for i, x in enumerate(results_positive) if x == 1]
        indices_neutral = [i for i, x in enumerate(results_neutral) if x == 1]
        success = [x for x in indices_positive if x in indices_neutral]

    fails = data_len - len(success)
    return fails    # to maximise success, we minimise fails


optimal_distance = scipy.optimize.minimize(number_of_detections,x0=1000)
print(optimal_distance["fun"])
print(optimal_distance["x"])


N = 100
positions = np.linspace(1, 1000, N)
results = 1*positions
for i, d in enumerate(positions):
    results[i] = number_of_detections(d)
plt.plot(positions, results, marker = ".", color = "orange")
plt.show()
