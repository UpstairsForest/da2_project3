import matplotlib.pyplot as plt
import numpy as np
from boost import pion_pair


pion_m_positive = 2.48807 * 10**(-34)   # kg, 139.57039*1e6 eV/c^2      from pdg
pion_m_neutral = 2.40618 * 10**(-34)    # kg, 134.9768*1e6   #eV/c^2    from pdg
kaon_m = 8.80059 * 10**(-34)            # kg, 493.677*1e6 eV/c^2        from pdg

pion_lifetime_positive = 2.6033*10**(-8) #seconds   from pdg
kaon_lifetime = 1.2380*10**(-8)          #seconds    from pdg

average_decay_length_pion_positive = 4.188*1e3 #meters  from script

average_decay_length_kaon = pion_m_positive / kaon_m * kaon_lifetime / pion_lifetime_positive * average_decay_length_pion_positive


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


N = 50
positions = np.linspace(1, 1000, N)
misses = []
for k in range(N):  # over some distances range
    data_len = 1000
    decay_position = np.zeros(data_len)
    positive_pion_velocity = np.zeros((data_len, 3))
    neutral_pion_velocity = np.zeros((data_len, 3))
    for i in range(data_len):   # generate data
        s = np.random.exponential(average_decay_length_kaon)
        decay_position[i] = s
        v, w = pion_pair()
        positive_pion_velocity[i] = v
        neutral_pion_velocity[i] = w
    misses.append(number_of_detections(positions[k]))
    # evaluate the misses and store them for that distance and data set

plt.plot(positions, misses, marker = ".", color = "orange") # fit this
plt.show()
