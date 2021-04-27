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


def number_of_detections(detector_pos, decay_pos, positive_pion_vel, neutral_pion_vel):
    distance = detector_pos - decay_pos

    if positive_pion_vel[2] <= 0 and distance>0:
            results_positive = 0
    elif positive_pion_vel[2] >= 0 and distance<0:
            results_positive = 0
    else:
        t = abs(distance / positive_pion_vel[2])
        dx = t * positive_pion_vel[0]
        dy = t * positive_pion_vel[1]
        if dx**2+dy**2 > 4:
            results_positive = 0
        else:
            results_positive = 1

    if neutral_pion_vel[2] <= 0 and detector_pos - decay_pos > 0:
        results_neutral = 0
    elif neutral_pion_vel[2] >= 0 and detector_pos - decay_pos < 0:
        results_neutral = 0
    else:
        t = abs(distance)/neutral_pion_vel[2]
        dx = t*neutral_pion_vel[0]
        dy = t * neutral_pion_vel[1]
        if dx**2+dy**2 > 4:
            results_neutral = 0
        else:
            results_neutral = 1

    if results_positive == 1 and results_neutral == 1:
        success = 1
    else:
        success = 0

    fails = 1 - success
    return fails    # to maximise success, we minimise fails


data_len = 10
decay_position = np.zeros(data_len)
positive_pion_velocity = np.zeros((data_len, 3))
neutral_pion_velocity = np.zeros((data_len, 3))
for i in range(data_len):      # N times
    s = np.random.exponential(average_decay_length_kaon)    # generate data
    decay_position[i] = s
    v, w = pion_pair()
    positive_pion_velocity[i] = v
    neutral_pion_velocity[i] = w

    N = 100
    positions = np.linspace(1, 1000, N)
    misses = []
    for j in range(N):  # over some distances range
        misses.append(number_of_detections(positions[j], decay_position[i], positive_pion_velocity[i], neutral_pion_velocity[i]))
        # evaluate the misses fct and store the number for that distance always with the same data
        # I think only one point is not enough, we should generate several and take the average nr of misses because
        # only one is up to fluctuations, does not represent how good this position really is
    plt.plot(positions,misses, marker = ".", color = "orange") # fit this
    plt.ylabel("Number of misses")
    plt.xlabel("Distance of Detector")
    plt.show()

# I think rather than fit the outcome, we should somehow try to make an intersection of the 0 lines for each plot
