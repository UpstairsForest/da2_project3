# generates all the plots shown in the report
import numpy as np
import scipy.optimize
import functions_library as fl
import constants_library as cl
import matplotlib.pyplot as plt


N_decays = 100000 # number of generated decays
N_positions = 100 # number of evaluated detector positions
N_iterations = 10 # number of sets of decays evaluated for the average
z_min = 0
z_max = 1000
z_min_fit = 200
z_max_fit = 400

z_detector = np.linspace(z_min, z_max, N_positions) # evaluated detector positions
r_decays = np.array([np.random.exponential(cl.L_k_estimated, size = N_decays), np.zeros(N_decays), np.zeros(N_decays)]).T

e_isotropic = fl.isotropic_angle_distribution(N_decays) # isotropically distributed unit vectors
"""
plt.plot(e_isotropic[:, 0], e_isotropic[:,1], ls = '', marker = '.', ms = 1, alpha = 0.1)
plt.show()
"""
kaon_p_cpi = cl.p_pi_tot * e_isotropic # momenta of the charged pions in the kaon rest frame
kaon_p_npi = -1* kaon_p_cpi # momenta of the neutral pions are opposite to those of the charged ones in kaon rest frame

lab_p_cpi = fl.boost(np.asarray([np.full(N_decays, cl.E_cpi_tot), kaon_p_cpi[:, 0], kaon_p_cpi[:, 1], kaon_p_npi[:, 2]]).T)[:, 1:]
lab_p_npi = fl.boost(np.asarray([np.full(N_decays, cl.E_npi_tot), kaon_p_npi[:, 0], kaon_p_npi[:, 1], kaon_p_npi[:, 2]]).T)[:, 1:]

v_cpi_nondiv = fl.momentum_to_velocity(lab_p_cpi, cl.m_cpi) # charged pi velocities with non-divergent beam
v_npi_nondiv = fl.momentum_to_velocity(lab_p_npi, cl.m_npi)
"""
plt.plot(v_npi_nondiv[:,2], v_npi_nondiv[:, 1], ls = '', marker = '.', ms = 1, alpha = 0.1)
plt.show()
"""
theta_gaussian = np.random.normal(size = N_decays, scale = 0.001) # scale is std in rad
"""
plt.hist(theta_gaussian, bins = 100)
plt.show()
"""
r_decays_div = fl.rot_y(1*r_decays, theta_gaussian) # deviating decay positions
v_cpi_div = fl.rot_y(v_cpi_nondiv, theta_gaussian) # deviating velocities in the lab rest frame
v_npi_div = fl.rot_y(v_npi_nondiv, theta_gaussian)

misses_average = np.zeros(N_positions)
for i in range(N_iterations):
    if i/(N_iterations/10) - int(i/(N_iterations/10)) == 0:
        print("     iteration:", i)
    misses_average = (i * misses_average + fl.number_of_misses(z_detector, r_decays, v_cpi_nondiv, v_npi_nondiv)) / (i + 1)

coef = scipy.optimize.curve_fit(fl.model_fit, z_detector, misses_average)[0] # coeffs for the fit

def fit(x):
    return coef[0]*x*x*x*x + coef[1]*x*x*x + coef[2]*x*x + coef[3]*x + coef[4]


optimal_detector_position = scipy.optimize.minimize(fit, x0 = 200)["x"][0]
print("the optimal detector position with non-divergent beam is")
print("the optimal detector position is:", optimal_detector_position)

plt.plot(z_detector, misses_average, marker = ".")
plt.show()
