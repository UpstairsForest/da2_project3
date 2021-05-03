import numpy as np
import scipy
import functions_library as fl
import constants_library as cl


N_decays = 1e4 # number of generated decays
N_positions = 1e2 # number of evaluated detector positions
N_iterations = 10 # number of sets of decays evaluated for the average
z_min = 0
z_max = 1000

z_detector = np.arange(z_min, z_max, N_positions) # evaluated detector positions
r_decays = np.random.exponential(cl.L_k)

e_isotropic = fl.isotropic_angle_distribution(N_decays) # isotropically distributed unit vectors
kaon_p_cpi = cl.tot_p_pi * e_isotropic # momenta of the charged pions in the kaon rest frame
kaon_p_npi = -1* kaon_p_cpi # momenta of the neutral pions are opposite to those of the charged ones in kaon rest frame

lab_p_cpi = fl.boost(np.asarray([np.full(N_decays, cl.E_cpi), kaon_p_cpi[0], kaon_p_cpi[1], kaon_p_npi[2]]))[1:]
lab_p_npi = fl.boost(np.asarray([np.full(N_decays, cl.E_cpi), kaon_p_npi[0], kaon_p_npi[1], kaon_p_npi[2]]))[1:]

nondiv_v_cpi = fl.momentum_to_velocity(lab_p_cpi)
nondiv_v_npi = fl.momentum_to_velocity(lab_p_npi)

theta_gaussian = np.random.normal(size = N_decays, scale = 0.001) # scale is std in rad
r_decays = fl.rot_y(r_decays, theta_gaussian) # deviating decay positions
div_v_cpi = fl.rot_y(nondiv_v_cpi, theta_gaussian) # deviating velocities in the lab rest frame
div_v_npi = fl.rot_y(nondiv_v_npi, theta_gaussian)

for i in range(N_iterations):
    misses_average = (i*misses_average + fl.detector(z_detector)) / (i+1) # update the averages instead of storing all i

coefficients = scipy.optimize.curve_fit(fl.model_fit, z_detector, misses_average) # coeffs for the fit
optimal_detector_position = scipy.optimize.minimize(fl.fit(), x0 = 300)["x"][0]
