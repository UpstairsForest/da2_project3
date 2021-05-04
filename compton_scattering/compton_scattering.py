import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy

c = 2.99792458 * 10 ** 8  # m/s
m = 9.10938356 * 10 ** (-31)  # kg
m = 0.51099895000 #Mev
E_in = 0.6617  # MeV

thetas_deg = np.array([10, 20, 30, 40, 50, 60, 70, 80])
thetas = (np.pi / 180) * thetas_deg

mu = 0
sig = 0.05  # MeV


def E_out(tita, mass):
    return E_in / (1 + (E_in / (mass) * (1 - np.cos(tita))))


def pull(generated, reconstructed, uncertainty_reco):
    return (reconstructed - generated) / uncertainty_reco


def pdf(x, y, deltas, mass):
    return 1/(deltas*(2*np.pi)*.5)*np.exp(-(y-E_out(x, mass))**2/(2*deltas*2))


def NLL_2(mass):
    return -2*np.sum(np.log(pdf(thetas, E_out_reco, sig, mass)))



'''3a Simulate the experiment'''

# generates a list of 8 random values from the gaussian to simulate the detector's finite resolution
del_E = np.random.normal(mu, sig, 8)
E_out_reco = E_out(thetas, m) + del_E

plt.figure(1)
plt.xlabel('theta [rad]')
plt.ylabel('E [J]')
plt.errorbar(thetas, E_out_reco, yerr=sig, xerr=None, fmt='o', markersize=3)
plt.savefig('1 E_out_reco vs Theta.png')


'''3b maximum likelihood fit'''

plt.figure(2)
plt.xlabel('mass [MeV/c^2]')
plt.ylabel('2*NLL')
masses_x = np.linspace(0.9 * m, 1.1 * m, 300)
NLL_x = []
for elem in masses_x:
    NLL_x.append(NLL_2(elem))
plt.plot(masses_x, NLL_x)
plt.savefig('2 2NLL.png')

# finding the minimum of the NLL with a loop
start_m = 0.4
while NLL_2(start_m) > NLL_2(start_m + 0.001):
    start_m += 0.001

min_m = start_m

# finding the uncertainty of the minimised parameter m.
# the uncert of a parameter can be obtained by seeing where the NLL goes up by 0.5 units (Lecture 12 Wilks Theorem)

var = min_m+(0.01)
while NLL_2(var) - NLL_2(min_m) < 1:
    var += (0.01)
min_m_uncert = var - min_m - 0.005 # subtracting the half of the step size to get more precise results


#for checking purposes
#optimized_func = scipy.optimize.minimize(NLL_2, np.array([0.95*m]))
#print(optimized_func['x'][0])

# find the NLL with the curve_fit function
apple = curve_fit(E_out, thetas, E_out_reco, p0=0.5)
# E_out is the function, we give our measurement points E_out_reco and p0, a guess for mass
mass_reco, mass_reco_uncert = (apple[0][0], np.sqrt(apple[1][0][0]))

plt.figure(3)
plt.xlabel('theta [rad]')
plt.ylabel('E [MeV]')
t_values = np.linspace(0, 2, 1000)
plt.plot(t_values, E_out(t_values, mass_reco))
plt.errorbar(thetas, E_out_reco, yerr=sig, xerr=None, fmt='o', markersize=3)
plt.savefig('3 fit.png')

print(f'''3b: the measured mass found manually using the NLL is m_reco = {min_m} +/- {min_m_uncert} 
and with scipy.optimize.curve_fit is m_reco = {mass_reco} +/- {mass_reco_uncert}''')


'''4a repeat the simulation 1000 times'''

masses = []
mass_uncerts = []
for i in range(1000):
    del_E = np.random.normal(mu, sig, 8)
    E_list = E_out(thetas, m)
    E_out_reco = E_list + del_E

    pineapple = curve_fit(E_out, thetas, E_out_reco, p0=0.5)
    m_reco_i, m_reco_uncert_i = (pineapple[0][0], np.sqrt(pineapple[1][0][0]))

    masses.append(m_reco_i)
    mass_uncerts.append(m_reco_uncert_i)

masses = np.array(masses)
mass_uncerts = np.array(mass_uncerts)


'''4b histogram of the masses'''

plt.figure(4)
bins = np.linspace(min(masses), max(masses), 30)
plt.hist(masses, bins)
plt.xlabel('mass [MeV/c^2]')
plt.ylabel('n entries')
plt.savefig('4 histogram of reconstructed masses.png')

# Calculation of the mean and the standard deviation of the Histogram

n, bins = np.histogram(masses, 30)
mids = 0.5 * (bins[1:] + bins[:-1])
mean = np.average(mids, weights=n)
var = np.average((mids - mean) ** 2, weights=n)
print(f'''4b: the standard deviation of the histogram is {np.sqrt(var)} and the mean is {mean}''')


'''4c histogram of the pull'''

# Histogram of pull distribution
plt.figure(5)
pull1 = pull(masses, mass_reco, mass_reco_uncert) #curve_fitw
pull2 = pull(masses, min_m, min_m_uncert) #manually
bins1 = np.linspace(min(pull1), max(pull1), 30)
bins2 = np.linspace(min(pull2), max(pull2), 30)
plt.hist(pull1, bins1, alpha=0.4, color='r', label='curve_fit')
plt.hist(pull2, bins2, alpha=0.4, color='b', label='manually')
plt.xlabel('mass [MeV/c^2]')
plt.ylabel('n entries')
plt.legend()
plt.savefig('5 histogram of pull distribution of masses.png')


n1, bins1 = np.histogram(pull1, 30)
n2, bins2 = np.histogram(pull2, 30)
mids1 = 0.5 * (bins1[1:] + bins1[:-1])
mids2 = 0.5 * (bins2[1:] + bins2[:-1])
mean1 = np.average(mids1, weights=n)
var1 = np.average((mids1 - mean1) ** 2, weights=n)
mean2 = np.average(mids2, weights=n)
var2 = np.average((mids2 - mean2) ** 2, weights=n)

print(f'''4c: the standard deviation of the histogram for pull1 (curve fit) is {np.sqrt(var1)} and the mean is {mean1}
the standard deviation of the histogram for pull2 (manually) is {np.sqrt(var2)} and the mean is {mean2}''')