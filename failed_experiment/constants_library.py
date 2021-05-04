# contains all the literature values and derived constants such as the average kaon decay length
import numpy as np
from scipy.constants import c
from scipy.optimize import minimize


L_cpi = 4.188e3 # m
tau_cpi = 2.6033e-8 # +- 0.0005e-8 s
tau_k = 1.2380e-8 # +- 0.0020e-8

m_k = 4.93677e8 # +- 0.00016e8 eV/c**2
m_cpi = 1.3957039e8 # +- 0.0000018e8
m_npi = 1.349768e8 # +- 0.000005e8

E_k_rest = c * c * m_k # rest energies of the three particles
E_cpi_rest = c * c * m_cpi
E_npi_rest = c * c * m_npi

p_pi_tot = np.sqrt(((E_k_rest * E_k_rest - E_cpi_rest * E_cpi_rest - E_npi_rest * E_npi_rest) ** 2
                    - 4 * E_cpi_rest * E_cpi_rest * E_npi_rest * E_npi_rest) / (4 * E_k_rest * E_k_rest)) / c

p_k_tot = m_cpi * L_cpi / tau_cpi # from the given formula

v_cpi = p_pi_tot*c / np.sqrt(c * c * m_cpi * m_cpi + p_pi_tot * p_pi_tot)
v_k = p_k_tot*c / np.sqrt(c*c*m_k*m_k + p_k_tot*p_k_tot)

E_cpi_tot = np.sqrt(E_cpi_rest*E_cpi_rest + p_pi_tot*p_pi_tot*c*c)
E_npi_tot = np.sqrt(E_npi_rest*E_npi_rest + p_pi_tot*p_pi_tot*c*c)

x = np.loadtxt('dec_lengths.txt')
z = 8000 #the larger the number the more precise the NLL gets, but it also takes longer to run
l_k = np.linspace(1,5000, num=z)

#the negative log likelihood
def nll(l_k):
    return -1*np.sum(np.log((0.84/4188*np.exp(-x/4188))+0.16/l_k*np.exp(-x/l_k)))

#using scipy.optimize.minimize to get the minimum and with that the average decay length
f = minimize(nll, 2)
L_k_estimated = f.x[0]
tau_k_estimated = L_k_estimated * m_k / p_k_tot # kaon momentum is equal to charged pion momentum in other experiment

print('The average decay length is', np.round(L_k_estimated), '+- 10 m')
print('The average decay time of a kaon is', np.round(tau_k_estimated, 11),'+- 0.022e-8 s')
print('The theoretical value for the decay time is', tau_k, 's')