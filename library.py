import numpy as np
from scipy.constants import c
from scipy.optimize import minimize
from math import sqrt

d_pp = 4.188 * 1e3 # m
t_pp = 2.6033 * 1e-8 # +- 0.0005 * 1e-8 s
t_k = 1.2380 *1e-8 # +- 0.0020 * 1e-8

m_k = 4.93677 * 1e8 # +- 0.00016 *1e8 eV/c**2
m_pp = 1.3957039 * 1e8 # +- 0.0000018 * 1e8
m_np = 1.349768 * 1e8 # +- 0.000005 * 1e8

E_k = c*c * m_k # rest energies of the three particlesp
E_pp = c*c * m_pp
E_np = c*c * m_np

p_pion = sqrt(((E_k*E_k - E_pp*E_pp - E_np*E_np)**2 - 4*E_pp*E_pp*E_np*E_np) / (4*E_k*E_k))/c
p_kaon = m_pp * d_pp / t_pp

v_kaon = p_kaon*c / sqrt(c*c*m_k*m_k + p_kaon*p_kaon)
v_pion = p_pion*c / sqrt(c*c*m_pp*m_pp + p_pion*p_pion)

pion_E = sqrt(m_pp * c * c * m_pp * c * c + p_pion * c * p_pion * c)
pion_fv = np.array([pion_E, 0, 0, p_pion])

x = np.loadtxt('dec_lengths.txt')
z = 8000 #the larger the number the more precise the NLL gets, but it also takes longer to run
l_k = np.linspace(1,5000, num=z)

#the negative log likelihood
def nll(l_k):
    return -1*np.sum(np.log((0.84/4188*np.exp(-x/4188))+0.16/l_k*np.exp(-x/l_k)))

#using scipy.optimize.minimize to get the minimum and with that the average decay length
f = minimize(nll, 2)
average_decay_length_kaon = f.x[0]
print('The average decay length is', np.round(average_decay_length_kaon, 1), "m")

kaon_lifetime = f.x[0] * m_k / p_kaon
print('The average decay time of a kaon is', np.round(kaon_lifetime, 11), 's')
print('The theoretical value for the decay time is', 1.2380*10**-8, 's')