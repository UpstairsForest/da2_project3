import numpy as np
from scipy.constants import c
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

p_pion = sqrt((E_k**4 + E_pp**4 + E_np**4 - 2*E_pp*E_pp*E_np*E_np - 2*E_k*E_k*(E_pp*E_pp + E_np*E_np))
               /(4*(E_k*E_k + E_pp*E_pp + E_np*E_np)))/c

p_kaon = m_pp * d_pp / t_pp

v_kaon = p_kaon*c / sqrt(c*c*m_k*m_k + p_kaon*p_kaon)

v_pion = p_pion*c / sqrt(c*c*m_pp*m_pp + p_pion*p_pion)

d_k = m_pp/m_k * t_k/t_pp * d_pp
