import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

x = np.loadtxt('dec_lengths.txt')
z = 8000 #the larger the number the more precise the NLL gets, but it also takes longer to run
l_k = np.linspace(1,5000, num=z)

#the negative log likelihood
def nll(l_k):
    return -1*np.sum(np.log((0.84/4188*np.exp(-x/4188))+0.16/l_k*np.exp(-x/l_k)))

#just to create the horizontal line which gives the uncertainty
y = []
for i in range(z):
    y.append(916722.5)

#plotting everything
plt.figure()
plt.title('NLL', fontweight="bold")
plt.plot(l_k,list(map(nll, l_k)))
plt.plot(l_k, y,color='red', linestyle='--')
plt.ylabel('NLL')
plt.xlabel('x[m]')
plt.grid(color='lightgrey', linestyle='-')
plt.xlim([545, 580])
plt.ylim([916722, 916722.9])
plt.show()

#using scipy.optimize.minimize to get the minimum and with that the average decay length
f = minimize(nll, 2)
kaon_average_decay_length = f.x
print('The average decay length is', kaon_average_decay_length)


#compare with the theoretical value
pion_m_positive = 2.48807 * 10**(-34)   # kg, 139.57039*1e6 eV/c^2      from pdg
kaon_m = 8.80059 * 10**(-34)            # kg, 493.677*1e6 eV/c^2        from pdg
pion_decay_length = 4188                # m
pion_lifetime_positive = 2.6033*10**(-8) #seconds   from pdg
kaon_lifetime = 1.2380*10**(-8)          #seconds    from pdg

pion_momentum = pion_decay_length/pion_lifetime_positive * pion_m_positive
kaon_lifetime = f.x * kaon_m / pion_momentum
print('The average decay time of a kaon is', kaon_lifetime, 's')
print('The theoretical value for the decay time is', 1.2380*10**-8, 's')