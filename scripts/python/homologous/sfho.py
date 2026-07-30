from in_solution import *
from singularity_eos import StellarCollapse
from scipy.integrate import odeint

import os, sys
sys.path.append('../stellar')

from progenitors import save_raw_profile

def gamma( T, rho):
    T_val = np.atleast_1d(T)[0]
    rho_val = np.atleast_1d(rho)[0]
    return np.atleast_1d(sfho.GruneisenParamFromDensityTemperature(T_val, rho_val, [0.5, 0])) * (T_val/rho_val)

# loads in our equation of state
sfho = StellarCollapse('../../../eos/sfho.h5')
print('>> sfho loaded in successfully.')

M = 1.4 * msun
R = 3.8e8
t = 1.0 * np.ones(1)
rs = np.linspace(0.01, 10.0, 2000)
lmbda = 0.002

r, f = solvef(rs, lmbda)
k = CalculateK(M, f, r)
a, adot = FindJeansLength(r, R, lmbda, k)
vel = CalculateVelocity(r, adot)
rho = CalculateDensity(a, f, k)

# initial setup to solve for logT(logrho)
T0 = 7.5e9 # arbitrary central temperature
print('>> initializing to solve for temperature profile.')

# solving for temp profile
temp = np.reshape(odeint(gamma, T0, rho), rho.size)
print('>> solved for temperature profile.')


ye = np.ones(rho.size) * 0.5
eps = np.zeros(rho.size)
press = np.zeros(rho.size)
u = np.zeros(rho.size)


for i in range(rho.size):
    eps[i] = sfho.InternalEnergyFromDensityTemperature(rho[i], temp[i], [ye[i], 0])
    press[i] = sfho.PressureFromDensityTemperature(rho[i], temp[i], [ye[i], 0])
    u[i] = eps[i] * rho[i]

prof = np.column_stack([
    r,
    vel,
    rho,
    press,
    ye,
    temp,
    eps,
    np.zeros(rho.size),
    np.zeros(rho.size),
    np.zeros(rho.size)
])

save_raw_profile( prof, 'sfho', 'homologous')