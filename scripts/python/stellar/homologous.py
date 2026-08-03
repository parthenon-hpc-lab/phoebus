# consolidation of Mariam's homologous generation scripts, plus
# additional resources to create profiles that are either 
#     - consistent with a tabulated eos (using singularity-eos) instead of ideal gas, or
#     - made with the Yahil 1983 homologous collapse case in mind.


# law. 29 jul 26. 


import numpy as np
import math

from singularity_eos import StellarCollapse
from scipy.integrate import odeint
from astropy import constants as const

G = const.G.cgs.value
c = const.c.cgs.value
msun = const.M_sun.cgs.value

def gen_homologous_goldreich( M: float = 1.4, R: float = 3.8e8, lmbda: float = 0.002, rmin: float = 0.001, rmax: float = 10.0, zones: int = 10000, gamma: float = 4.0/3.0, verbose: bool = True):
    
    M *= msun
    rs = np.linspace(rmin, rmax, zones)
    cv = 1.0

    r, f = solvef(rs, lmbda)
    k = CalculateK(M, f, r)
    a, adot = FindJeansLength(r, R, lmbda, k)

    vel = CalculateVelocity(r, adot)
    rho = CalculateDensity(a, f, k)
    eps = CalculateSpecificInternalEnergy(rho, k, gamma)
    press = CalculatePressure(rho, k, gamma)
    temp = CalculateTemperature(cv, eps)
    rhoe = np.zeros(r.size)

    r *= a[0]  ### dimensionful

    for n in range(r.size):
        rhoe[n] = rho[ n] + rho[ n] * eps[ n] / c ** 2.0

    prof = np.column_stack([
    r,
    vel,
    rho,
    press,
    np.ones(rho.size) * 0.5,
    temp,
    eps,
    np.zeros(rho.size),
    np.zeros(rho.size),
    np.zeros(rho.size)
    ])

    if verbose: print(f'>>> generated homologous collapse profile (Goldreich & Weber 1980) with mass {M/msun:.2f} msun and radius {R:.2e} cm.')


    return prof

def gen_homologous_eos( EOSPATH: str, eostype: str = 'stellarcollapse', T0: float = 6.5e9, M: float = 1.4, R: float = 3.8e8, lmbda: float = 0.002, rmin: float = 0.001, rmax: float = 10.0, zones: int = 10000, verbose: bool = True ):
    
    if eostype.lower() == 'stellarcollapse':
        eos = StellarCollapse( EOSPATH )
    else:
        raise UserWarning(f'{eostype} EOS not currently supported for homologous initialization.')
            
    M *= msun
    rs = np.linspace(rmin, rmax, zones)

    r, f = solvef(rs, lmbda)
    k = CalculateK(M, f, r)
    a, adot = FindJeansLength(r, R, lmbda, k)
    vel = -1.0 * CalculateVelocity(r, adot) # we need to make sure the velocity is falling in
    rho = CalculateDensity(a, f, k)
    
    r *= a[0] # dimension correction

    def gamma( T, rho):
        T_val = np.atleast_1d(T)[0]
        rho_val = np.atleast_1d(rho)[0]
        return np.atleast_1d(eos.GruneisenParamFromDensityTemperature(T_val, rho_val, [0.5, 0])) * (T_val/rho_val)

    # solving for temp profile using the Gruneisen param (as defined above)
    temp = np.reshape(odeint(gamma, T0, rho), rho.size)

    # iterating with equation of state to close other primitives
    ye = np.ones(rho.size) * 0.5
    eps = np.zeros(rho.size)
    press = np.zeros(rho.size)
    u = np.zeros(rho.size)


    for i in range(rho.size):
        eps[i] = eos.InternalEnergyFromDensityTemperature(rho[i], temp[i], [ye[i], 0])
        press[i] = eos.PressureFromDensityTemperature(rho[i], temp[i], [ye[i], 0])
        u[i] = eps[i] * rho[i]

    prof = np.column_stack([
        r,
        vel,
        rho,
        press,
        ye,
        temp,
        eps,
        np.zeros(rho.size), # omega
        np.zeros(rho.size), # abar
        np.zeros(rho.size)  # zbar
    ])

    if verbose: print(f'>>> generated homologous collapse profile (Goldreich & Weber 1980) with mass {M/msun:.2f} msun and radius {R:.2e} cm. using an {eostype} EOS.')

    return prof

def gen_homologous_yahil( ):
    # TODO: implement this later? not sure if we need a demo of the Yahil 1983 collapse as well.
    pass


# -----------------------------------------------------------------
# original functionality to generate GW profiles, as written by Mariam

def solvef(rs, lmbda):  # Solves the second order equation for density normalization function f (eq. 16)
    def func(u, x):
        return (u[1], -2.0 / x * u[1] - u[0] ** 3.0 + lmbda)

    f0 = [1, 0]
    fs = []
    r = []
    us = odeint(func, f0, rs)
    for i in range(len(rs)):
        if us[i, 0] < 0:
            break
        fs.append(us[i, 0])
        r.append(rs[i])

    return np.array(r), fs

def FindJeansLength(r, R, lmbda, k):  # eq. 15
    a = [R / r[len(r) - 1]]
    a = np.array(a)
    t = (a / (6.0 * lmbda) ** (1.0 / 3.0) * (np.pi * G / k ** 3.0) ** (1.0 / 6.0)) ** (
        3.0 / 2.0
    )
    adot = (
        2.0
        / 3.0
        * (6.0 * lmbda) ** (1.0 / 3.0)
        * (k ** 3.0 / (math.pi * G)) ** (1.0 / 6.0)
        * t ** (-1.0 / 3.0)
    )
    return a, adot

def CalculateK(M, f, r):
    I = 0
    for i in range(len(r) - 1):
        I = I + (
            f[i + 1] ** 3.0 * r[i + 1] ** 2.0 + f[i] ** 3.0 * r[i] ** 2.0
        ) / 2.0 * (r[i + 1] - r[i])
    k = math.pi * G * (M / (4.0 * math.pi * I)) ** (2.0 / 3.0)
    return k

def CalculateDensity(a, f, k):  # eq. 10
    rho = np.zeros([len(a), len(f)])
    for i in range(len(a)):
        for j in range(len(f)):
            rho[i][j] = (
                (k / (math.pi * G)) ** (3.0 / 2.0) * a[i] ** (-3.0) * f[j] ** 3.0
            )
    rhoc = max(rho[0, :])
    for i in range(len(a)):
        for j in range(len(f)):
            if rho[i][j] / rhoc < 1.0e-5:
                rho[i][j] = 1.0e-5 * rhoc
    return np.reshape(rho, rho.shape[1])

def CalculateVelocity(r, adot):  # u=adot*r (r-component)
    u = np.zeros([len(adot), len(r)])
    for i in range(len(adot)):
        for j in range(len(r)):
            u[i][j] = adot[i] * r[j]
    return np.reshape(u, u.shape[1])

def CalculateGravPotential(r, a, f, lmbda, k):
    phi = np.zeros([len(a), len(r)])
    for i in range(len(a)):
        for j in range(len(r)):
            psi = 1 / 2.0 * lmbda * r[j] ** 2.0 - 3.0 * f[j]
            phi[i][j] = 4.0 / 3.0 * (k ** 3 / (math.pi * G)) ** (1.0 / 2.0) * psi / a[i]
    return phi

def CalculateSpecificInternalEnergy(rho, k, gamma):
    eps = np.zeros(rho.size)
    for i in range(rho.size):
        eps[i] = k / (gamma - 1.0) * rho[i] ** (gamma - 1.0)
    return eps

def CalculatePressure(rho, k, gamma):
    press = k * rho ** gamma
    return press

def CalculateTemperature(cv, eps):
    return eps / cv