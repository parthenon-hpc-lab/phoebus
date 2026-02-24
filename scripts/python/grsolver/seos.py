from singularity_eos import StellarCollapse, thermalqs, Helmholtz
import numpy as np
import math
import matplotlib.pyplot as pl
from scipy.optimize import root, bisect

SMALL = 1e-20  # to avoid dividing by zero


def FindTemperature(rho, p, lmbda, tmin, tmax, eos):
    def f(x):
        return eos.PressureFromDensityTemperature(rho, x, lmbda) - p

    sol = bisect(f, tmin, tmax)
    T = sol
    return T


def CalculateInternalEnergy_DEPR(rho, ye, p, tmin, tmax, filename):
    eos1 = StellarCollapse(filename, use_sp5=False, filter_bmod=True)
    nlambda = eos1.nlambda  # get number of elements per lambda
    lmbda = np.zeros(nlambda, dtype=np.double)
    u = np.zeros(len(rho))
    eps = np.zeros(len(rho))
    T = np.zeros(len(rho))
    for i in range(len(rho)):
        lmbda[0] = ye[i]
        T[i] = FindTemperature(rho[i], p[i], lmbda, tmin, tmax, eos1)
        eps[i] = eos1.InternalEnergyFromDensityTemperature(rho[i], T[i], lmbda)
        u[i] = eps[i] * rho[i]
    return eps, u, T

def CalculateInternalEnergy_StellarCollapse(rho, T, ye, filename, verbose=False):
    eos1 = StellarCollapse(filename, use_sp5=False, filter_bmod=True)
    nlambda = eos1.nlambda  # get number of elements per lambda
    lmbda = np.zeros(nlambda, dtype=np.double)
    u = np.zeros(len(rho))
    eps = np.zeros(len(rho))

    for i in range(len(rho)):
        lmbda[0] = ye[i]
        eps[i] = eos1.InternalEnergyFromDensityTemperature(rho[i], T[i], lmbda)
        u[i] = eps[i] * rho[i]

    # if verbose: print(nlambda, lmbda)
    return eps, u

def CalculateInternalEnergy_Helmholtz(rho, T, abar, zbar, filename='/home/u1/lawhite/singularity-eos/data/helmholtz/helm_table.dat', verbose=False):
    eos1 = Helmholtz(filename)
    nlambda = eos1.nlambda  # get number of elements per lambda
    lmbda = np.zeros(nlambda, dtype=np.double)
    u = np.zeros(len(rho))
    eps = np.zeros(len(rho))

    for i in range(len(rho)):
        lmbda[0] = abar[i]
        lmbda[1] = zbar[i]
        eps[i] = eos1.InternalEnergyFromDensityTemperature(rho[i], T[i], lmbda)
        u[i] = eps[i] * rho[i]

    # if verbose: print(nlambda, lmbda)
    return eps, u