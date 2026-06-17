from singularity_eos import StellarCollapse, Helmholtz
import numpy as np


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