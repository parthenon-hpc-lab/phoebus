from singularity_eos import StellarCollapse, Helmholtz
import numpy as np


def CalculateInternalEnergy_StellarCollapse(rho, T, ye, filename, verbose=False):
    eos = StellarCollapse(filename, use_sp5=False, filter_bmod=True)
    nlambda = eos.nlambda  # get number of elements per lambda
    lmbda = np.zeros(nlambda, dtype=np.double)
    u = np.zeros(len(rho))
    eps = np.zeros(len(rho))

    for i in range(len(rho)):
        lmbda[0] = ye[i]
        eps[i] = eos.InternalEnergyFromDensityTemperature(rho[i], T[i], lmbda)
        u[i] = eps[i] * rho[i]

    # if verbose: print(nlambda, lmbda)
    return eps, u

def CalculateInternalEnergy_Helmholtz(rho, T, abar, zbar, filename='/home/u1/lawhite/singularity-eos/data/helmholtz/helm_table.dat', verbose=False):
    eos = Helmholtz(filename)
    nlambda = eos.nlambda  # get number of elements per lambda
    lmbda = np.zeros(nlambda, dtype=np.double)
    u = np.zeros(len(rho))
    eps = np.zeros(len(rho))

    for i in range(len(rho)):
        lmbda[0] = abar[i]
        lmbda[1] = zbar[i]
        eps[i] = eos.InternalEnergyFromDensityTemperature(rho[i], T[i], lmbda)
        u[i] = eps[i] * rho[i]

    # if verbose: print(nlambda, lmbda)
    return eps, u


def get_EOS_bounds( filename: str, eos_type='stellarcollapse') -> np.ndarray:

    if eos_type.lower().strip('\s') == 'stellarcollapse':
        eos = StellarCollapse(filename, use_sp5=False, filter_bmod=True)
        return np.asarray([eos.rhoMin, eos.rhoMax, eos.TMin, eos.TMax, eos.sieMin, eos.sieMax, eos.YeMin, eos.YeMax])
    
    elif eos_type.lower().strip('\s') == 'helmholtz':
        raise LookupError(f'the Helmholtz EOS has no infrastructure to retrieve bounds through python bindings. please find bounds manually.')

    else:
        raise NameError(f'no such EOS found under the type or name {eos_type}.')
    
