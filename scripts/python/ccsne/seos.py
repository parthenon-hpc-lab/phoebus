from singularity_eos import StellarCollapse, Helmholtz
import numpy as np


def calculate_eos_energy_stellarcollapse(rho: np.ndarray, T: np.ndarray, ye: np.ndarray, filename: str ):

    '''
    Calculates an EOS-consistent specific internal energy for a stellar/ccsne progenitor profile using a StellarCollpase equation
    of state and ``Singularity-EOS`` Python bindings.

    Args:
        rho (np.ndarray): Density of the profile.
        T (np.ndarray): Temperature of the profile.
        ye (np.ndarray): Electron fraction of the profile.
        filename (str): Full filename and path pointing to the *.h5 tabulated equation of state.

    Returns:
        (tuple): tuple containing:

            eps (np.ndarray): Specific internal energy of the profile.
            u (np.ndarray): Energy density of the profile.

    '''

    eos = StellarCollapse(filename, use_sp5=False, filter_bmod=True)
    nlambda = eos.nlambda  # get number of elements per lambda
    lmbda = np.zeros(nlambda, dtype=np.double)
    u = np.zeros(len(rho))
    eps = np.zeros(len(rho))

    for i in range(len(rho)):
        lmbda[0] = ye[i]
        eps[i] = eos.InternalEnergyFromDensityTemperature(rho[i], T[i], lmbda)
        u[i] = eps[i] * rho[i]

    return eps, u

def calculate_eos_energy_helmholtz(rho: np.ndarray, T: np.ndarray, abar: np.ndarray, zbar: np.ndarray, filename: str):

    '''
    Calculates an EOS-consistent specific internal energy for a stellar/ccsne progenitor profile using a Helmholtz equation
    of state and ``Singularity-EOS`` Python bindings.

    Args:
        rho (np.ndarray): Density of the profile.
        T (np.ndarray): Temperature of the profile.
        ye (np.ndarray): Electron fraction of the profile.
        filename (str): Full filename and path pointing to the *.h5 tabulated equation of state.

    Returns:
        (tuple): tuple containing:
        
            eps (np.ndarray): Specific internal energy of the profile.
            u (np.ndarray): Energy density of the profile.

    '''
        
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

    return eps, u


def get_eos_bounds( filename: str, eos_type: str = 'stellarcollapse') -> np.ndarray:

    '''
    Calculates the upper and lower bounds (in density, temperature, specific internal energy, and electron fraction)
    of a tabulated Stellar Collapse equation of state. 

    Args:
        filename (str): Full filename and path pointing to the *.h5 tabulated equation of state.
        eos_type (str, optional): The type of tabulated EOS. Default is 'StellarCollapse'.

    Returns:
        np.ndarray: EOS bounds, in the following order: [rhoMin, rhoMax, TMin, TMax, sieMin, sieMax, YeMin, YeMax]

    '''

    if eos_type.lower().strip() == 'stellarcollapse':
        eos = StellarCollapse(filename, use_sp5=False, filter_bmod=True)
        return np.asarray([eos.rhoMin, eos.rhoMax, eos.TMin, eos.TMax, eos.sieMin, eos.sieMax, eos.YeMin, eos.YeMax])
    
    elif eos_type.lower().strip() == 'helmholtz':
        raise LookupError(f'the Helmholtz EOS has no infrastructure to retrieve bounds through python bindings. please find bounds manually.')

    else:
        raise NameError(f'no such EOS found under the type or name `{eos_type}`.')
    
