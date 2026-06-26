'''
    refactor of the original convert.py that allows for conversion of phoebus profiles 
    as a numpy array (instead of extra i/o with files), and also creates an output 
    file with useful diagnostics/metrics about the model to help with *.pin file 
    creation and provide reasonable floors/ceilings.

'''
import os
import numpy as np
from astropy import constants as const

from seos import get_eos_bounds

# useful constants, in cgs
G = const.G.cgs.value
c = const.c.cgs.value
kB = const.k_B.cgs.value
msun = const.M_sun.cgs.value

def convert_PHB_profile( prof: np.ndarray ) -> np.ndarray:

    # the input profile has the following columns:
    # 'radius [cm]', 'density [g/cm^3]', 'temperature [K]',  'ye', 'sie [erg/g]', 'velocity [cm/s]','pressure [dyne/cm^2]',  'density [ADM]', 'pressure [ADM]', 'S [ADM]', 'S_rr [ADM]'
    # we need to convert both the primitive and the ADM quantities

    prof_conv = np.copy(prof) # making a fresh copy of the profile so we don't overwrite the physical units.

    rhoc = prof_conv[0, 1] # central density
    M0 = 1.0 / (rhoc ** 0.5) * ((c ** 2.0 / G) ** 1.5) # characteristic mass
    R0 = (G * M0) / (c ** 2.0) # characteristic radius

    # --- primitive quantities
    prof_conv[:, 0] *= (c ** 2.0) / G / M0                  # radius
    prof_conv[:, 1] *= (G / c ** 2.0) ** 3.0 * M0 ** 2.0    # mass density
    prof_conv[:, 2] *= kB                                   # temperature
                                                        # note that we don't convert ye, it's unitless!
    prof_conv[:, 4] /= c ** 2.0                             # specific internal energy
    prof_conv[:, 5] /= c                                    # velocity
    prof_conv[:, 6] *= G ** 3.0 / c ** 8.0 * M0 ** 2.0      # pressure

    # --- ADM quantities
    prof_conv[:, 7] *= (G / c ** 2.0) ** 3.0 * M0 ** 2.0    # adm density
    prof_conv[:, 8] *= G ** 3.0 / c ** 7.0 * M0 ** 2.0      # adm momentum
    prof_conv[:, 9] *=  G ** 3.0 / c ** 8.0 * M0 ** 2.0     # adm entropy (?)
    prof_conv[:, 10] *= G ** 3.0 / c ** 8.0 * M0 ** 2.0     # adm entropy, radial component (?)

    return prof_conv, rhoc, M0, R0


def make_summary_file( rhoc: float, M0: float, R0: float, prof_conv: np.ndarray, model_name: str, model_type: str, EOSPATH: str, eos_type = "stellarcollapse", OUTPATH = '') -> None:
    
    try:
        fout = open(os.path.join(OUTPATH, f'{model_name}_{model_type.lower()}.dat'), 'w')
        fmt   = '%22s  %-18.13e\n'

        fout.write(f'# -----summary and conversions from {model_type.upper()} progenitor `{model_name}.`')
        fout.write('\n\n')

        # --- characteristic quantities, length scales
        fout.write('# -----characteristic values [cgs]\n')
        fout.write( fmt % ('central density', rhoc ))
        fout.write( fmt % ('characteristic mass', M0 ))
        fout.write( fmt % ('characteristic radius', R0 ) )
        fout.write('\n')
        fout.write('# -----length scales [phb]\n')
        fout.write( fmt % ('radius, 500 km', 500e5/R0))
        fout.write( fmt % ('radius, 1e4 km', 1e9/R0))
        fout.write('\n')

        # --- model bounds
        fout.write('# -----bounds, progenitor [phb]\n')
        fout.write( fmt % ('minimum density', np.min(prof_conv[:, 1] )) )
        fout.write( fmt % ('maximum density', np.max(prof_conv[:, 1] )) )
        fout.write( fmt % ('minimum temperature', np.min(prof_conv[:, 2] )) )
        fout.write( fmt % ('maximum temperature', np.max(prof_conv[:, 2] )) )
        fout.write( fmt % ('minimum sie', np.min(prof_conv[:, 4] )) )
        fout.write( fmt % ('maximum sie', np.max(prof_conv[:, 4] )) )
        fout.write('\n')


        # --- eos bounds
        bounds = get_eos_bounds( EOSPATH, eos_type )

        fout.write('# -----bounds, progenitor [phb]\n')
        fout.write( fmt % ('minimum density', bounds[0]) )
        fout.write( fmt % ('maximum density', bounds[1]) )
        fout.write( fmt % ('minimum temperature', bounds[2]) )
        fout.write( fmt % ('maximum temperature', bounds[3]) )
        fout.write( fmt % ('minimum sie', bounds[4]) )
        fout.write( fmt % ('maximum sie', bounds[5]) )
        fout.write( fmt % ('minimum ye', bounds[6]) )
        fout.write( fmt % ('maximum ye', bounds[7]) )
        fout.write('\n')

        # --- conversion factors
        fout.write('# -----conversion factors, multiply to get back to cgs [phb -> cgs]\n')
        fout.write( fmt % ('radius', 1/R0 ))
        fout.write( fmt % ('density', 1/((c ** 2.0) / G / M0)))
        fout.write( fmt % ('temperature', 1/kB ))
        fout.write( fmt % ('sie', c**2.0))
        fout.write( fmt % ('velocity', c ))
        fout.write( fmt % ('pressure', 1 / (G ** 3.0 / c ** 8.0 * M0 ** 2.0) ))
        fout.write('\n')

        fout.close()

    except:
        raise IOError(f'unable to write summary file for {model_type.upper()} model `{model_name}`.')