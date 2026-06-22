'''
    refactor of the original convert.py that allows for conversion of phoebus profiles 
    as a numpy array (instead of extra i/o with files), and also creates an output 
    file with useful diagnostics/metrics about the model to help with *.pin file 
    creation and provide reasonable floors/ceilings.

'''
import numpy as np
from astropy import constants as const

from seos import get_EOS_bounds

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

    return prof_conv


def make_summary_file( rhoc: float, M0: float, R0: float, prof_conv: np.ndarray, model_name: str, model_type: str, eos_type = "stellarcollapse" ) -> None:
    
    try:
        fout = open(f'{model_name.lower()}_{model_type.lower()}_summary.dat', 'w')
        fmt   = '%30s  %18.13e\n'

        fout.write(f'# -----summary and conversions from {model_type.upper()} progenitor {model_name}.')

        # --- characteristic quantities, length scales
        fout.write('# -----characteristic values [cgs]\n')
        fout.write( ('central density', rhoc ) % fmt)
        fout.write( ('characteristic (geom) mass', M0 ) % fmt)
        fout.write( ('characteristic radius', R0 ) % fmt )
        fout.write('\n')
        fout.write('-----length scales [phb]\n')
        fout.write( ('radius, 500 km', 500e5/R0) % fmt)
        fout.write( ('radius, 1e4 km', 1e9/R0) % fmt)

        # --- model bounds
        fout.write('# -----bounds, progenitor [phb]\n')
        fout.write( ('minimum density', np.min(prof_conv[:, 1] )) % fmt )
        fout.write( ('maximum density', np.max(prof_conv[:, 1] )) % fmt )
        fout.write( ('minimum temperature', np.min(prof_conv[:, 2] )) % fmt )
        fout.write( ('maximum temperature', np.max(prof_conv[:, 2] )) % fmt )
        fout.write( ('minimum sie', np.min(prof_conv[:, 4] )) % fmt )
        fout.write( ('maximum sie', np.max(prof_conv[:, 4] )) % fmt )


        # --- eos bounds
        bounds = get_EOS_bounds( eos_type )

        fout.write('# -----bounds, progenitor [phb]\n')
        fout.write( ('minimum density', bounds[0]) % fmt )
        fout.write( ('maximum density', bounds[1]) % fmt )
        fout.write( ('minimum temperature', bounds[2]) % fmt )
        fout.write( ('maximum temperature', bounds[3]) % fmt )
        fout.write( ('minimum sie', bounds[4]) % fmt )
        fout.write( ('maximum sie', bounds[5]) % fmt )
        fout.write( ('minimum ye', bounds[6]) % fmt )
        fout.write( ('maximum ye', bounds[7]) % fmt )

        # --- conversion factors
        fout.write('# -----conversion factors [phb -> cgs]\n')
        fout.write('# (multiply to get back to cgs)\n')
        fout.write( ('radius', 1/R0 ) % fmt)
        fout.write( ('density', 1/((c ** 2.0) / G / M0)) % fmt)
        fout.write( ('temperature', 1/kB, 'kB') % fmt)
        fout.write( ('sie', c**2.0) % fmt)
        fout.write( ('velocity', c ) % fmt)
        fout.write( ('pressure', 1 / (G ** 3.0 / c ** 8.0 * M0 ** 2.0) ) % fmt)

        fout.close()

    except:
        raise IOError(f'unable to write summary file for {model_type.upper()} model {model_name}.')