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


def convert_PHB_profile(prof: np.ndarray) -> np.ndarray:

    # the input profile has the following columns:
    # 'radius [cm]', 'density [g/cm^3]', 'temperature [K]',  'ye', 'sie [erg/g]', 'velocity [cm/s]','pressure [dyne/cm^2]',  'density [ADM]', 'pressure [ADM]', 'S [ADM]', 'S_rr [ADM]'
    # we need to convert both the primitive and the ADM quantities

    prof_conv = np.copy(
        prof
    )  # making a fresh copy of the profile so we don't overwrite the physical units.

    rhoc = prof_conv[0, 1]  # central density
    M0 = 1.0 / (rhoc**0.5) * ((c**2.0 / G) ** 1.5)  # characteristic mass
    R0 = (G * M0) / (c**2.0)  # characteristic radius

    # --- primitive quantities
    prof_conv[:, 0] *= (c**2.0) / G / M0  # radius
    prof_conv[:, 1] *= (G / c**2.0) ** 3.0 * M0**2.0  # mass density
    prof_conv[:, 2] *= kB  # temperature
    # note that we don't convert ye, it's unitless!
    prof_conv[:, 4] /= c**2.0  # specific internal energy
    prof_conv[:, 5] /= c  # velocity
    prof_conv[:, 6] *= G**3.0 / c**8.0 * M0**2.0  # pressure

    # --- ADM quantities
    prof_conv[:, 7] *= (G / c**2.0) ** 3.0 * M0**2.0  # adm density
    prof_conv[:, 8] *= G**3.0 / c**7.0 * M0**2.0  # adm momentum
    prof_conv[:, 9] *= G**3.0 / c**8.0 * M0**2.0  # adm entropy (?)
    prof_conv[:, 10] *= G**3.0 / c**8.0 * M0**2.0  # adm entropy, radial component (?)

    return prof_conv, rhoc, M0, R0


def make_info_file(
    rhoc: float,
    M0: float,
    R0: float,
    prof_conv: np.ndarray,
    model_name: str,
    model_type: str,
    EOSPATH: str,
    eos_type='stellarcollapse',
    OUTPATH='',
) -> None:

    try:
        fout = open(
            os.path.join(OUTPATH, f'{model_name}_{model_type.lower()}.info'), 'w'
        )
        fmt = '%18.13e    %-25s\n'

        fout.write(
            f'# -----info, units, and conversions from {model_type.upper()} progenitor `{model_name}.`'
        )
        fout.write('\n\n')

        # --- characteristic quantities, length scales
        fout.write('# -----characteristic values [cgs]\n')
        fout.write(fmt % (rhoc, 'central density'))
        fout.write(fmt % (M0, 'characteristic mass'))
        fout.write(fmt % (R0, 'characteristic radius'))
        fout.write('\n')
        fout.write('# -----length scales [phb]\n')
        fout.write(fmt % (500e5 / R0, 'radius, 500 km'))
        fout.write(fmt % (1e9 / R0, 'radius, 1e4 km'))
        fout.write('\n')

        # --- model bounds
        fout.write('# -----bounds, progenitor [phb]\n')
        fout.write(fmt % (np.min(prof_conv[:, 1]), 'minimum density'))
        fout.write(fmt % (np.max(prof_conv[:, 1]), 'maximum density'))
        fout.write(fmt % (np.min(prof_conv[:, 2]), 'minimum temperature'))
        fout.write(fmt % (np.max(prof_conv[:, 2]), 'maximum temperature'))
        fout.write(fmt % (np.min(prof_conv[:, 4]), 'minimum sie'))
        fout.write(fmt % (np.max(prof_conv[:, 4]), 'maximum sie'))
        fout.write('\n')

        # --- eos bounds
        bounds = get_eos_bounds(EOSPATH, eos_type)

        fout.write('# -----bounds, equation of state [phb]\n')
        fout.write(fmt % (bounds[0] / (1 / ((c**2.0) / G / M0)), 'minimum density'))
        fout.write(fmt % (bounds[1] / (1 / ((c**2.0) / G / M0)), 'maximum density'))
        fout.write(fmt % (bounds[2] * kB, 'minimum temperature'))
        fout.write(fmt % (bounds[3] * kB, 'maximum temperature'))
        fout.write(fmt % (bounds[4] / (c**2.0), 'minimum sie'))
        fout.write(fmt % (bounds[5] / (c**2.0), 'maximum sie'))
        fout.write(fmt % (bounds[6], 'minimum ye'))
        fout.write(fmt % (bounds[7], 'maximum ye'))
        fout.write('\n')

        # --- conversion factors
        fout.write(
            '# -----conversion factors, multiply to get back to cgs [phb -> cgs]\n'
        )
        fout.write(fmt % (R0, 'radius'))
        fout.write(fmt % (1 / ((c**2.0) / G / M0), 'density'))
        fout.write(fmt % (1 / kB, 'temperature'))
        fout.write(fmt % (c**2.0, 'sie'))
        fout.write(fmt % (c, 'velocity'))
        fout.write(fmt % (1 / (G**3.0 / c**8.0 * M0**2.0), 'pressure'))
        fout.write('\n')

        fout.close()

    except:
        raise IOError(
            f'unable to write summary file for {model_type.upper()} model `{model_name}`.'
        )
