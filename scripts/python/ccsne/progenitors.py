'''
    new skeleton for a unified library that can handle retrieval and formatting 
    of progenitor data for a unified case; should manage profiles taken from 
    the following stellar evolution/ccsne codes:
        > MESA
        > KEPLER
        > GR1D (default is at bounce)

    also contains some helper utilities for reading in GR1D time series data and bounce times.

    >>> outstanding tasks

        TODO: add documentation to all helper functions
        TODO: do we add the processing functions (subsample, weighted moving avg) here?
        TODO: testing! maybe unit tests too
        
    law. 16 jun 2026
'''

from astropy import constants as const
import pandas as pd
import numpy as np
import os

# we won't need this after testing
from scipy.interpolate import CubicSpline as cubic


from convert import convert_PHB_profile, make_summary_file


def get_MESA_profile( PATH: str, header=4, verbose=False ) -> np.ndarray:

    # loading in the profile using pandas dataframe
    try: 
         prof = pd.read_csv(PATH, sep=r"\s+", header=header)
    except: 
         raise FileNotFoundError(f'MESA profile not found at {PATH}.')

    # reversing the profile so that our inner radial coord is at 0.
    prof = prof.iloc[::-1].reset_index(drop=True)

    # quick check to see if our model was rotating or not:
    try: 
         omega = prof['omega']
    except KeyError:
        omega = np.zeros( len(prof) )
        print('>>> angular velocity not found, setting to zero.')
        
    # new numpy array of needed quantities.
    profnp = np.column_stack([
                prof['radius_cm'],
                prof['velocity'],
                prof['density'],
                prof['pressure'],
                prof['ye'],
                prof['temperature'],
                prof['energy'],
                omega,
                prof['abar'],
                prof['zbar']
            ])
    
    return profnp


def get_KEPLER_profile( PATH: str, header=0, verbose=False ) -> np.ndarray:

    # loading in the profile using pandas dataframe
    try: 
         prof = pd.read_csv(PATH, sep=r"\s+", header=header)
    except: 
         raise FileNotFoundError(f'KEPLER profile not found at {PATH}.')


    # quick check to see if our model was rotating or not:
    try: 
         omega = prof['cell angular velocity']
    except KeyError:
        omega = np.zeros( len(prof) )
        print('>>> angular velocity not found, setting to zero.')
        
    # new numpy array of needed quantities.
    profnp = np.column_stack([
                prof['cell outer radius'],
                prof['cell outer velocity'],
                prof['cell density'],
                prof['cell pressure'],
                prof['cell Y_e'],
                prof['cell temperature'],
                prof['cell specific energy'],
                omega,
                prof['cell A_bar'],
                prof['cell Z_bar']
            ])
    
    return profnp


def get_GR1D_profile( PATH: str, at_bounce = True, time = -1, verbose = False ) -> np.ndarray:
    
    try:

        if at_bounce: # pulls the bounce time from GR1D's default output
            time = get_tbounce( PATH )
    
        rho_data, rho_times     = read_time_series( os.path.join(PATH,'rho.xg') )
        eps_data, eps_times     = read_time_series( os.path.join(PATH,'eps.xg') )  # assuming entropy.xg holds sie
        temp_data, temp_times   = read_time_series( os.path.join(PATH,'temperature.xg') )
        ye_data, ye_times       = read_time_series( os.path.join(PATH,'ye.xg') )
        p_data, p_times         = read_time_series( os.path.join(PATH,'press.xg') )
        v_data, v_times         = read_time_series( os.path.join(PATH,'v.xg') )
        v_data, v_times         = read_time_series( os.path.join(PATH,'v.xg') )
        zbar_data, zbar_times   = read_time_series( os.path.join(PATH,'xzbar.xg') )
        abar_data, abar_times   = read_time_series( os.path.join(PATH,'xabar.xg') )

    except: 
         raise FileNotFoundError(f'GR1D *.xg, *.dat files not found at {PATH}.')

    try: 
        omg_data, omg_times   = read_time_series( os.path.join(PATH,'omega.xg') )
        has_omega = True
    except:
        has_omega = False
        print('>>> angular velocity not found, setting to zero.')
    
    if has_omega:  
        times_common = set(rho_times) & set(eps_times) & set(temp_times) & set(ye_times) & set(p_times) & set(v_times) & set(zbar_times) & set(abar_times) & set(omg_times)
    else:
        times_common = set(rho_times) & set(eps_times) & set(temp_times) & set(ye_times) & set(p_times) & set(v_times) & set(zbar_times) & set(abar_times)
    
    # Ensure all time series contain the time
    if not times_common: raise ValueError('no overlapping times in all files.')
    
    nearest_time = find_nearest_time( time, sorted(times_common) )

    radius, rho = rho_data[nearest_time]
    _, eps      = eps_data[nearest_time]
    _, temp     = temp_data[nearest_time]
    _, ye       = ye_data[nearest_time]
    _, press    = p_data[nearest_time]
    _, vel      = v_data[nearest_time]
    _, zbar     = zbar_data[nearest_time]
    _, abar     = abar_data[nearest_time]

    if has_omega: 
        _, omega = omg_data[nearest_time]
    else: 
        omega = np.zeros_like(radius)

    profnp = np.column_stack([
                radius,
                vel,
                rho,
                press,
                ye,
                temp / const.k_B.to('MeV/K').value, # converting back to kelvin
                eps,
                omega,
                abar,
                zbar
            ])
    
    return profnp, time


### ---------------------------------------------------------------
### ------------ data handling utilities for going from lagrangian-style grids (most 1d codes, mass coord) to eulerian resolution for phoebus

# takes a weighted moving average of data, intended for smoothing of velocity profiles after subsampling...
def wmavg( data, n=1, exp=1, use_inverse=True, avg_edges=True, verbose=False):
    datavg = np.copy(data)

    # constructing our array of weights
    if use_inverse:
        ws = np.ones(n) / ((np.arange(n) + 2) ** exp) 
        weights = np.concatenate((ws[::-1], [1], ws), axis=0)
    else:
        ws = np.arange(n) + 1
        weights = np.concatenate((ws[::-1], [1], ws), axis=0)
    if verbose: print(weights)

    # applying weights to the data from indices n, i - n.    
    for i in range(n, data.size - n): 
            datavg[i] = np.sum(data[i - n: i + 1 + n] * weights) / np.sum(weights)

    # handling averaging from the edges inward (i.e. the first n zones and the last i - n zones)
    # if not enabled, you can get weird edge effects or discontinuities!
    if avg_edges:
        
        for i in range(n):
            datavg[i] = np.sum(data[: (2*i) + 1] * weights[n - i: n + i + 1]) / np.sum(weights[n - i: n + i + 1])
            datavg[-(i + 1)] = np.sum(data[-(2*i) - 1:] * weights[n - i: n + i + 1]) / np.sum(weights[n - i: n + i + 1])
            if verbose:
                print(i, data[: (2*i) + 1], weights[n - i: n + i + 1])
                print(i, data[-(2*i) - 1:], weights[n - i: n + i + 1])
        
    return datavg

def interp_to_eulerian( prof, factor = 4, use_drad = False, drad = 1e7):

    nzones  = prof.shape[0]
    rad0     = prof[:, 0] # we assume the whole profile is passed in.
    drad0   = (rad0[-1] - rad0[0]) / (nzones * factor)

    # if we want to ensure a certain initial resolution in the profiles, e.g. dr = 1e7 cm
    if use_drad:
        while drad0 > drad:
            factor += 1
            drad0 = (rad0[-1] - rad0[0]) / (nzones * factor)
    
    rad = np.linspace(rad0[0], rad0[-1], nzones * factor)

    prof_interp = np.zeros( (nzones * factor, prof.shape[1]) )
    prof_interp[:, 0] = rad

    for i in range(1, prof.shape[1]):
        prof_interp[:, i] = np.interp(rad, rad0, prof[:, i])

    return prof_interp


# KEEP FOR GENERAL TESTING/V&V!!
# old method of sampling
def subsample_depr( x, radius_cm, factor=4, verbose=False, get_factor=False, uniform=False):
    nzones = x.size # number of zones within the mesa raw data

    if uniform: 
        total_radius = radius_cm.iloc[-1] - radius_cm.iloc[0] # total radial domain of the mesa profile
        dr = total_radius / (nzones * factor) # << new uniform radiial spacing
        
        if get_factor: 
            while dr > 1e7:
                factor += 1
                dr = total_radius / (nzones * factor)

        if verbose:
            print(f'new radial spacing:\t{dr:1.4e}')
            print(f'zone factor:\t{factor}')

        new_grid = np.linspace(radius_cm.iloc[0], radius_cm.iloc[-1], nzones * factor)
    
    else:
        unif_radius = 5e9 - radius_cm.iloc[0] 
        unif_nzones = int(unif_radius / 2e7) # ensuring high resolution within the core to capture < 5e9 cm
        log_nzones = int(nzones * 0.70 * factor) # assuming 70% of the zones don't capture the core, multiplying by the general factor
        
        unif_grid = np.linspace(radius_cm.iloc[0], 5e9, unif_nzones)
        log_grid = np.logspace( np.log10(5e9), np.log10(radius_cm.iloc[-1]), log_nzones)
        new_grid = np.concatenate((unif_grid, log_grid[1:]), axis=0)

        if verbose:
            print(f'new uniform spacing:\t{unif_radius / unif_nzones :1.4e}')
            print(f'new log spacing:\t{np.log10( log_grid[1] - log_grid[0]):1.4e}')
            print(f'zone factor (log):\t{factor}')

    cinterp = cubic( radius_cm, x, extrapolate = None)
    
    if get_factor:
        return cinterp(new_grid), factor
    return cinterp(new_grid)


### ---------------------------------------------------------------
### ------------ handles saving profiles to either adm.py input or phoebus style input (in physical and code units)

def save_raw_profile( profile: np.ndarray, model_name: str, model_type: str, time = 0.0, OUTPATH = '', verbose = True ):
    
    # formatting for the header (to align with columns)
    fmt_header = '\t'.join(['%-20s'] * 10)
    tup_header = ( 'radius [cm]', 'velocity [cm/s]', 'density [g/cm^3]', 'pressure [dyne/cm^2]', 'ye', 'temperature [K]', 'sie [erg/g]', 'omega [rad/s]', 'abar', 'zbar')

    np.savetxt(
        os.path.join(OUTPATH, f'{model_name}_{model_type.lower()}.prof'),
        profile,
        delimiter   ='\t',
        fmt         = '%20.15e',
        header      = f'initial {model_type.upper()} profile from model `{model_name}` at time {time:.4f} s.\n{fmt_header % tup_header}'
    )

    if verbose: print(f'>>> saved raw profile from {model_type.upper()} model `{model_name}` at time {time:.4f} s to {OUTPATH}.')


def save_ADM_profile( profile: np.ndarray, model_name: str, model_type: str, EOSPATH: str, eos_type = 'stellarcollapse', time = 0.0, OUTPATH = '', save_unconverted = True, save_summary = True, verbose = True ):

    # formatting for header
    fmt_header = '\t'.join(['%-20s'] * 11)
    tup_header = ( 'radius [cm]', 'density [g/cm^3]', 'temperature [K]',  'ye', 'sie [erg/g]', 'velocity [cm/s]','pressure [dyne/cm^2]', 'density [ADM]', 'pressure [ADM]', 'S [ADM]', 'S_rr [ADM]')
    tup_header_conv = ( 'radius', 'density', 'temperature',  'ye', 'sie', 'velocity','pressure', 'density [ADM]', 'pressure [ADM]', 'S [ADM]', 'S_rr [ADM]')
    
    # saves the unconverted profile with primitives + ADM quantities
    if save_unconverted:
        np.savetxt(
            os.path.join(OUTPATH, f'{model_name}_{model_type.lower()}_adm.prof'),
            profile,
            delimiter   ='\t',
            fmt         = '%20.15e',
            header      = f'primitives + ADM for {model_type.upper()} profile from model `{model_name}`at time {time:.4f} s.\n{fmt_header % tup_header}'
        )
        if verbose: print(f'>>> saved primitive + ADM profile from {model_type.upper()} model `{model_name}` at time {time:.4f} s to {OUTPATH}.')


    # converting the actual profile to phoebus code units
    profile_conv, rhoc, M0, R0 = convert_PHB_profile( profile )

    # creates a summary file with useful conversions and progenitor bounds
    if save_summary:
        make_summary_file( rhoc, M0, R0, profile_conv, model_name, model_type, EOSPATH, eos_type, OUTPATH)

    # saves the converted profile
    np.savetxt(
        os.path.join(OUTPATH, f'{model_name}_{model_type.lower()}_adm_converted.prof'),
        profile_conv,
        delimiter   ='\t',
        fmt         = '%20.15e',
        header      = f'converted primitives + ADM for {model_type.upper()} profile from model `{model_name}` at time {time:.4f} s.\n{fmt_header % tup_header_conv}'
    )
    if verbose: print(f'>>> saved converted primitive + ADM profile from {model_type.upper()} model `{model_name}` at time {time:.4f} s to {OUTPATH}.')


### ---------------------------------------------------------------
### ------------ GR1D utilities for reading time series data/output

def read_time_series( PATH: str ):
    """Reads a time series file with radial profiles at each time step."""
    time_series_data = {}
    ordered_times = []

    with open(PATH, 'r') as file:
        current_time = None
        r_values, quantity_values = [], []

        for line in file:
            line = line.strip()

            if line.startswith('"Time'):
                if current_time is not None and r_values:
                    time_series_data[current_time] = (np.array(r_values), np.array(quantity_values))
                try:
                    current_time = float(line.split('=')[1].strip())
                    ordered_times.append(current_time)
                except ValueError:
                    continue
                r_values, quantity_values = [], []
            else:
                try:
                    r, quantity = map(float, line.split())
                    r_values.append(r)
                    quantity_values.append(quantity)
                except ValueError:
                    continue

        if current_time is not None and r_values:
            time_series_data[current_time] = (np.array(r_values), np.array(quantity_values))

    return time_series_data, sorted(ordered_times)

def find_nearest_time( target_time: float, times: set ):
    """Find the time closest to the requested target time."""
    return min(times, key=lambda t: abs(t - target_time))

def get_tbounce( PATH: str ) -> float:
    ''' Retrieves the time of bounce from the default GR1D output.'''
    return np.loadtxt( os.path.join(PATH, 'tbounce.dat') )[0]
