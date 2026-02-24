#!/usr/bin/env python3

"""
This script retrieves raw MESA profile data variables, reverses them to be center to surface, and outputs them in the format needed for GRSOLVER for use in Phoebus.

Usage:
    python read_MESA_for_grsolver.py 20m_cc.dat

Arguments:
    input_file: filename of MESA profile
"""

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline as cubic
from argparse import ArgumentParser


def load(file):
    df = pd.read_csv(file, sep=r"\s+", header=4)
    df = df.iloc[::-1].reset_index(drop=True)
    nzones = len(df)
    if nzones > 0:
        print("loaded file:", file, " with ", nzones, " zones.")
    else:
        print("failed to load file", file)
        exit()
    return df

# takes a weighted moving average of data, intended for smoothing of velocity profiles after subsampling...
def wmavg( data, n=1, exp=1, verbose=False, use_inverse=True, avg_edges=True):
    datavg = np.copy(data)

    if use_inverse:
        ws = np.ones(n) / ((np.arange(n) + 2) ** exp)
        weights = np.concatenate((ws[::-1], [1], ws), axis=0)
    else:
        ws = np.arange(n) + 1
        weights = np.concatenate((ws[::-1], [1], ws), axis=0)
    
    if verbose:
        print(weights)
        
    for i in range(n, data.size - n): 
            datavg[i] = np.sum(data[i - n: i + 1 + n] * weights) / np.sum(weights)

    if avg_edges:
        
        for i in range(n):
            datavg[i] = np.sum(data[: (2*i) + 1] * weights[n - i: n + i + 1]) / np.sum(weights[n - i: n + i + 1])
            datavg[-(i + 1)] = np.sum(data[-(2*i) - 1:] * weights[n - i: n + i + 1]) / np.sum(weights[n - i: n + i + 1])
            if verbose:
                print(i, data[: (2*i) + 1], weights[n - i: n + i + 1])
                print(i, data[-(2*i) - 1:], weights[n - i: n + i + 1])
        
    return datavg

def subsample( x, radius_cm, factor=4, verbose=False, get_factor=False, uniform=False):
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


def retrieve_and_save(df, file, sample=True, average_vel=True, average_all=False, n_wma=100, exp_wma=3):
    radius = df["radius_cm"]

    if sample: 
        rad_sampled, factor = subsample(radius, radius, verbose = True, get_factor=True)
        velocity = subsample(df["velocity"], radius, factor)
        # if model non-rotating, pass array of zeros
        try:
            angular_velocity = subsample(df["omega"], radius, factor)
        except KeyError:
            print("angular velocity not found, setting to zero.")
            angular_velocity = subsample(np.zeros_like((radius)), radius, factor)
        density = subsample(df["density"], radius, factor)  # g/cm^3
        pressure = subsample(df["pressure"], radius, factor)  # g/cm^3
        ye = subsample(df["ye"], radius, factor)
        temperature = subsample(df["temperature"], radius, factor)  # K
        sie = subsample(df["energy"], radius, factor)  #  ! internal energy (ergs/g)
        abar = subsample(df['abar'], radius, factor)
        zbar = subsample(df['zbar'], radius, factor)

        if average_all:
            average_vel = True
            velocity = wmavg(velocity, n=n_wma, exp=exp_wma)
            density = wmavg(density, n=n_wma, exp=exp_wma)
            pressure = wmavg(pressure, n=n_wma, exp=exp_wma)
            ye = wmavg(ye, n=n_wma, exp=exp_wma)
            temperature = wmavg(temperature, n=n_wma, exp=exp_wma)
            sie = wmavg(sie, n=n_wma, exp=exp_wma)
            abar = wmavg(abar, n=n_wma, exp=exp_wma)
            zbar = wmavg(zbar, n=n_wma, exp=exp_wma)

        if average_vel: 
            velocity = wmavg(velocity, n=n_wma, exp=exp_wma)

        radius = rad_sampled

    else: # << defaults to Carl's original script
        velocity = df["velocity"]
        # if model non-rotating, pass array of zeros
        try:
            angular_velocity = df["omega"]
        except KeyError:
            print("angular velocity not found, setting to zero.")
            angular_velocity = np.zeros_like((radius))
        density = df["density"]  # g/cm^3
        pressure = df["pressure"]  # g/cm^3
        ye = df["ye"]
        temperature = df["temperature"]  # K
        sie = df["energy"]  #  ! internal energy (ergs/g)
        abar = df['abar']
        zbar = df['zbar']
        

    np.savetxt(
        file.split(".")[0] + str("_raw_data.txt"),
        np.column_stack(
            [
                radius,
                velocity,
                angular_velocity,
                density,
                pressure,
                ye,
                temperature,
                sie,
                abar,
                zbar
            ]
        ),
        header="MESA output from model `" + str(file.split(".")[0]) + "` | \ "
        "radius (cm) velocity (cm/s) angular_velocity (rad/s)  density (g/cm^3) pressure (dyne/cm^2) ye temperature (K) specific internal energy (erg/g) abar zbar",
    )
    return


def main():
    parser = ArgumentParser(
        description="Retrieve raw MESA profile data variables and output to format needed for GRSOLVER for use in Phoebus."
    )
    parser.add_argument(
        "file",
        type=str,
        default="file",
        help="name of the input MESA file",
    )
    args = parser.parse_args()

    df = load(args.file)
    retrieve_and_save(df, args.file)

    return


if __name__ == "__main__":
    main()
