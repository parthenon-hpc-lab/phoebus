'''
    main driver for the ccsne conversion pipeline, intended to be lightweight and user friendly.
    one should be able to just run this with some combination of flags/user-input arguments and
    get a raw profile, phoebus profile (cgs, optional), and phoebus profile (code units).

    law. 23 jun 2026
'''

# useful i/o things
import os
from argparse import ArgumentParser

# our custom helper classes/methods
import progenitors
import adm
import convert


def main( ):

    # TODO: update this so that we can read in all arguments from a single text file (e.g. *.pin) 
    # can be done using custom argparse.Action ?? ref. stack overflow.
    # could also consider condensing some of the options available, mainly custom grid, etc.
    
    parser = ArgumentParser(
        prog='ccsne',
        description=''
    )
    # ----- progenitor read in 
    # model name, type, header (optional), at bounce (optional, gr1d), time (if not at bounce)
    # path to data, output path (optional)
    parser.add_argument('-n', '--model-name')
    parser.add_argument('-t', '--model-type')
    parser.add_argument('--header')
    parser.add_argument('-b', '--atbounce')


    # ----- progenitor processing
    # wma + options, interp factor, use_drad + drad (optional)


    # adm solver options
    # 

    # grid construction methods
    parser.add_argument('-rc', '--radcut')
    parser.add_argument('-dc', '--rhocut')
    parser.add_argument('-gc', '--custom-grid')

    parser.add_argument('-ep', '--eos-path')
    parser.add_argument('-et', '--eos-type')

    parser.add_argument('-i', '--interp')

    parser.add_argument('-z', '--zones') # desired zones in new unif grid


    # get initial profile
    # process/interpolate to eulerian grid
    # save raw, pre ADM profile (progenitors.py)
    # initailize ADM solver, solve for new profile
    # save phoebus profiles (progenitors.py)
        # optional: make summary file

