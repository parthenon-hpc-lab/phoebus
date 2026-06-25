'''
    main driver for the ccsne conversion pipeline, intended to be lightweight and user friendly.
    one should be able to just run this with some combination of flags/user-input arguments and
    get a raw profile, phoebus profile (cgs, optional), and phoebus profile (code units).

    law. 23 jun 2026
'''

# useful i/o things
import os
import shutil
import argparse

# our custom helper classes/methods
from progenitors import get_MESA_profile, get_KEPLER_profile, get_GR1D_profile
from progenitors import wmavg, interp_to_eulerian
from progenitors import save_raw_profile, save_ADM_profile

from adm import ADMSolver
from convert import *

# helper class to make sure we can read in a file with arguments
class LoadFile( argparse.Action ):
    def __call__(self, parser, namespace, values, option_string = None):
        with open(values, 'r') as f:
            # parse arguments in the file (one per line), and store them in the target namespace
            # print( f.read().split())
            parser.parse_args(f.read().split(), namespace)


def get_params( parser ) -> None:

    # ----- progenitor read in 
    # path, model name, type, header (optional), at bounce (optional, gr1d), time (if not at bounce)
    parser.add_argument('--model-path', type=str)
    parser.add_argument('--model-name', type=str)
    parser.add_argument('--model-type', type=str)
    parser.add_argument('--model-header', type=int, default=-1)
    parser.add_argument('--atbounce', action='store_true', default=False)
    parser.add_argument('--timestamp', type=float, default=0.0)

    # ----- progenitor processing
    # wma options, interp factor, use_drad + drad (optional)
    parser.add_argument('--wma-all', action = 'store_true', default=False)
    parser.add_argument('--wma-vel', action = 'store_true', default=False)
    parser.add_argument('--wma-n', type=int, default=100)
    parser.add_argument('--wma-exp', type=int, default=3)
    parser.add_argument('--factor-interp', type=int, default=4)
    parser.add_argument('--use-drad-interp', action = 'store_true', default=False)
    parser.add_argument('--drad-interp', type=float, default=1e7)

    # ----- adm solver options
    # -- path (post progenitor), eos path, eos type
    parser.add_argument('--adm-problem', type=str, default='stellartable')
    parser.add_argument('--eos-path', type=str)
    parser.add_argument('--eos-type', type=str, default='stellarcollapse')

    # -- grid construction methods
    parser.add_argument('--use-radcut', action = 'store_true', default=False)
    parser.add_argument('--radcut', type=float, default=1.0e9)

    parser.add_argument('--use-rhocut', action = 'store_true', default=False)
    parser.add_argument('--rhocut', type=float, default=2.0e3)

    parser.add_argument('--use-custom', action = 'store_true', default=False)
    parser.add_argument('--custom-min', type=float, default=5e5)
    parser.add_argument('--custom-max', type=float, default=5e9)

    # -- grid and interpolation settings
    parser.add_argument('--zones', type=int, default=10000)
    parser.add_argument('--interp-method', type=str, default='cubic')
    parser.add_argument('--bc-type', type=str, default='clamped')

    # -- adm solver settings
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--converge-criteria', type=float, default=1.0e-12)
    parser.add_argument('--extrapolate', action = 'store_true', default=False)

    # ----- saving files
    parser.add_argument('--save-path', type=str, default='')
    parser.add_argument('--save-summary', action = 'store_true', default=False)
    parser.add_argument('--save-unconverted', action = 'store_true', default=False)
    parser.add_argument('--save-input', action = 'store_true', default=False)

    # ----- FOR TESTING ONLY, DEPRECATED METHODS
    parser.add_argument('--depr', action = 'store_true', default=False)

def main( ):
    
    parser = argparse.ArgumentParser( prog='ccsne' )

    # loads in the parameter file
    parser.add_argument('-f', '--file', type=str, help='name of the input parameter file + path (if needed)', action = LoadFile)
    parser.add_argument('-v', '--verbose', action = 'store_true')

    # get our input file parameters
    get_params( parser ) # getting all of the possible parameters
    params = parser.parse_args() # parameters!

    # reading the intitial profile
    if params.model_type.lower() == 'mesa':
        if params.model_header > 0:
            prof_raw = get_MESA_profile( params.model_path, params.model_header, params.verbose)
        else:
            prof_raw = get_MESA_profile( params.model_path, verbose=params.verbose)
    
    elif params.model_type.lower() == 'kepler':
        if params.model_header > 0:
            prof_raw = get_KEPLER_profile( params.model_path, params.model_header, params.verbose)
        else:
            prof_raw = get_KEPLER_profile( params.model_path, verbose=params.verbose)
    
    
    elif params.model_type.lower() == 'gr1d':
        if params.atbounce:
            prof_raw = get_GR1D_profile( params.model_path, params.atbounce, verbose=params.verbose)
    
        else:
            prof_raw = get_GR1D_profile( params.model_path, False, params.timestamp, verbose=params.verbose)


    # default
    if not params.depr:
        # interp from lagrangian to eulerian resolution (higher!)
        if params.use_drad_interp:
            prof_raw = interp_to_eulerian( prof_raw, use_drad = True, drad=params.drad_interp )
        else:
            prof_raw = interp_to_eulerian( prof_raw, factor = params.factor_interp, use_drad=False)
    
    # if we're testing against deprecated methods only
    else:
        if param.verbose: print('>>> WARNING: using deprecated methods of subsampling, for testing only!!')
        rad_depr = prof_raw[:, 0]
        prof_raw[:, 0], factor = subsample_depr( prof_raw[:, i], rad_depr, params.verbose, get_factor = True)
        for i in range(1, prof_raw.shape[1]):
            prof_raw[:, i] = subsample_depr( prof_raw[:, i], rad_depr, factor)

    # weighted moving average
    if params.wma_all:
        prof_raw[:, 1] = wmavg( prof_raw[:, 1], params.wma_n, params.wma_exp)
        prof_raw[:, 2] = wmavg( prof_raw[:, 2], params.wma_n, params.wma_exp)
        prof_raw[:, 3] = wmavg( prof_raw[:, 3], params.wma_n, params.wma_exp )
        prof_raw[:, 4] = wmavg( prof_raw[:, 4], params.wma_n, params.wma_exp)
        prof_raw[:, 5] = wmavg( prof_raw[:, 5], params.wma_n, params.wma_exp)
        prof_raw[:, 6] = wmavg( prof_raw[:, 6], params.wma_n, params.wma_exp)
        prof_raw[:, 7] = wmavg( prof_raw[:, 7], params.wma_n, params.wma_exp)
        prof_raw[:, 8] = wmavg( prof_raw[:, 8], params.wma_n, params.wma_exp)
        prof_raw[:, 9] = wmavg( prof_raw[:, 9], params.wma_n, params.wma_exp)

    elif params.wma_vel:
        prof_raw[:, 1] = wmavg( prof_raw[:, 1], params.wma_n, params.wma_exp)

    # make sure our save directories exists...
    if not os.path.exists( params.save_path ):
        os.makedirs( params.save_path )

    # save raw, pre ADM profile (progenitors.py)
    save_raw_profile( prof_raw, params.model_name, params.model_type, params.timestamp, params.save_path, params.verbose )
    raw_prof_name = f'{params.model_name.lower()}_{params.model_type.lower()}.prof'

    # initialize ADM solver, solve for new profile
    adm = ADMSolver( params.adm_problem, os.path.join(params.save_path, raw_prof_name), params.eos_path, params.eos_type, params.use_rhocut, params.rhocut, params.use_radcut, params.radcut, params.use_custom, params.custom_min, params.custom_max, params.zones, params.interp_method, params.bc_type, params.iterations, params.converge_criteria, params.extrapolate, params.verbose)
    adm_prof = adm.get_final_profile()

    # save phoebus profiles (progenitors.py)
    # optional: make summary file
    save_ADM_profile( adm_prof, params.model_name, params.model_type, params.eos_path, params.eos_type, params.timestamp, params.save_path, params.save_unconverted, params.save_summary, params.verbose)

    # saving input file in same output directory for later (optional)
    if params.save_input:
        shutil.copy( params.file, os.path.join( params.save_path, f'{params.model_name}_{params.model_type.lower()}.in'))

# ----- run!!
if __name__ == "__main__":
    main()