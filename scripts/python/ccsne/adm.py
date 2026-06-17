import progenitors
import seos

import numpy as np

from scipy.integrate import solve_ivp

# interpolation schemes
from scipy.interpolate import interp1d as linear # deprecated, do not recommend
from scipy.interpolate import Akima1DInterpolator as akima
from scipy.interpolate import CubicSpline as cubic

# physical constants
from astropy import constants as const
G = const.G.cgs.value
c = const.c.cgs.value

class ADMSolver:

    '''

    a reworked version of Mariam's GR_Solver class, with some of the original methods
    and a cleaner i/o structure.

    >>> outstanding tasks

        TODO: update class documentation
        TODO: update method documentation
        TODO: ask brandon + devs about TOV and homologous cases, could reproduce if needed?
        TODO: find references for adm methodology that's being used
        TODO: does this need to be a class? might make the most sense to remain that way.
        TODO: clean up ADM iteration algorithm-- is there a cleaner/faster way to do this?
    
    law. 16 jun 2026

    '''

    def __init__(self, problem: str, grid: np.ndarray, DATPATH: str, EOSPATH: str, interp_method = 'cubic', bc_type = 'clamped', num_iterations = 1000):
        self.problem    = problem
        self.DATPATH    = DATPATH
        self.EOSPATH    = EOSPATH
        self.n          = num_iterations
        self.grid       = grid # desired output grid/domain
        self.method     = interp_method
        self.bounds    = bc_type

        self.loadData()

    def interp( self, y: np.ndarray ):

        if self.method == 'linear':
            return linear(self.r0, y, fill_value = 'extrapolate')
        elif self.method == 'cubic':
            return cubic(self.r0, y, bc_type=self.bounds) # default bc_type is 'not a knot'
        elif self.method == 'akima':
            return akima(self.r0, y)
        elif self.method == 'makima': # modified akima
            return akima(self.r0, y, method='makima')


    def loadData(self, interp_method='cubic'):
        
        if self.problem == 'stellarcollapse':
            prof = np.loadtxt(self.DATPATH)
        elif self.problem == 'tov':
            # come back to this if needed
            pass
        elif self.problem == 'homologous':
            # come back to this if needed
            pass

        # we need to preserve the original data from the profile to calculate lapse functions, etc.
        self.r0     = prof[:, 0]
        self.rho0   = prof[:, 2]
        # other primitives from profile
        vel         = prof[:, 1]
        press       = prof[:, 3]
        ye          = prof[:, 4]
        temp        = prof[:, 5]
        omega       = prof[:, 7]
        abar        = prof[:, 8]
        zbar        = prof[:, 9]
        
        # finding energy density (consistent with eos) since we don't include that in profiles:
        if self.problem == "stellartable":
         
            if self.eosfilename.lower() == 'helmholtz':
                eps, u = CalculateInternalEnergy_Helmholtz( self.rho0, temp, abar, zbar )
            else:
                eps, u = CalculateInternalEnergy_StellarCollapse( self.rho0, temp, ye, self.eosfilename )
            
            rho = self.rho0 + u / c ** 2.0  # energy density

        # the rest of our data will need interpolators to map to the new phoebus grid
        self.v_int      = self.interp( vel )
        self.rho_int    = self.interp( rho ) # energy density, GR naming convention
        self.p_int      = self.interp( press )
        self.rho_m_int  = self.interp( self.rho0 ) # mass density
        self.ye_int     = self.interp( ye )
        self.temp_int   = self.interp( temp )
        self.eps_int    = self.interp( eps )
        self.vang_int   = self.interp( omega )


    def calculateNewtonianMetric( self ):
        # TODO: still need to find a reference for this...

        phi = np.zeros(len(self.grid))
        dphi = np.zeros(len(self.grid))

        for i in range(len(self.grid)):
            I = (
                -2.0
                * np.pi
                * G
                * self.rho
                * self.r0
                / self.grid[i]
                * (self.grid[i] + self.r0 - abs(self.grid[i] - self.r0))
            )

            phi[i] = np.trapezoid(I, x=self.r0)
                
        dphi = np.gradient(phi, self.grid)
        alpha2 = 1.0 + 2.0 * phi / c ** 2.0
        a2 = 1.0 + 2.0 * self.grid * dphi / c ** 2.0
        
        return alpha2, a2


    def calculateInitialADM( self ):
        pass


    def calculateADM( self ):
        pass


    def iterate( self ):
        pass

    def extrapolateData( self ):
        pass

    def saveData( self, fout: str, zones = 10000 ):
        pass