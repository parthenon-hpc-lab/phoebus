import progenitors
import seos

import numpy as np

from scipy.integrate import solve_ivp
from scipy.interpolate import Akima1DInterpolator as akima
from scipy.interpolate import CubicSpline as cubic

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

    def __init__(self, problem: str, DATPATH: str, EOSPATH: str, num_iterations = 1000):
        self.problem    = problem
        self.DATPATH    = DATPATH
        self.EOSPATH    = EOSPATH
        self.n          = num_iterations

        self.loadData()

    def loadData(self):
        
        if self.problem == 'stellarcollapse':
            prof = np.loadtxt(self.DATPATH)


        elif self.problem == 'tov':
            # come back to this if needed
            pass

        elif self.problem == 'homologous':
            # come back to this if needed
            pass

        self.r0     = prof[:, 0]
        self.rho    = prof[:, 1]
        # TODO: start here 


    def calculateNewtonianMetric( self ):
        pass


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