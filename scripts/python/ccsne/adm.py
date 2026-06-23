from seos import *

import numpy as np

# for iteration and numerical solutions
from scipy import optimize
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

    def __init__(self, problem: str, DATPATH: str, EOSPATH: str, use_rho_cut = True, rho_cut = 2.0e3, 
                 use_rad_cut = False, rad_cut = 1e9, use_custom_grid = False, custom_grid = np.zeros(1), 
                 zones=10000, interp_method = 'cubic', bc_type = 'clamped', num_iterations = 100, 
                 converge_criteria = 1.0e-12, verbose = True):
        self.problem    = problem
        self.DATPATH    = DATPATH
        self.EOSPATH    = EOSPATH
        self.n          = num_iterations
        self.method     = interp_method
        self.bounds     = bc_type
        self.zones      = zones
        self.conv_crit  = converge_criteria
        self.verbose    = verbose

        self.get_grid_data( use_rho_cut, rho_cut, use_rad_cut, rad_cut, use_custom_grid, custom_grid )

    def interp( self, y: np.ndarray ):

        if self.method == 'linear':
            return linear(self.r0, y, fill_value = 'extrapolate')
        elif self.method == 'cubic':
            return cubic(self.r0, y, bc_type=self.bounds) # default bc_type is 'not a knot'
        elif self.method == 'akima':
            return akima(self.r0, y)
        elif self.method == 'makima': # modified akima
            return akima(self.r0, y, method='makima')


    def get_grid_data( self, use_rho_cut: bool, rho_cut: float, use_rad_cut: bool, rad_cut: float, 
                        use_custom_grid: bool, custom_grid: np.ndarray ):
        
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

        # now we make the new phoebus grid, depending on the given criteria
        # all grids (except custom) are uniform and do NOT go to zero (for now), extrapolation occurs after ADM calculations
        if use_rho_cut:
            irho = np.abs(self.rho0 - rho_cut).argmin() # finding index of nearest density to rho_cut
            self.grid = np.linspace(1e2, self.r0[irho], self.zones)
        elif use_rad_cut:
            self.grid = np.linspace(1e2, rad_cut, self.zones)
        elif use_custom_grid:
            self.grid = custom_grid
        
        # finding energy density (consistent with eos) since we don't include that in profiles:
        if self.problem == "stellartable":
         
            if self.eosfilename.lower() == 'helmholtz':
                eps, u = calculate_eos_energy_helmholtz( self.rho0, temp, abar, zbar )
            else:
                eps, u = calculate_eos_energy_stellarcollapse( self.rho0, temp, ye, self.eosfilename )
            
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


    def calculate_newtonian_metric( self ):
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


    def calculate_initial_ADM( self ):
        
        # TODO: also find a reference for this method, and iteration...

        # 3-metric??
        gamma2 = 1.0 / (
            1.0 - (self.v_int(self.grid) ** 2.0 + (self.v_ang_int(self.grid) * self.grid) ** 2.0) / c ** 2.0)

        if self.problem == "tov":
            return -1, -1, -1 # we can come back to this case if needed...

        alpha2, a2 = self.calculate_newtonian_metric()

        rho_adm = alpha2 * self.rho_int(self.grid) * gamma2

        P_adm = (
            np.sqrt(alpha2)
            * gamma2
            * (self.rho_int(self.grid) + self.p_int(self.grid) / c ** 2.0)
            * self.v_int(self.grid)
        )  ## r-component

        S_adm = (alpha2 * gamma2 - 1.0) * self.rho_int(self.grid) * c ** 2.0 + (
            alpha2 * gamma2 + 2.0
        ) * self.p_int(self.grid)

        # interpolation for our lapse function
        self.alpha2_int = self.interp( alpha2 )
        self.a2_int = self.interp( a2 )
        
        return rho_adm, P_adm, S_adm



    def calculate_metric( self, rho_adm: np.ndarray, P_adm: np.ndarray, S_adm: np.ndarray ):
        
        # interpolated quantities
        rho_adm_int     = self.interp( rho_adm )
        j_adm_int       = self.interp( P_adm )

        r = self.grid

        # integrands for initial value problem solver
        def f(x, V):

            da = (V[0]
                / 8.0
                / x
                * (
                    32.0 * np.pi * x ** 2.0 * V[0] ** 2 * rho_adm_int(x) * G / c ** 2.0
                    + 3.0 * x ** 2.0 * V[0] ** 2.0 * V[1] ** 2.0
                    + 4.0
                    - 4.0 * V[0] ** 2.0
                ))
            
            dK = 8.0 * np.pi * V[0] ** 2.0 * j_adm_int(x) * G / c ** 3.0 - 3.0 / x * V[1]
        
            return [da, dK]
        
        ## INITIAL CONDITIONS (r = 0)
        V0 = [1, 0]
        
        ## SOLVE COUPLED DIFFERENTIAL EQUATIONS FOR EXTRINSIC CURVATURE

        sol = solve_ivp(f, [min(r), max(r)], V0, t_eval=r, vectorized=True)
        result = sol.y
        a = result[0]
        K = result[1]

        print(rho_adm.size, r.size)

        ################# Solve for lapse
        ## INITIALIZE MATRIX
        da = (
            a
            / 8.0
            / r
            * (
                32.0 * np.pi * r ** 2.0 * a ** 2.0 * rho_adm * G / c ** 2.0
                + 3.0 * r ** 2 * a ** 2 * K ** 2
                + 4
                - 4.0 * a ** 2
            )
        )
        ## BOUNDARY CONDITIONS

        M = np.zeros([len(r), len(r)])
        V = np.zeros(len(r))  # <--------  MX+V=0
        eps = r[1] - r[0]
        A0 = (
            1.0 / a[0] ** 2 / eps ** 2
            + da[0] / (2 * a[0] ** 3.0 * eps)
            - 1.0 / (a[0] ** 2.0 * r[0] * eps)
        )
        B0 = -(
            2.0 / (a[0] ** 2.0 * eps ** 2)
            + 4 * np.pi * G / c ** 2.0 * (S_adm[0] / c ** 2.0 + rho_adm[0])
            + 3.0 / 2.0 * K[0] ** 2.0
        )
        C0 = (
            1.0 / (a[0] ** 2.0 * eps ** 2.0)
            - da[0] / (2.0 * a[0] ** 3.0 * eps)
            + 1.0 / (a[0] ** 2.0 * r[0] * eps)
        )
        ##V[0]=A0
        M[0, 0] = A0 + B0
        M[0, 1] = C0
        AL = (
            1 / a[len(r) - 1] ** 2 / eps ** 2
            + da[len(r) - 1] / (2 * a[len(r) - 1] ** 3.0 * eps)
            - 1.0 / (a[len(r) - 1] ** 2.0 * r[len(r) - 1] * eps)
        )
        BL = -(
            2.0 / (a[len(r) - 1] ** 2.0 * eps ** 2)
            + 4
            * np.pi
            * G
            / c ** 2.0
            * (S_adm[len(r) - 1] / c ** 2 + rho_adm[len(r) - 1])
            + 3.0 / 2.0 * K[len(r) - 1] ** 2.0
        )
        CL = (
            1.0 / (a[len(r) - 1] ** 2.0 * eps ** 2.0)
            - da[len(r) - 1] / (2.0 * a[len(r) - 1] ** 3.0 * eps)
            + 1.0 / (a[len(r) - 1] ** 2.0 * r[len(r) - 1] * eps)
        )
        V[len(r) - 1] = eps / r[len(r) - 1] * CL
        M[len(r) - 1, len(r) - 2] = AL
        M[len(r) - 1, len(r) - 1] = BL + (1.0 - eps / r[len(r) - 1]) * CL

        ## THE REST OF THE MATRIX
        for i in range(1, len(r) - 1):
            A = (
                1 / a[i] ** 2 / eps ** 2
                + da[i] / (2 * a[i] ** 3.0 * eps)
                - 1.0 / (a[i] ** 2.0 * r[i] * eps)
            )
            B = -(
                2.0 / (a[i] ** 2.0 * eps ** 2)
                + 4 * np.pi * G / c ** 2.0 * (S_adm[i] / c ** 2.0 + rho_adm[i])
                + 3.0 / 2.0 * K[i] ** 2.0
            )
            C = (
                1.0 / (a[i] ** 2.0 * eps ** 2.0)
                - da[i] / (2.0 * a[i] ** 3.0 * eps)
                + 1.0 / (a[i] ** 2.0 * r[i] * eps)
            )
            M[i, i - 1] = A
            M[i, i] = B
            M[i, i + 1] = C

        def f(x):
            return np.dot(M, x) + V

        sol = optimize.root(f, 100 * np.ones(len(r)))
        alpha = sol.x

        ################# Solve for shift
        beta = -a ** 2.0 / 2.0 * alpha * r * K
        return a, K, alpha, beta


    def calculate_ADM( self, a, alpha, beta ):

        r = self.grid

        rho_adm = np.zeros(len(r))
        P_adm = np.zeros(len(r))
        S_adm = np.zeros(len(r))
        Srr_adm = np.zeros(len(r))

        for i in range(len(r)):
            # upper metric
            g00 = -1.0 / alpha[i] ** 2.0
            g0r = beta[i] / alpha[i] ** 2.0
            gamma2 = 1.0 / (1 - self.v_int(r[i]) ** 2.0 / c ** 2.0)
            rho_adm[i] = alpha[i] ** 2.0 * (
                (self.rho_int(r[i]) + self.p_int(r[i]) / c ** 2.0) * gamma2
                - self.p_int(r[i]) / c ** 2.0 * g00
            )
            P_adm[i] = alpha[i] * beta[i] * (
                (self.rho_int(r[i]) + self.p_int(r[i]) / c ** 2.0) * gamma2
                + self.p_int(r[i]) / c ** 2.0 * g00
            ) + alpha[i] * (
                (self.rho_int(r[i]) + self.p_int(r[i]) / c ** 2.0)
                * gamma2
                * self.v_int(r[i])
                + self.p_int(r[i]) / c ** 2.0 * g0r
            )  ## r-component  (upper index)
            S_adm[i] = (alpha[i] ** 2.0 * gamma2 - 1.0) * self.rho_int(
                r[i]
            ) * c ** 2.0 + (alpha[i] ** 2.0 * gamma2 + 2.0) * self.p_int(r[i])
            Srr_adm[i] = self.rho_int(r[i]) * self.v_int(r[i]) ** 2.0 * gamma2 * a[
                i
            ] ** 4.0 + self.p_int(r[i]) * a[i] ** 4.0 * (
                1.0 / a[i] ** 2
                - beta[i] ** 2.0 / alpha[i] ** 2.0
                + self.v_int(r[i]) ** 2.0 / c ** 2.0 * gamma2
            )
        return rho_adm, P_adm, S_adm, Srr_adm


    def iterate( self ):
        
        ### INITIAL ADM QUANTITIES
        rho_adm, P_adm, S_adm = self.calculate_initial_ADM()
        r = self.grid
        alpha_prev = np.sqrt(self.alpha2_int(r))

        for i in range( self.n ):

            a, K, alpha, beta = self.calculate_metric(r, rho_adm, P_adm, S_adm)

            if self.verbose: # TODO: add a more informative print statement here
                print(f'{i: 02d} {np.max(abs(alpha)): 04.5e} {np.max(abs(alpha_prev- alpha)): 04.5e}')

            # default criteria is 1e-12, loosening to 6e-12 works as well.
            if np.max(abs(alpha_prev - alpha)) < self.conv_crit:
                break

            if i == (self.n - 1):
                raise ArithmeticError(f'ADM calculation has not converged after {self.n} iterations.')
            alpha_prev = alpha
            rho_adm, P_adm, S_adm, Srr_adm = self.calculate_ADM(r, a, K, alpha, beta)

        return rho_adm, P_adm, S_adm, Srr_adm, a, K, alpha

    def extrapolate_data( self ):

        # all final output grids will be uniform once extrapolated to zero.
        r = np.linspace(0, self.grid[-1], self.zones)

        rho_adm0, P_adm0, S_adm0, Srr_adm0, a0, K0, alpha0 = self.iterate()

        # we need interpolators for these (using the input grid for creation, then extrapolating)
        self.rho_adm = self.interp( rho_adm0 )(r, extrapolate=True)
        self.P_adm = self.interp( P_adm0 )(r, extrapolate=True)
        self.S_adm = self.interp( S_adm0 )(r, extrapolate=True)
        self.Srr_adm = self.interp( Srr_adm0 )(r, extrapolate=True)
        
        # TODO: determine if we want to save the metric or not (e.g. do we need these?)
        a = self.interp( a0 )(r, extrapolate=True)
        K = self.interp( K0 )(r, extrapolate=True)
        alpha = self.interp(  alpha0)(r, extrapolate=True)

        self.grid0 = r # new grid for r -> 0!
        self.rho   = self.rho_m_int(r, extrapolate=True)
        # other primitives from profile
        self.vel    = self.v_int(r, extrapolate=True)
        self.press  = self.p_int(r, extrapolate=True)
        self.ye     = self.ye_int(r, extrapolate=True)
        self.eps    = self.eps_int(r, extrapolate=True)  
        self.temp   = self.temp_int(r, extrapolate=True)


    def get_final_profile( self ):

        self.extrapolate_data() # this actually does the full pipeline!

        phb_profile = np.columnstack([
            self.grid0,
            self.rho,
            self.temp,
            self.ye,
            self.eps,
            self.vel,
            self.press,
            self.rho_adm,
            self.P_adm,
            self.S_adm,
            self.Srr_adm
        ])

        return phb_profile