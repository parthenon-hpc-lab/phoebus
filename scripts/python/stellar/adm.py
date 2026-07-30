
#  A refactor of Mariam's GR_Solver class, with some of the original methods and a cleaner i/o structure.
# 
#     >>> outstanding tasks
# 
#         TODO: update class documentation
#         TODO: update method documentation
#         TODO: ask brandon + devs about TOV and homologous cases, could reproduce if needed?
# 
#     law. 16 jun 2026


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
    Solves for the necessary GR quantities needed to initialize monopole GR in ``phoebus`` using the ADM (Arnowitt-Deser-Misner) formalism.

    For the full details of the methodologies used, refer to 

    - ch. 5.2 (Gravity) of Mariam Gogilashvili's thesis (Gogilashvili 2024, FSU); **recommended** as this contains full derivations,
    - `Introduction to 3+1 Numerical Relativity <https://academic.oup.com/book/9640>`_ for a detailed reference on 3 + 1 formalisms,
    - the ``phoebus`` instrument paper; `Barker et al. 2024 <10.48550/arXiv.2410.09146>`_, and   
    - the header file for monopole GR in the main ``phoebus`` codebase; ``src/monopole_gr/monopole_gr.hpp``.

    A refactor of Mariam Gogilashvili's original GR_Solver class.

    '''

    def __init__(self, problem: str, DATPATH: str, EOSPATH: str, eos_type: str = 'stellarcollapse', use_def_rad: bool = False, use_rho_cut: bool = False, rho_cut: float = 2.0e3, use_rad_cut: bool = False, rad_cut: float = 1e9, use_custom: bool = False, custom_min: float = 5e5, custom_max: float = 5e9, zones: int = 10000, interp_method: str = 'cubic', bc_type: str = 'clamped', interp_method_adm: str = 'piecewise', num_iterations: int = 100, dalpha_eps: float = 1.0e-12, extrapolate: bool = True, verbose: bool = True):
        
        r'''
        Constructor for the ADMSolver class.

        Args:
            problem (str): Which problem you want to generate ``phoebus`` input for. Options are 'StellarTable', 'TOV' (not implemented), and 'homologous' (not implemented).
            DATPATH (str): Desired directory to save file(s) in.
            EOSPATH (str): Path and filename pointing to the *.h5 tabulated equation of state.
            eos_type (str, optional): The type of tabulated EOS. Options are 'StellarCollapse' or 'Helmholtz'; default is 'StellarCollapse'
            use_def_rad (bool, optional): If enabled, uses the original radial grid of the input progenitor profile.
            use_rho_cut (bool, optional): If enabled, uses a density threshold (in g/cm^3) to truncate outer radial coordinate.
            rho_cut (float, optional): Density to make the radial truncation at, if use_rho_cut enabled.
            use_rad_cut (bool, optional): If enabled, uses a radial threshold (in cm) to truncate outer radial coordinate.
            rad_cut (float, optional): Radius to make the radial truncation at, if use_rad_cut enabled.
            use_custom (bool, optional): If enabled, allows user to input custom grid using radial bounds (in cm).
            custom_min (float, optional): Minimum radius of grid, if use_custom enabled.
            custom_max (float, optional): Maximum radius of grid, if use_custom enabled.
            zones (int, optional): Radial resolution of the desired post-ADM grid. Also number of zones used for ADM calculations.
            interp_method (str, optional): Interpolation method for primitive calculations/quantities. Options are 'linear', 'cubic', 'akima', 'makima'. 
            bc_type (str, optional): Type of boundary conditions, if using cubic interpolation.
            interp_method_adm (str, optional): Interpolation method for ADM calculations/quantities. Options are 'linear', 'piecewise'.
            num_iterations (int, optional): Number of iterations the solver will do when attempting to converge below some $d \alpha$ threshold.
            dalpha_eps (float, optional): Allowable variation of the lapse, $\alpha$, between solver iterations.
            extrapolate (bool, optional): If enabled, extrapolates the post-ADM profile to r = 0.
            verbose (bool, optional): Enables command line output. Defaults to True.

        '''

        self.problem    = problem.lower()
        self.DATPATH    = DATPATH
        self.EOSPATH    = EOSPATH
        self.eos_type   = eos_type.lower()
        self.n          = num_iterations
        self.method     = interp_method.lower()
        self.method_adm = interp_method_adm.lower()
        self.bounds     = bc_type.lower()
        self.zones      = zones
        self.zones0     = zones
        self.dalpha_eps = dalpha_eps
        self.do_extrap  = extrapolate
        self.verbose    = verbose

        self.problem    = problem.lower()
        self.DATPATH    = DATPATH
        self.EOSPATH    = EOSPATH
        self.eos_type   = eos_type.lower()
        self.n          = num_iterations
        self.method     = interp_method.lower()
        self.method_adm = interp_method_adm.lower()
        self.bounds     = bc_type.lower()
        self.zones      = zones
        self.zones0     = zones
        self.dalpha_eps = dalpha_eps
        self.do_extrap  = extrapolate
        self.verbose    = verbose

        self.get_grid_data( use_def_rad, use_rho_cut, rho_cut, use_rad_cut, rad_cut, use_custom, custom_min, custom_max)
        if self.verbose: print(f'>>> initializing ADM solver.')


    def interp( self, y: np.ndarray ):

        '''
        Allows for flexible selection of primitive interpolation method, based on `interp_method` at initialization.
        Interpolates based on the original radial grid of the progenitor profile. Allows for extrapolation with all methods.

        Args:
            y (np.ndarray): the data to be interpolated.

        '''

        if self.method == 'linear':
            return linear(self.r0, y, fill_value = 'extrapolate')
        elif self.method == 'cubic':
            return cubic(self.r0, y, bc_type=self.bounds, extrapolate=True) # default bc_type is 'not a knot'
        elif self.method == 'akima':
            return akima(self.r0, y, extrapolate=True)
        elif self.method == 'makima': # modified akima
            return akima(self.r0, y, method='makima', extrapolate=True)

    def interp_adm( self, y: np.ndarray ):

        '''
        Allows for flexible selection of ADM interpolation method, based on `interp_method_adm` at initialization.
        Interpolates based on the new, post-ADM grid. Allows for extrapolation with all methods.

        Args:
            y (np.ndarray): the data to be interpolated.

        '''
        
        if self.method_adm == 'linear': # todo: change this to an additional flag, not the same as above method
            return linear(self.grid, y, fill_value = 'extrapolate')

        elif self.method_adm == 'piecewise':
            return piecewise(self.grid, y)


    def get_grid_data( self, use_def_rad: bool=False, use_rho_cut: bool=True, rho_cut: float=2e3, use_rad_cut: bool=False, rad_cut: float= 5e9, use_custom: bool=False, custom_min: float=5.0e5, custom_max: float=5.0e9):
        
        '''
        Creates a new radial grid based on initialization parameters (`use_def_rad`, `use_*_cut`, `use_custom`) and generates
        interpolators for all primitive quantities. Also recalculates specific internal energy using ``Singularity-EOS`` to 
        be consistent with the equation of state that will be used in ``phoebus``.

        If no initialization parameters are input, `use_rho_cut` at a threshold density of 2.0e3 g/cm^3 is used to create the new grid.

        Args:
            use_def_rad (bool, optional): If enabled, uses the original radial grid of the input progenitor profile. Defaults to False.
            use_rho_cut (bool, optional): If enabled, uses a density threshold (in g/cm^3) to truncate outer radial coordinate. Defaults to True.
            rho_cut (float, optional): Density to make the radial truncation at, if use_rho_cut enabled. Default is 2.0e3 (g/cm^3).
            use_rad_cut (bool, optional): If enabled, uses a radial threshold (in cm) to truncate outer radial coordinate. Defaults to False.
            rad_cut (float, optional): Radius to make the radial truncation at, if use_rad_cut enabled. Default is 5.0e9 cm.
            use_custom (bool, optional): If enabled, allows user to input custom grid using radial bounds (in cm). Defaults to False.
            custom_min (float, optional): Minimum radius of grid, if use_custom enabled. Default is 5.0e5 cm.
            custom_max (float, optional): Maximum radius of grid, if use_custom enabled. Default is 5.0e9 cm.

        '''

        if self.problem == 'stellartable':
            prof = np.loadtxt(self.DATPATH)

        elif self.problem == 'tov':
            raise UserWarning('TOV problem not currently supported for initialization. Refer to `scripts/python/grsolver` for deprecated methods.')

        elif self.problem == 'homologous':
            prof = np.loadtxt(self.DATPATH)
            # ensuring we only use the default radius as input.
            self.use_rho_cut = False
            self.use_rad_cut = False
            self.use_custom = False
            self.use_def_rad = True
            # raise UserWarning('Homologous collapse problem not currently supported for initialization. Refer to `scripts/python/grsolver` for deprecated methods.')
        
        else:
            raise UserWarning(f'{self.problem} problem is not implemented in this pipeline.')
        

        # we need to preserve the original data from the profile to calculate lapse functions, etc.
        self.r0     = prof[:, 0]
        self.rho0   = prof[:, 2]
        # other primitives from profile
        vel         = prof[:, 1]
        press       = prof[:, 3]
        ye          = prof[:, 4]
        temp        = prof[:, 5]
        eps         = prof[:, 6]
        omega       = prof[:, 7]
        abar        = prof[:, 8]
        zbar        = prof[:, 9]

        # now we make the new phoebus grid, depending on the given criteria
        # all grids are uniform and do NOT go to zero (for now), extrapolation occurs after ADM calculations
        if use_rho_cut:
            irho = np.abs(self.rho0 - rho_cut).argmin() # finding index of nearest density to rho_cut
            self.grid = np.linspace(self.r0[0], self.r0[irho], self.zones)
        elif use_rad_cut:
            self.grid = np.linspace(self.r0[0], rad_cut, self.zones)
        elif use_custom:
            self.grid = np.linspace(custom_min, custom_max, self.zones)
        elif use_def_rad:
            self.grid = self.r0
            self.zones = self.r0.shape[0] 
        
        # finding energy density (consistent with eos) since we don't include that in profiles:
        if self.problem == "stellartable":
         
            if self.eos_type.lower() == 'helmholtz':
                eps, u = calculate_eos_energy_helmholtz( self.rho0, temp, abar, zbar, self.EOSPATH )
            elif self.eos_type.lower() == 'stellarcollapse':
                eps, u = calculate_eos_energy_stellarcollapse( self.rho0, temp, ye, self.EOSPATH )
            else:
                raise UserWarning(f'StellarCollapse style EOS {self.eos_type} not currently supported for CCSNe initialization.')
            
            self.rho = self.rho0 + u / c ** 2.0  # energy density

        elif self.problem == 'homologous':
            u = self.rho0 * eps
            self.rho = self.rho0 + u / c ** 2.0
        # the rest of our data will need interpolators to map to the new phoebus grid
        self.v_int      = self.interp( vel )
        self.rho_int    = self.interp( self.rho ) # energy density, GR naming convention
        self.p_int      = self.interp( press )
        self.rho_m_int  = self.interp( self.rho0 ) # mass density
        self.ye_int     = self.interp( ye )
        self.temp_int   = self.interp( temp )
        self.eps_int    = self.interp( eps )
        self.v_ang_int   = self.interp( omega )


    def calculate_newtonian_metric( self ):

        r'''
        Using a spherically symmetric metric, calculates the values of $\alpha^2$, $a^2$ in the weak-field (e.g. Newtonian) limit.

        Beginnning with our ADM metric in natural units (s.t. $c = 1$), s.t.
        
         $$ds^2 = (-\alpha^2 +\beta_i \beta^i) dt^2 + 2 \beta_i dt dx^i + \gamma_{ij} dx^i dx^j$$
        
         and taking a sph. symm. ansatz at $t=0$ (s.t. our shift $\beta = 0$), we write the metric as

        - $ds^2 = -\alpha^2 dt^2 + a^2 dr^2 + b^2 r^2 d \Omega^2$
        - $ds^2 = -e^{2 \varphi}dt^2 +  e^{-2 \varphi}dr^2 + r^2 d \Omega^2$ 

        In the weak-field limit (e.g. when gravitational potential $\varphi \rightarrow 0$), we can take a small-x Taylor expansion 
        to solve for $a$, $\alpha$, s.t.

        - $\alpha^2 = e^{2 \varphi} \approx 1 + 2 \varphi$, and
        - $a^2 = e^{-2 \varphi} \approx 1 + 2r \partial \varphi$,

        recalling that $\varphi = -r \partial \varphi$ and given a symm. spherical gravitational potential, s.t.

        $$\varphi = - \frac{2 \pi G}{c^2} \rho r^2$$

        Returns:
            tuple:

            - alpha2 (np.ndarray): Lapse function ($\alpha$) squared, weak-field approximation.
            - a2 (np.ndarray): Metric component ($a$) squared, weak-field approximation.

        '''

        # phi is the spherically symm. gravitational potential (for monopole assumption)
        phi = np.zeros(self.zones)
        dphi = np.zeros(self.zones)

        for i in range(self.zones):
            I = (
                -2.0
                * np.pi
                * G
                * self.rho
                * self.r0
                / self.grid[i]
                * (self.grid[i] + self.r0 - abs(self.grid[i] - self.r0))
            )

            try:
                phi[i] = np.trapezoid(I, x=self.r0)
            except:
                phi[i] = np.trapz(I, x=self.r0)

        dphi = np.gradient(phi, self.grid)

        # we find alpha^2, a^2 using the metric in the sph. symm., weak field (newtonian) limit and the metric in eq. 15 of Barker et al. 2024
        alpha2 = 1.0 + 2.0 * phi / c ** 2.0
        a2 = 1.0 + 2.0 * self.grid * dphi / c ** 2.0

        return alpha2, a2


    def calculate_initial_ADM( self ):

        r'''
        Calculates initial values for ADM density, momentum, and stress tensor given the Newtonian values of $\alpha^2$, $a^2$.

        Given our 3-metric , s.t.

        $$\gamma^2 = (1 - (v^2 + \omega^2 r^2))^{-1}$$

        We find inital values for ADM quantities as given by

        - $\rho^{(ADM)} = \rho \alpha^2 \gamma^2 $
        - $P^{(ADM)} = \alpha \gamma^2 (\rho P)^2 \omega$
        - $S^{(ADM)} = (\alpha^2 \gamma^2 - 1) \rho^2 + (\alpha^2 \gamma^2 + 2)P$

        Also ref. Eqs. 5.10-12 in Gogilashvili 2024 for further relation to $T^{\mu \nu}$.

        Returns:
            tuple:
    
            - rho_adm (np.ndarray): Initial ADM density, on the (new) radial grid.
            - P_adm (np.ndarray): Initial ADM momentum, on the (new) radial grid.
            - S_adm (np.ndarray): Initial ADM stress tensor, on the (new) radial grid.
        '''

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
        self.alpha2_int = self.interp_adm( alpha2 )
        self.a2_int = self.interp_adm( a2 )
        
        return rho_adm, P_adm, S_adm



    def calculate_metric( self, rho_adm: np.ndarray, P_adm: np.ndarray, S_adm: np.ndarray ):

        r'''
        Calculates the components $a$ (metric) and $K_r^r$ (extrinsic curvature) using the Einstein constraint equations 
        with a Hamiltonian constraint (Barker et al. 2024, eqs. 16-17) and the boundary conditions as detailed in eqs. 23-28.
        
        Args:
            rho_adm (np.ndarray): ADM density, on the (new) radial grid.
            P_adm (np.ndarray): ADM momentum, on the (new) radial grid.
            S_adm (np.ndarray): ADM stress tensor, on the (new) radial grid.

        Returns:
            tuple:
    
            - a (np.ndarray): Metric component ($a$).
            - K (np.ndarray): Extrinsic curvature ($K_r^r$).
            - alpha (np.ndarray): Lapse function ($\alpha$).
            - beta (np.ndarray): Shift function ($\beta$).
        '''

        # interpolated quantities
        rho_adm_int     = self.interp_adm( rho_adm )
        j_adm_int       = self.interp_adm( P_adm )

        r = self.grid

        # integrands for initial value problem solver, eq. 16-17 in Barker et al. 2024
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

        sol = solve_ivp(f, [r[0], r[-1]], V0, t_eval=r, vectorized=True)
        result = sol.y
        
        # break if original data is too sparse and our ivp dimensions are incorrect...
        if (result.shape[1] != self.zones):
            raise ValueError( f'initial raw profile at `{self.DATPATH}` too radially sparse for interpolation. increase the factor or decrease drad spacing and try again.')

        a = result[0]
        K = result[1]

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

        M = np.zeros([self.zones, self.zones])
        V = np.zeros(self.zones)  # <--------  MX+V=0
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
            1 / a[self.zones - 1] ** 2 / eps ** 2
            + da[self.zones - 1] / (2 * a[self.zones - 1] ** 3.0 * eps)
            - 1.0 / (a[self.zones - 1] ** 2.0 * r[self.zones - 1] * eps)
        )
        BL = -(
            2.0 / (a[self.zones - 1] ** 2.0 * eps ** 2)
            + 4
            * np.pi
            * G
            / c ** 2.0
            * (S_adm[self.zones - 1] / c ** 2 + rho_adm[self.zones - 1])
            + 3.0 / 2.0 * K[self.zones - 1] ** 2.0
        )
        CL = (
            1.0 / (a[self.zones - 1] ** 2.0 * eps ** 2.0)
            - da[self.zones - 1] / (2.0 * a[self.zones - 1] ** 3.0 * eps)
            + 1.0 / (a[self.zones - 1] ** 2.0 * r[self.zones - 1] * eps)
        )
        V[self.zones - 1] = eps / r[self.zones - 1] * CL
        M[self.zones - 1, self.zones - 2] = AL
        M[self.zones - 1, self.zones - 1] = BL + (1.0 - eps / r[self.zones - 1]) * CL

        ## THE REST OF THE MATRIX
        for i in range(1, self.zones - 1):
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

        # TODO: by recommendation of brandon, consider implementing a 
        #       dedicated tridiagonal matrix solver in the near future?
        def f(x):
            return np.dot(M, x) + V

        sol = optimize.root(f, 100 * np.ones(self.zones))
        alpha = sol.x

        ################# Solve for shift
        beta = -a ** 2.0 / 2.0 * alpha * r * K
        return a, K, alpha, beta


    def calculate_ADM( self, a, alpha, beta ):

        r'''
        Calculates spherically symmetric, time-dependent ADM density, momentum, and stress tensor using full matrix equations and 4-metric $g_{\mu \nu}$.
        Refer to Appendix A of Barker et al. 2024, along with ch. 5.2.6 of Gogilashvili 2024 for full derivations.
        
        Args:
            a (np.ndarray): Metric component ($a$).
            alpha (np.ndarray): Lapse function ($\alpha$).
            beta (np.ndarray): Shift function ($\beta$).

        Returns:
            tuple:
            
            - rho_adm (np.ndarray): ADM density.
            - P_adm (np.ndarray): ADM momentum.
            - S_adm (np.ndarray): ADM stress tensor, trace.
            - Srr_adm (np.ndarray): ADM stress tensor.
    
        '''

        r = self.grid

        rho_adm = np.zeros(self.zones)
        P_adm = np.zeros(self.zones)
        S_adm = np.zeros(self.zones)
        Srr_adm = np.zeros(self.zones)

        for i in range(self.zones):
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

        r'''
        Iterates through the full ADM solver pipeline until we find that $d \alpha < \epsilon$ after $n$ iterations, 
        where $\epsilon$ is the threshold value set at initialization by ``dalpha_eps`` and $n \less N$, where $N$ is the 
        maximum number of iterations as set by the user at initialization.
        
        Returns:
            tuple:
            
            - rho_adm (np.ndarray): Final ADM density.
            - P_adm (np.ndarray): Final ADM momentum.
            - S_adm (np.ndarray): Final ADM stress tensor, trace.
            - Srr_adm (np.ndarray): Final ADM stress tensor.
            - a (np.ndarray): Final metric component ($a$).
            - K (np.ndarray): Final extrinsic curvature ($K_r^r$).
            - alpha (np.ndarray): Final lapse function ($\alpha$).
        
        '''
        
        ### INITIAL ADM QUANTITIES
        rho_adm, P_adm, S_adm = self.calculate_initial_ADM()
        r = self.grid
        alpha_prev = np.sqrt(self.alpha2_int(r))

        # doing an initial caluclation of all returned quantities (e.g. if convergence criteria is met on the zeroth iteration)
        a, K, alpha, beta = self.calculate_metric(rho_adm, P_adm, S_adm)
        _, _, _, Srr_adm = self.calculate_ADM(a, alpha, beta)

        for i in range( self.n ):

            a, K, alpha, beta = self.calculate_metric(rho_adm, P_adm, S_adm)

            if self.verbose: # TODO: add a more informative print statement here
                print( f'\titeration {i:2d}\talpha = {np.max(abs(alpha)):4.5e}\tdalpha = {np.max(abs(alpha_prev - alpha)):4.5e}')

            # default criteria is 1e-12, loosening to 6e-12 is robust as well.
            if np.max(abs(alpha_prev - alpha)) < self.dalpha_eps:
                break

            if i == (self.n - 1):
                raise ArithmeticError(f'ADM calculation has not converged after {self.n} iterations.')
            
            alpha_prev = alpha
            rho_adm, P_adm, S_adm, Srr_adm = self.calculate_ADM(a, alpha, beta)

        return rho_adm, P_adm, S_adm, Srr_adm, a, K, alpha

    # TODO: figure out if we actually need to interpolate to zero here or leave the inner boundary as is
    # i.e. reference FLASH/BANG??
    def calculate_all( self ):
        r'''
        Handles all iteration, solving, and final interpolation/extrapolation to the user-input radial grid.
        '''

        if self.verbose: print(f'>>> iterating to solve for final ADM quantities.')
        rho_adm0, P_adm0, S_adm0, Srr_adm0, a0, K0, alpha0 = self.iterate()

        # if we extrapolate to zero...
        if self.do_extrap:
            if self.verbose: print(f'>>> extrapolating to r = 0.')
            r = np.linspace(0, self.grid[-1], self.zones0)
        else:
            r = self.grid

        # we need interpolators for these (using the input grid for creation, then extrapolating)
        self.rho_adm = self.interp_adm( rho_adm0 )(r)
        self.P_adm = self.interp_adm( P_adm0 )(r)
        self.S_adm = self.interp_adm( S_adm0 )(r)
        self.Srr_adm = self.interp_adm( Srr_adm0 )(r)
        
        # TODO: determine if we want to save the metric or not (e.g. do we need these?)
        # a = self.interp_adm( a0 )(r)
        # K = self.interp_adm( K0 )(r)
        # alpha = self.interp_adm( alpha0 )(r)

        self.grid0 = r # new grid for r -> 0!
        self.rho   = self.rho_m_int(r)
        # other primitives from profile
        self.vel    = self.v_int(r)
        self.press  = self.p_int(r)
        self.ye     = self.ye_int(r)
        self.eps    = self.eps_int(r)  
        self.temp   = self.temp_int(r)


    def get_final_profile( self ):

        '''
        Serves as the driver for ADMSolver.

        Returns:
            np.ndarray: Post-ADM, Eulerian stellar/ccsne profile (from MESA, KEPLER, GR1D...). 
            Assumed to be in following column order: [radius, density, temperature, ye, sie, velocity, pressure, density (adm), momentum (adm), $S$ (adm), $S^r_r$ (adm)]

        '''

        self.calculate_all() # this actually does the full pipeline!

        phb_profile = np.column_stack([
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


class piecewise:

    '''
        small python implementation of Jonah's quadradtic piecewise interpolation method that's in the main codebase;
        see monopole_gr/monopole_gr_base.hpp for a full reference and the original source code.

        :meta private:

    '''

    def __init__(self, x0: np.ndarray, y):
        self.x0 = x0
        self.y = y

    def __call__(self, x):

        # rgrid/radius/x0 is existing grid, r/x is what we're interpolating to 

        n = self.x0.size
        x = np.asarray(x)
        # case for if a single float is passed in
        if x.shape == ():
            x = np.asarray([x])

        dx = np.abs(self.x0[1] - self.x0[0]) # spacing, we asssume that the original grid is uniform.
        c0 = np.zeros(x.size)
        c1 = np.zeros(x.size)
        c2 = np.zeros(x.size)
        xoffset = np.zeros(x.size)

        for i in range(x.size):

            ix = (np.abs( self.x0 - x[i] )).argmin() # closest index on the original grid
            flr = ix * dx + np.min(self.x0)

            c0[i] = self.y[ix]

            if (ix == 0) :
                c1[i] = -(3 * self.y[ix] - 4. * self.y[ix + 1] + self.y[ix + 2]) / (2 * dx)
                c2[i] = (self.y[ix] - 2 * self.y[ix + 1] + self.y[ix + 2]) / (2 * dx * dx)
            elif (ix == n - 1):
                c1[i] = (3 * self.y[ix - 2] - 4 * self.y[ix - 1] + 3 * self.y[ix]) / (2 * dx)
                c2[i] = (self.y[ix - 2] - 2 * self.y[ix - 1] + self.y[ix]) / (2 * dx * dx)
            else:
                c1[i] = (self.y[ix + 1] - self.y[ix - 1]) / (2 * dx)
                c2[i] = (self.y[ix + 1] - 2 * self.y[ix] + self.y[ix - 1]) / (2 * dx * dx)
    
            xoffset[i] = x[i] - flr

        return c0 + (c1 * xoffset) + (c2 * xoffset * xoffset)
