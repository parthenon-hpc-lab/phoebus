# consolidation of Mariam's homologous generation scripts, plus
# additional resources to create profiles that are either 
#     - consistent with a tabulated eos (using singularity-eos) instead of ideal gas, or
#     - made with the Yahil 1983 homologous collapse case in mind.


# law. 29 jul 26. 


import numpy as np
import math
import time
import matplotlib.pyplot as pl
import sys
import glob
from scipy.integrate import odeint
from astropy import constants as const

G = const.G.cgs.value
c = const.c.cgs.value
msun = const.M_sun.cgs.value

def gen_homologous_goldreich( ):
    pass

def gen_homologous_eos( ):
    pass

def gen_homologous_yahil( ):
    pass