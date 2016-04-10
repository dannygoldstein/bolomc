
__author__ = 'Danny Goldstein <dgold@berkeley.edu>'
__whatami__ = 'Burns+2014 Table 1 data interface.'
__all__ = ['get_mwebv','get_hostebv', 'get_hostrv_prior']

import os
import sncosmo
import numpy as np
from itertools import product
from scipy.stats import uniform
from distributions import TruncNorm

bdfname = os.path.join('/'.join(__file__.split('/')[:-1]), '../data/burns2014.tab')

##############################################################################
# See Burns+2014 #############################################################
##############################################################################

_allowed_bintypes = ('unbinned', 
                     'binned',
                     'gmm') # Gaussian mixture model

_allowed_dusttypes = (sncosmo.OD94Dust,
                      sncosmo.F99Dust)

_allowed = list(product(_allowed_dusttypes, _allowed_bintypes))

# Lookup table that stores the column numbers of dust / bintype
# combinations.

_index = {_allowed[i] : i + 6 for i in range(len(_allowed))}

##############################################################################
        
def get_hostebv(name):
    """Get the E(B-V)_HOST and uncertainty for SN `name` from Table 1 of 
    Burns+2014."""
    row = _search_row(name)
    token = row[5]
    left, right = token.split('(')
    right = right[:-1]
    ebmv = float(left)
    err = float(right)
    return ebmv, err

def get_mwebv(name):
    """Get the E(B-V)_MW and uncertainty for SN `name` from Table 1 of 
    Burns+2014."""
    row = _search_row(name)
    token = row[4]
    left, right = token.split('(')
    right = right[:-1]
    ebmv = float(left)
    err = float(right)
    return ebmv, err

def get_hostrv_prior(name, rv_bintype, dust_type):
    row = _search_row(name)
    token = row[_index[(dust_type, rv_bintype)]]
    
    if token.endswith('ts'):
        raise KeyError('No Rv constraints for %s %s %s'.format(
            name, rv_bintype, dust_type))
    elif token.startswith('<'):
        lim = float(token[1:])
        dist = uniform(0., lim)
    else:
        mean, lims = token.split('^')
        mean = float(mean)
        low, high = map(float, lims[1:-1].split('}_{'))
        avg_unc = (low + high) / 2.

        # TODO: implement asymmetric priors
        dist = TruncNorm(0, np.inf, mean, avg_unc)
    return dist

def _search_row(name):
    if name.startswith('SN'):
        name = name[2:]
    with open(bdfname, 'r') as f:
        for line in f:
            if line.startswith(name):
                return line.split()
    raise KeyError(name)
