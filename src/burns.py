
__author__ = 'Danny Goldstein <dgold@berkeley.edu>'
__whatami__ = 'Burns+2014 data interface.'
__all__ = ['get_mwebv']

import os

bdfname = os.path.join(__file__, '../data/burns2014.tab')

def get_mwebv(name):
    """Get the E(B-V)_MW and uncertainty for SN `name` from Table 1 of 
    Burns+2014."""
    row = _search_row(name)
    token = row[4]
    left, right = token.split(')')
    right = right[:-1]
    ebmv = float(left)
    err = float(right)
    return ebmv, err

def _get_burns(name):
    if name.startswith('SN'):
        name = name[2:]
    with open(bdfname, 'r') as f:
        for line in f:
            if line.startswith(name):
                return line.split()
    raise KeyError(name)
