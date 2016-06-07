
__author__ = "Danny Goldstein <dgold@berkeley.edu>"
__whatami__ = "Utilities for setting up an optimal wavelength grid."

import numpy as np
from util import filter_to_wave_eff as we

def any_v(filters):
    return 'cspv3009' in filters \
        or 'cspv3014' in filters \
        or 'cspv9844' in filters

def optimal_wavelength_grid(lc):

    grid = []
    f = lc['filter']
    
    filts, counts = np.unique(f, return_counts=True)
    sfilt = filts[np.argsort(counts)[::-1]]

    if 'cspu' in f:
        grid.append(we('cspu'))
    if 'cspb' in f:
        grid.append(we('cspb'))
    if 'cspg' in f and not ('cspb' in f and any_v(f)):
        grid.append(we('cspg'))
    if any_v(f):
        for filt in sfilt:
            if 'cspv' in filt:
                vmost = filt
                break
        grid.append(we(vmost))
    if 'cspr' in f:
        grid.append(we('cspr'))
    if 'cspi' in f:
        grid.append(we('cspi'))
    if 'cspys' in f or 'cspyd' in f:
        for filt in sfilt:
            if 'cspy' in filt:
                ymost = filt
                break
        grid.append(we(ymost))
    if 'cspjs' in f or 'cspjd' in f:
        for filt in sfilt:
            if 'cspj' in filt:
                jmost = filt
                break
        grid.append(we(jmost))
    if 'csphs' in f or 'csphd' in f:
        for filt in sfilt:
            if 'csph' in filt:
                hmost = filt
                break
        grid.append(we(hmost))
    if 'cspk' in f:
        grid.append(we('cspk'))

    return np.asarray(grid)
