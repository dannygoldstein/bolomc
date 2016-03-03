
import re
import os
import sys
import sncosmo
import numpy as np
import emcee as em

join = os.path.join
burns14data_filename = join(__file__, '../data/burns2014.tab')

def salt2modelccm89():
    dust = sncosmo.CCM89Dust
    model = sncosmo.Model(source='salt2',
                          effects=[dust(),dust()],
                          effect_names=['host','mw'],
                          effect_frames=['rest','obs'])
    return model
    
def hsiaomodelccm89():
    dust = sncosmo.CCM89Dust
    model = sncosmo.Model(source='hsiao',
                          effects=[dust(),dust()],
                          effect_names=['host','mw'],
                          effect_frames=['rest','obs'])
    return model

def _search_row(name):
    if name.startswith('SN'):
        name = name[2:]
    with open(burns14data_filename, 'r') as f:
        for line in f:
            if line.startswith(name):
                return line.split()
    raise KeyError(name)
    
def get_ebmvmw(name):
    row = _search_row(name)
    token = row[4]
    left, right = token.split(')')
    right = right[:-1]
    ebmv = float(left)
    err = float(right)
    return ebmv, err

if __name__ == "__main__":
    
    # Fit a single light curve with the model.

    lc_filename = sys.argv[1]
    lc = sncosmo.read_lc(lc_filename, format='csp')
    
    # model
    
    model = salt2modelccm89()
    
    
