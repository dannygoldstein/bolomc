__author__ = 'Danny Goldstein <dgold@berkeley.edu>'
__whatami__ = 'bolomc helper functions.'

import sncosmo

def filter_to_wave_eff(filt):
    filt = sncosmo.get_bandpass(filt)
    return filt.wave_eff
