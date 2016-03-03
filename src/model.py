
__author__ = 'Danny Goldstein <dgold@berkeley.edu>'
__whatami__ = 'Predictive model for SN Ia bolometric light curves ' \
              'given CSP photometry and host reddening estimates.'

import os
import sys
import sncosmo
import numpy as np
import emcee as em

from .burns import get_ebvmw
from .vec import ParamVec
    
class FitContext(object):

    """Implementation of the PGM (Figure 1) from Goldstein & Kasen
    (2016). Defines a set of priors and a likelihood function for
    predicting bolometric light curves given broadband CSP photometry.
    """
    
    def __init__(self, lcfile, dust_type=sncosmo.CCM89Dust, exclude_bands=None):
        self.dust_type = dust_type
        self.exclude_bands = exclude_bands
        self.lc = sncosmo.read_lc(lcfile, format='csp')
        self.lc.sort(['filter', 'mjd'])
        self.mwebv = get_mwebv(self.lc.meta['name'])

        self.bands = [sncosmo.get_bandpass(band) for
                      band in np.unique(self.lc['filter']) if
                      band not in self.exclude_bands]
        
        
    def __call__(self, params):
        try:
            vec = ParamVec(params)
        except BoundsError as e:
            return -np.inf
        return self.logprior(vec) + self.loglike(vec)

    def create_model(self, vec):
        model = sncosmo.Model(source='salt2',
                              effects=[self.dust_type(),
                                       self.dust_type()],
                              effect_names=['host','mw'],
                              effect_frames=['obs','rest'])
        
        # Don't use SALT2 CL. 
        model.set(c=0.)
        # Burns+14 analysis uses RV_MW = 3.1.
        model.set(mwr_v=3.1)

        model.set(mwebv=self.mwebv)
        model.set(hostr_v=vec.rv)
        model.set(hostebv=vec.ebv)
        model.set(x0=vec.x0)
        model.set(x1=vec.x1)
        
        return model

    def logprior(self, vec):
        pass
        
    def loglike(self, vec):
        model = 
        pass
        

if __name__ == "__main__":
    
    # Fit a single light curve with the model.

    lc_filename = sys.argv[1]
    fc = FitContext(lc_filename)
    
    # Run the model. 
    
    sampler = emcee.EnsembleSampler()
