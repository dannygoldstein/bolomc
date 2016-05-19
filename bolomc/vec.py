
__author__ = 'Danny Goldstein <dgold@berkeley.edu>'
__whatami__ = 'Parameter vector class for MCMC sampling.'

import numpy as np
from errors import BoundsError

class ParamVec(object):
        
    def __init__(self, vec, nph, nl, check_bounds=True):
        self.vec = vec
        self.nph = nph
        self.nl = nl
        if check_bounds:
            self._check_bounds()

    def _check_bounds(self):
        
        ermsg = '%s is out of bounds (%.4f, %.4f): %s'
        inclermsg = '%s is out of bounds [%.4f, %.4f): %s'
        
        if self.lp <= 0:
            raise BoundsError(ermsg % ('lp', 0, np.inf, self.lp))
        if self.llam <= 0:
            raise BoundsError(ermsg % ('llam', 0, np.inf, self.llam))
        if self.rv <= 0:
            raise BoundsError(ermsg % ('rv', 0, np.inf, self.rv))
        if (self.sedw < 0).any():
            raise BoundsError(inclermsg % ('sedw', 0, np.inf, self.sedw))
                
    @property
    def lp(self):
        return self.vec[0]
        
    @lp.setter
    def lp(self, x):
        self.vec[0] = x
    
    @property
    def llam(self):
        return self.vec[1]

    @llam.setter
    def llam(self, x):
        self.vec[1] = x
        
    @property
    def rv(self):
        # host
        return self.vec[2]
        
    @rv.setter
    def rv(self, x):
        self.vec[2] = x
        
    @property
    def ebv(self):
        # host
        return self.vec[3]
        
    @ebv.setter
    def ebv(self, x):
        self.vec[3] = x
    
    @property
    def sedw(self):
        return self.vec[4:].reshape(self.nph, self.nl)

    @sedw.setter
    def sedw(self, x):
        view = self.vec[4:].reshape(self.nph, self.nl)
        view[:, :] = x
        self.vec[4:] = view.reshape(self.nph * self.nl)
        
    @property
    def D(self):
        return self.vec.size
