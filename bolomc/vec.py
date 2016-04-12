
__author__ = 'Danny Goldstein <dgold@berkeley.edu>'
__whatami__ = 'Parameter vector class for MCMC sampling.'

from .exceptions import BoundsError

class ParamVec(object):
    
    def __init__(self, vec):
        self.vec = vec
        self._check_bounds()
        
    def _check_bounds(self):
        
        ermsg = '%s is out of bounds (%.4f, %.4f): %s'
        
        if self.lt <= 0:
            raise BoundsError(ermsg % ('lt', 0, np.inf, self.lt))
        if self.llam <= 0:
            raise BoundsError(ermsg % ('llam', 0, np.inf, self.llam))
        if self.rv <= 0:
            raise BoundsError(ermsg % ('rv', 0, np.inf, self.rv))
    
    @property
    def x0(self):
        return self.vec[0]

    @propety
    def x1(self):
        return self.vec[1]
        
    @property
    def t0(self):
        return self.vec[2]
        
    @property
    def lt(self):
        return self.vec[3]
    
    @property
    def llam(self):
        return self.vec[4]
        
    @property
    def rv(self):
        return self.vec[5]
        
    @property
    def ebv(self):
        return self.vec[6]
    
    @property
    def sed_warp(self):
        return self.vec[6:]
