
__author__ = 'Danny Goldstein <dgold@berkeley.edu>'
__whatami__ = 'Parameter vector class for MCMC sampling.'

from .exceptions import BoundsError

class ParamVec(object):
        
    def __init__(self, vec, np, nl):
        self.vec = vec
        self._check_bounds()
        self.np = np
        self.nl = nl

    def _check_bounds(self):
        
        ermsg = '%s is out of bounds (%.4f, %.4f): %s'
        inclermsg = '%s is out of bounds [%.4f, %.4f): %s'
        
        if self.lt <= 0:
            raise BoundsError(ermsg % ('lt', 0, np.inf, self.lt))
        if self.llam <= 0:
            raise BoundsError(ermsg % ('llam', 0, np.inf, self.llam))
        if self.rv <= 0:
            raise BoundsError(ermsg % ('rv', 0, np.inf, self.rv))
        if self.amplitude < 0:
            raise BoundsError(inclermsg % ('amplitude', 0, 
                                           np.inf, self.amplitude))

                
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
        return self.vec[4:].reshape(np, nl)
        
    
