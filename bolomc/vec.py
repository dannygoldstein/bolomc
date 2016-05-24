
__author__ = 'Danny Goldstein <dgold@berkeley.edu>'
__whatami__ = 'Parameter vector class for MCMC sampling.'

import numpy as np
from errors import BoundsError

class ParamVec(object):

    @classmethod
    def from_file(cls, f):
        with open(f, 'r') as f:
            r1 = f.readline()
            nph, nl = [int(e[3:]) for e in r1.split()]
            a = np.genfromtxt(f)
        idx = a[:, :2].astype('<i8')
        shape = idx.max(0) + 1
        return np.asarray([cls(r, nph, nl) for r in a[:, 3:]]).reshape(*shape)
        
    def __init__(self, vec, nph, nl, check_bounds=True):
        self.vec = vec
        self.nph = nph
        self.nl = nl
        if check_bounds:
            self._check_bounds()

    def _check_bounds(self):
        
        ermsg = '%s is out of bounds (%.4f, %.4f): %s'
        inclermsg = '%s is out of bounds [%.4f, %.4f): %s'
        
        if self.rv <= 0:
            raise BoundsError(ermsg % ('rv', 0, np.inf, self.rv))
        if (self.sedw < 0).any():
            raise BoundsError(inclermsg % ('sedw', 0, np.inf, self.sedw))
                
    @property
    def rv(self):
        # host
        return self.vec[0]
        
    @rv.setter
    def rv(self, x):
        self.vec[0] = x
        
    @property
    def ebv(self):
        # host
        return self.vec[1]
        
    @ebv.setter
    def ebv(self, x):
        self.vec[1] = x
    
    @property
    def sedw(self):
        return self.vec[2:].reshape(self.nph, self.nl)

    @sedw.setter
    def sedw(self, x):
        view = self.vec[2:].reshape(self.nph, self.nl)
        view[:, :] = x
        self.vec[2:] = view.reshape(self.nph * self.nl)
        
    @property
    def D(self):
        return w + self.sedw.size
