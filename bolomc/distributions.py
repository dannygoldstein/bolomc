
__author__ = 'Danny Goldstein <dgold@berkeley.edu>'
__whatami__ = 'Probability distributions for bolomc.' 

from scipy import stats

class TruncNorm(object):

    def __init__(self, lower, upper, loc, scale):
        self.lower = float(lower)
        self.upper = float(upper)
        self.loc = float(loc)
        self.scale = float(scale)

    @property
    def a(self):
        try:
            return self._a
        except AttributeError:
            self._a = (self.lower - self.loc) / self.scale
            return self._a

    @property
    def b(self):
        try:
            return self._b
        except AttributeError:
            self._b = (self.upper - self.loc) / self.scale
            return self._b

    def __call__(self, x):
        return stats.truncnorm.logpdf(x, self.a, self.b, loc=self.loc,
                                      scale=self.scale)

    def rvs(self, n=None):
        return stats.truncnorm.rvs(self.a, self.b, loc=self.loc, 
                                   scale=self.scale, size=n)

