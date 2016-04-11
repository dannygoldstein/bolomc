import os

__all__ = ['sever','path']

def sever(s):
    splstr = s.split('/')
    dir = '/'.join(splstr[:-1])
    f = splstr[-1]
    return dir, f

pwd, _ = sever(__file__)

def path(snname):
    suffix = 'opt+nir_photo.dat'
    prefix = '../data/CSP_Photometry_DR2/%s'
    return os.path.join(pwd, prefix % (snname + suffix))

