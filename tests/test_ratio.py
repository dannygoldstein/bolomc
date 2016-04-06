
from bolomc import model

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

LC_W_NEG_RATIO = path('SN2005el')
LC_W_POS_RATIO = path('SN2009F')

def testNegRatio():
    fc_neg = model.FitContext(LC_W_NEG_RATIO)
    assert any(fc_neg.lc['ratio'] < 0)

def testPosRatio():
    fc_pos = model.FitContext(LC_W_POS_RATIO)
    assert all(fc_pos.lc['ratio'] > 0)
    
    
    
