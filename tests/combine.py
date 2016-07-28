import sys
import glob
import pickle
import sncosmo
from copy import copy
from bolomc import burns, bump, bolo
import numpy as np
from itertools import chain

vparams = ['UV_bump_amp', 'blue_bump_amp', 'i1_bump_amp', 'i2_bump_amp',
           'y1_bump_amp', 'y2_bump_amp', 'y3_bump_amp', 'j1_bump_amp',
           'j2_bump_amp', 'j3_bump_amp', 'h1_bump_amp', 'h2_bump_amp',
           'h3_bump_amp', 'k1_bump_amp', 'k2_bump_amp', 't0', 's', 'amplitude']

def read_samples(pkl):
    try:
        raw = pickle.load(open(pkl, 'rb'))
    except:
        pass
    thinned = raw[:, [0, -1]]
    flattened = thinned.reshape(-1, 18)
    return flattened

def make_dictionaries(sn, samples, ebv, rv):
    nvparams = nonvparams(sn)
    for sample in samples:
        d = dict(zip(vparams, sample))
        d.update(nvparams)
        d.update({'hostebv':ebv, 'hostrv':rv})
    return d

def nonvparams(snname):
    lc = sncosmo.read_lc('../data/CSP_Photometry_DR2/S%sopt+nir_photo.dat' % snname,
                         format='csp')
    z = lc.meta['zcmb']
    mwebv, _ = burns.get_mwebv(snname)
    return {'z':z, 'mwebv':mwebv}
    

if __name__ == '__main__':
    snname = sys.argv[1]
    pklfiles = glob.glob('*%s*.pkl' % snname)
    samples = []
    nvparams = []
    for f in pklfiles:
        try:
            samples.append(read_samples(f))
        except:
            continue
        else:
            qual = f.split('__')[1].split('.')[0]
            ebv = float(qual.split('_')[1])
            r_v = float(qual.split('_')[-1])
            nvparams.append([ebv, rv])
    dicts = []
    for samps, (ebv, rv) in zip(samples, nvparams):
        dicts.append(make_dictionaries(snname, samps, ebv, rv))
    dicts = list(chain(*dicts))

    pruned = samples[np.random.choice(range(len(dicts)), replace=False, size=1000)]

    model = sncosmo.Model(bump.BumpSource(),
                          effect_names=['host','mw'],
                          effect_frames=['rest','obs'],
                          effects=[sncosmo.OD94Dust(), sncosmo.F99Dust()])
    mods = [copy(model) for sample in pruned]
    for d, mod in zip(dicts, mods):
        mod.set(**d)

    bolos = [mod.source.bolometric(mod.source._phase) for mod in mods]
    stack = bolo.LCStack(mod.source._phase, bolos)
    stack.plot()
    
