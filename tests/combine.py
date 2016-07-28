import sys
import glob
import pickle
import sncosmo
from copy import copy
from bolomc import burns, bump, bolo
import numpy as np
from itertools import chain
from astropy.cosmology import Planck13

vparams = ['t0', 's', 'amplitude', 'UV_bump_amp', 'blue_bump_amp', 'i1_bump_amp', 'i2_bump_amp',
           'y1_bump_amp', 'y2_bump_amp', 'y3_bump_amp', 'j1_bump_amp',
           'j2_bump_amp', 'j3_bump_amp', 'h1_bump_amp', 'h2_bump_amp',
           'h3_bump_amp', 'k1_bump_amp', 'k2_bump_amp']

dwave = 10.

def bolometric(model, luminosity=True):
    phase = model.source._phase
    flux = np.sum(model.source.flux(phase, model.source._wave) * dwave, axis=1)
    if luminosity:
        z = model.get('z')
        dl = Planck13.luminosity_distance(z).to('cm').value
        L = 4 * np.pi * flux * dl * dl
        return L
    return flux

def Lfunc(model):
    y = bolometric(model)
    func = interp1d(x, y, kind='cubic')
    return func

def tpeak(model, retfunc=False):
    func = Lfunc(model)

    def minfunc(t):
        # objective function
        try:
            return -func(t) / 1e43
        except ValueError:
            return np.inf

    res = minimize(minfunc, 0.)
    if not res.success:
        raise RuntimeError(res.message)
    return res.x if not retfunc else (res.x, func)

def Lpeak(model):
    tpeak, func = tpeak(model, retfunc=True)
    return func(tpeak)

def dm15(model):
    tpeak, func = tpeak(model, retfunc=True)
    lpeak = func(tpeak)
    l15 = func(tpeak + 15)
    return 2.5 * np.log10(lpeak / l15)

def read_samples(pkl):
    raw = pickle.load(open(pkl, 'rb'))
    thinned = raw[:, [0, -1]]
    flattened = thinned.reshape(-1, 18)
    return flattened

def make_dictionaries(sn, samples, ebv, rv):
    nvparams = nonvparams(sn)
    dicts = []
    for sample in samples:
        d = dict(zip(vparams, sample))
        d.update(nvparams)
        d.update({'hostebv':ebv, 'hostr_v':rv})
        dicts.append(d)
    return dicts

def nonvparams(snname):
    lc = sncosmo.read_lc('../data/CSP_Photometry_DR2/%sopt+nir_photo.dat' % snname,
                         format='csp')
    z = lc.meta['zcmb']
    mwebv, _ = burns.get_mwebv(snname)
    return {'z':z, 'mwebv':mwebv}
    

if __name__ == '__main__':
    snname = sys.argv[1]
    pklfiles = glob.glob('fits/*%s*.pkl' % snname)
    samples = []
    nvparams = []
    for f in pklfiles:
        try:
            s = read_samples(f)
        except:
            continue
        else:
            samples.append(s)
            qual = f.split('__')[1].split('.pkl')[0]
            ebv = float(qual.split('_')[1])
            rv = float(qual.split('_')[-1])
            nvparams.append([ebv, rv])
    dicts = []
    for samps, (ebv, rv) in zip(samples, nvparams):
        dicts.append(make_dictionaries(snname, samps, ebv, rv))
    dicts = list(chain(*dicts))
    samples = np.vstack(samples)

    pruned = samples[np.random.choice(range(len(dicts)), replace=False, size=1000)]

    model = sncosmo.Model(bump.BumpSource(),
                          effect_names=['host','mw'],
                          effect_frames=['rest','obs'],
                          effects=[sncosmo.OD94Dust(), sncosmo.F99Dust()])
    mods = [copy(model) for sample in pruned]
    for d, mod in zip(dicts, mods):
        mod.set(**d)

    bolos = [bolometric(mod) for mod in mods]
    stack = bolo.LCStack(mod.source._phase, bolos)
    ax = stack.plot()
    ax.figure.savefig('combined.pdf')
    
