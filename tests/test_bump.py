
import matplotlib 
matplotlib.use('Agg')
import sncosmo
from copy import copy
import numpy as np
from bolomc import bump
from bolomc import burns
from bolomc.distributions import TruncNorm
import glob
from pymc import Matplot

from joblib import Parallel, delayed

def task(filename):
    
    lc = sncosmo.read_lc(filename, format='csp')

    model = sncosmo.Model(bump.BumpSource(),
                          effect_names=['host','mw'],
                          effect_frames=['rest','obs'],
                          effects=[sncosmo.OD94Dust(), sncosmo.F99Dust()])

    rv_prior = burns.get_hostrv_prior(lc.meta['name'], 
                                      'gmm', sncosmo.OD94Dust)

    host_ebv, err = burns.get_hostebv(lc.meta['name'])
    ebv_prior = TruncNorm(-np.inf, np.inf, host_ebv, err)

    model.set(z=lc.meta['zcmb'])
    model.set(mwebv=burns.get_mwebv(lc.meta['name'])[0])
    model.set(hostebv=host_ebv)
    model.set(hostr_v=rv_prior.mean)
    model.set(t0=burns.get_t0(lc.meta['name']))

    vparams = filter(lambda x: 'bump' in x, model._param_names)
    vparams += ['hostebv','hostr_v', 't0', 's']
    bounds = {b.name + "_bump_amp":(-1,2) for b in 
                                     model.source.bumps}
    bounds['hostr_v'] = (0, 6.)
    bounds['hostebv'] = (0, 0.2)
    bounds['s'] = (0, 3.)

    res, model = sncosmo.fit_lc(lc,model,['amplitude0', 'amplitude1']+vparams,
                                bounds=bounds)


    result = sncosmo.mcmc_lc(lc, model, vparams, priors={'hostebv':ebv_prior,
                                                         'hostr_v':rv_prior},
                             bounds=bounds,
                             nwalkers=500,
                             nburn=1000,
                             nsamples=20)
    
    samples = result[0].samples
    Matplot.plot(samples, '%s.samples' % lc.meta['name'], format='pdf', path='fits',
                 common_scale=False)
    vparams = result[0].vparam_names
    dicts = [dict(zip(vparams, samp)) for samp in samples]

    thinned = samples.reshape(500, 20, -1)[:, [0, -1]].reshape(1000, -1)

    models = [copy(result[1]) for i in range(len(thinned))]
    for d, m in zip(dicts, models):
        m.set(**d)

    fig = sncosmo.plot_lc(data=lc, model=models, ci=(50-68/2., 50., 50+68/2.))
    fig.savefig('fits/%s.pdf' % lc.meta['name'])
    
    


lc_files = glob.glob('../data/CSP_Photometry_DR2/*.dat')[40:70]
Parallel(n_jobs=len(lc_files))(delayed(task)(f) for f in lc_files)
