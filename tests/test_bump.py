
import matplotlib 
matplotlib.use('Agg')
import sncosmo
from copy import copy
import numpy as np
from bolomc import bump
from bolomc import burns
from bolomc.distributions import TruncNorm
import glob
from bolomc import plotting

from joblib import Parallel, delayed

def task(filename, kind='mcmc'):
    
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

    res, model = sncosmo.fit_lc(lc,model,['amplitude']+vparams,
                                bounds=bounds)

    bounds['t0'] = (model.get('t0')-2, model.get('t0')+2)
    
    vparams.append('amplitude')
    bounds['amplitude'] = (0.5 * model.get('amplitude'), 
                           2 * model.get('amplitude'))

    if kind == 'mcmc':

        result = sncosmo.mcmc_lc(lc, model, vparams, priors={'hostebv':ebv_prior,
                                                             'hostr_v':rv_prior},
                                 bounds=bounds,
                                 nwalkers=500,
                                 nburn=1000,
                                 nsamples=20)
        
        samples = result[0].samples.reshape(500, 20, -1)
        vparams = result[0].vparam_names
        plot_arg = np.rollaxis(samples, 2)
        plotting.plot_chains(plot_arg, param_names=vparams, 
                             filename='fits/%s_samples.pdf' % lc.meta['name'])

        dicts = [dict(zip(vparams, samp)) for samp in samples.reshape(500 * 20, -1)]
        thinned = samples.reshape(500, 20, -1)[:, [0, -1]].reshape(1000, -1)

        models = [copy(result[1]) for i in range(len(thinned))]
        for d, m in zip(dicts, models):
            m.set(**d)

        fig = sncosmo.plot_lc(data=lc, model=models, ci=(50-68/2., 50., 50+68/2.))
        fig.savefig('fits/%s.pdf' % lc.meta['name'])
        
    else:
        
        fitres, model = sncosmo.fit_lc(lc, model, vparams)
        fig = sncosmo.plot_lc(data=lc, model=model)
        fig.savefig('fits/%s_fit.pdf' % lc.meta['name'])


if __name__ == '__main__':
    lc_files = glob.glob('../data/CSP_Photometry_DR2/*.dat')
    Parallel(n_jobs=1)(delayed(task)(f) for f in lc_files if '2005eq' in f)
