
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
import pickle
from joblib import Parallel, delayed

def task(filename, i, j, nrv, nebv, kind='mcmc'):
    
    lc = sncosmo.read_lc(filename, format='csp')

    model = sncosmo.Model(bump.BumpSource(),
                          effect_names=['host','mw'],
                          effect_frames=['rest','obs'],
                          effects=[sncosmo.OD94Dust(), sncosmo.F99Dust()])

    rv_prior = burns.get_hostrv_prior(lc.meta['name'], 
                                      'gmm', sncosmo.OD94Dust)

    host_ebv, err = burns.get_hostebv(lc.meta['name'])
    ebv_prior = TruncNorm(-np.inf, np.inf, host_ebv, err)

    rv_prior, low, high = burns.get_hostrv_prior(lc.meta['name'], 
                                                 'gmm', sncosmo.OD94Dust,
                                                 retlims=True)
    host_ebv, err = burns.get_hostebv(lc.meta['name'])
    
    rv = np.linspace(low if low >= 0 else 0, high, nrv)[i]
    ebvlo = host_ebv - err
    ebvhi = host_ebv + err
    ebv = np.linspace(ebvlo if ebvlo >= 0 else 0, ebvhi, nebv)[j]

    model.set(z=lc.meta['zcmb'])
    model.set(mwebv=burns.get_mwebv(lc.meta['name'])[0])
    model.set(hostebv=ebv)
    model.set(hostr_v=rv)
    model.set(t0=burns.get_t0(lc.meta['name']))

    vparams = filter(lambda x: 'bump' in x, model._param_names)
    vparams += ['t0', 's']
    bounds = {b.name + "_bump_amp":(-1,2) for b in 
                                     model.source.bumps}
    #bounds['hostr_v'] = (rv_prior.mean - 0.5, rv_prior.mean + 0.5)
    #bounds['hostebv'] = (0, 0.2)
    bounds['s'] = (0, 3.)

    res, model = sncosmo.fit_lc(lc,model,['amplitude']+vparams,
                                bounds=bounds)

    bounds['t0'] = (model.get('t0')-2, model.get('t0')+2)
    
    vparams.append('amplitude')
    bounds['amplitude'] = (0.5 * model.get('amplitude'),
                           2 * model.get('amplitude'))

    qualifier = '_ebv_%.2f_rv_%.2f' % (ebv, rv)

    if kind != 'fit':
        if kind == 'mcmc':
            result = sncosmo.mcmc_lc(lc, model, vparams,
                                     bounds=bounds,
                                     nwalkers=500,
                                     nburn=1000,
                                     nsamples=20)
        elif kind == 'nest':
            result = sncosmo.nest_lc(lc, model, vparams, bounds=bounds,
                                     method='multi', npoints=800)

        samples = result[0].samples.reshape(500, 20, -1)
        vparams = result[0].vparam_names
        plot_arg = np.rollaxis(samples, 2)

        plotting.plot_chains(plot_arg, param_names=vparams, 
                             filename='fits/%s_samples%s.pdf' % (lc.meta['name'], qualifier))

        dicts = [dict(zip(vparams, samp)) for samp in samples.reshape(500 * 20, -1)]
        thinned = samples.reshape(500, 20, -1)[:, [0, -1]].reshape(1000, -1)

        pickle.dump(samples, open('fits/samples_%s.pkl' % qualifier, 'wb'))

        models = [copy(result[1]) for i in range(len(thinned))]
        for d, m in zip(dicts, models):
            m.set(**d)

        fig = sncosmo.plot_lc(data=lc, model=models, ci=(50-68/2., 50., 50+68/2.),
                              model_label=lc.meta['name'])
        fig.savefig('fits/%s%s.pdf' % (lc.meta['name'], qualifier))
        
    else:
        
        fitres, model = sncosmo.fit_lc(lc, model, vparams, bounds=bounds)
        fig = sncosmo.plot_lc(data=lc, model=model)
        fig.savefig('fits/%s_fit%s.pdf' % (lc.meta['name'], qualifier))


if __name__ == '__main__':
    lc_files = glob.glob('../data/CSP_Photometry_DR2/*.dat')
    nebv = 5
    nrv = 5
    for f in lc_files:
        if 'SN2005eq' in f:
            Parallel(n_jobs=nebv*nrv)(delayed(task)(f, i, j, nrv, nebv, kind='mcmc') 
                                      for i in range(nrv) for j in range(nebv))
