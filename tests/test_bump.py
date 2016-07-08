import matplotlib 
matplotlib.use('Agg')
import sncosmo
import numpy as np
from bolomc import bump
from bolomc import burns
from bolomc.distributions import TruncNorm
import glob

lc_files = glob.glob('../data/CSP_Photometry_DR2/*.dat')[40:70]
for lc_file in lc_files:
    try:

        lc = sncosmo.read_lc(lc_file, format='csp')

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
        bounds = {b.name + "_bump_amp":(-1.5,1.5) for b in 
                                         model.source.bumps}
        bounds['hostr_v'] = (0, 6.)
        bounds['hostebv'] = (0, 0.2)
        bounds['s'] = (0, 3.)

        res, model = sncosmo.fit_lc(lc,model,['amplitude0', 'amplitude1']+vparams,
                                    bounds=bounds)


        """
        result = sncosmo.mcmc_lc(lc, model, vparams, priors={'hostebv':ebv_prior,
                                                             'hostr_v':rv_prior},
                                 bounds=bounds,
                                 nwalkers=200,
                                 nburn=1,
                                 nsamples=20)
        """




        #model.set(**dict(zip(vparams, ans)))
        #fig=sncosmo.plot_lc(data=lc, model=model)
        #fig.savefig('test.pdf')

        fig = sncosmo.plot_lc(data=lc, model=model)
        fig.savefig('fits/%s.pdf' % lc.meta['name'])
        plt.close(fig)
        print 'saved %s' % lc.meta['name']
    except:
        continue
