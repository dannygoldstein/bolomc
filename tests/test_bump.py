import matplotlib 
matplotlib.use('Agg')
import sncosmo
import numpy as np
from bolomc import bump
from bolomc import burns
from bolomc.distributions import TruncNorm

lc = sncosmo.read_lc('../data/CSP_Photometry_DR2/SN2005elopt+nir_photo.dat', format='csp')

model = sncosmo.Model(bump.FreeBumpSource(2), 
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

vparams = filter(lambda x: x not in ['z', 'hostebv', 'hostr_v', 't0', 's'], model._param_names)
bounds = {param:(-1, 1) for param in vparams}
vparams += ['hostebv','hostr_v', 't0', 's']
bounds['hostr_v'] = (0, 6.)
bounds['hostebv'] = (0, 0.2)

'''res, model = sncosmo.fit_lc(lc,model,['amplitude']+vparams,
                            bounds=bounds)'''

model.set(    z = +0.014885,
             t0 = +53645.135842,
      amplitude = 9e-7,
              s = +0.843746,
     mu_phase_0 = -0.411683, 
      mu_wave_0 = +0.865245, 
    sig_phase_0 = +0.384941, 
     sig_wave_0 = +0.631725, 
          cov_0 = +0.702311,
          amp_0 = -0.143817,
     mu_phase_1 = -0.950194,
      mu_wave_1 = +0.340015,
    sig_phase_1 = +0.830830,
     sig_wave_1 = -0.186577,
          cov_1 = +0.860065,
          amp_1 = -0.018582,
        hostebv = +0.005398,
        hostr_v = +1.077445,
          mwebv = +0.052835)

'''
result = sncosmo.mcmc_lc(lc, model, vparams, priors={'hostebv':ebv_prior,
                                                     'hostr_v':rv_prior},
                         bounds=bounds,
                         nwalkers=200,
                         nburn=1,
                         nsamples=20)

'''                


#model.set(**dict(zip(vparams, ans)))
#fig=sncosmo.plot_lc(data=lc, model=model)
#fig.savefig('test.pdf')

