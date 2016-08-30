import sys
import yaml
import sncosmo
import pickle
import numpy as np
import pandas as pd
from copy import copy
from bolomc import bump, burns

__whatami__ = "Construct a bolometric light curve from CSP data."
__author__ = "Danny Goldstein <dgold@berkeley.edu>"


# Shorthand for a few variables with long names.
od94 = sncosmo.OD94Dust
f99 = sncosmo.F99Dust

# Read the LC to fit. 
lc = sncosmo.read_lc('scripts/sn2011fe.lc', format='ascii')
name = lc.meta['name']

# Configure the properties of the host galaxy dust.
dust_type = od94 if config['dust_type'] == 'od94' else f99
bintype = config['burns_bintype']

# rv, ebv
my_jobs = [(3.1, 0.)]

if len(my_jobs) == 0:
    result = None  # this processor will be idle
    param_names = None
else:
    result = []
    param_names = []
    for job in my_jobs:
        r_v, ebv = job

        # Create model. 
        model = bump.bump_model(dust_type)
        param_names.append(model._param_names)

        model.set(z=lc.meta['zhelio'])
        model.set(mwebv=0.011)
        model.set(hostebv=ebv)
        model.set(hostr_v=r_v)
        model.set(t0=lc.meta['t0'])

        # Identify parameters to vary in the fit.
        vparams = filter(lambda x: 'bump' in x or 'slope' in x, model._param_names)
        vparams += ['t0', 's', 'amplitude']

        # Set boundaries on the parameters to vary.
        bounds = {b.name + "_bump_amp":(-1,2) for b in 
                  model.source.bumps}
        bounds['s'] = (0, 3.)

        # Get an idea of where the mode of the posterior is by doing
        # an MLE fit.
        res, model = sncosmo.fit_lc(lc, model, vparams, bounds=bounds, 
                                    mag=True)

        # Add bounds for MCMC fit.
        bounds['t0'] = (model.get('t0') - 2, model.get('t0') + 2)
        bounds['amplitude'] = (0.5 * model.get('amplitude'), 
                               2 * model.get('amplitude'))

        # Do MCMC.
        fres, fitmod = sncosmo.mcmc_lc(lc, model, vparams,
                                       bounds=bounds,
                                       nwalkers=config['nwalkers'],
                                       nburn=config['nburn'],
                                       nsamples=config['nsamples'],
                                       guess_t0=False, guess_amplitude=False,
                                       mag=True)
        samples = fres.samples

        # Represent results as a list of dictionaries mapping
        # parameter names to parameter values. Ultimately, we want all
        # of the model parameters, including the ones that do not
        # vary, as a big numpy array. In order to do that we will
        # access the `parameters` attribute of the
        
        pdicts = [dict(zip(fres.vparam_names, sample)) for sample in samples]

        # Aggregate all of the parameters, including the ones that do
        # not vary.
        
        parameters = []
        for d in pdicts:
            fitmod.set(**d)
            parameters.append(copy(fitmod.parameters))
        result.append(np.asarray(parameters))
        param_names.append(fitmod._param_names)
       
    result = np.vstack(result)
    param_names = np.vstack(param_names)

# Send everything to the master process.     
gathered = comm.gather(result, root=0)
pnames_gathered = comm.gather(param_names, root=0)

if rank == 0:
    # with the results gathered from everyone, write the aggregate to
    # a file
    
    ffunc = lambda a: a is not None
    samples = np.vstack(filter(ffunc, gathered))
    param_names = np.vstack(filter(ffunc, pnames_gathered))

    # Ensure all names are the same.
    if not (param_names == param_names[0]).all():
        raise Exception("Inter-run model parameter names are inconsistent.")

    df = pd.DataFrame(data=samples, columns=param_names[0])
    pickle.dump((lc, config, df.to_records(index=False)), 
                open(config['outfile_name'], 'wb'))
