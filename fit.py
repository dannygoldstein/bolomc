import sys
import yaml
import numpy as np
from copy import copy
from mpi4py import MPI
from bolomc import bump, burns

# Initialize MPI. 
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# partition an iterable i into n parts
_split = lambda i,n: [i[:len(i)/n]]+_split(i[len(i)/n:],n-1) if n != 0 else []

# Shorthand for a few variables with long names.
od94 = sncosmo.OD94Dust
f99 = sncosmo.F99Dust

# Load the run configuration file. 
config_filename = sys.argv[1]
config = yaml.load(config_filename)

# Read the LC to fit. 
lc = sncosmo.read_lc(config['lc_filename'], format='csp')
name = lc.meta['name']

# Configure the properties of the host galaxy dust.
dust_type = od94 if config['dust_type'] == 'OD94' else f99
bintype = config['burns_bintype']

# Get the host galaxy dust data.
host_ebv, err = burns.get_hostebv(name)
_, rv_low, rv_hi = burns.get_hostrv_prior(name, bintype, dust_type,
                                          retlims=True)
ebvlo = host_ebv - err
ebvhi = host_ebv + err

# Do ebv / r_v gridding. 
nrv = config['nrv']
nebv = config['nebv']
rv_grid = np.linspace(rv_low if rv_low >= 0 else 0, rv_high, nrv)
ebv_grid = np.linspace(ebvlo if ebvlo >= 0 else 0, ebvhi, nebv)
total_grid = [(rv, ebv) for rv in rv_grid for ebv in ebv_grid]
my_jobs = _split(total_grid, size)[rank]

if len(my_jobs) == 0:
    result = None  # this processor will be idle
    names = None
else:
    result = []
    for job in my_jobs:
        r_v, ebv = job

        # Create model. 
        model = sncosmo.Model(bump.BumpSource(),
                              effect_names=['host','mw'],
                              effect_frames=['rest','obs'],
                              effects=[dust_type(), sncosmo.F99Dust()])

        model.set(z=lc.meta['zcmb'])
        model.set(mwebv=burns.get_mwebv(name)[0])
        model.set(hostebv=ebv)
        model.set(hostr_v=r_v)
        model.set(t0=burns.get_t0(lc.meta['name']))
        
        # prepare for initial fit. 
        vparams = filter(lambda x: 'bump' in x, model._param_names)
        vparams += ['t0', 's', 'amplitude']

        bounds = {b.name + "_bump_amp":(-1,2) for b in 
                  model.source.bumps}
        bounds['s'] = (0, 3.)

        res, model = sncosmo.fit_lc(lc,model,['amplitude']+vparams,
                                    bounds=bounds)

        bounds['t0'] = (model.get('t0')-2, model.get('t0')+2)
        bounds['amplitude'] = (0.5 * model.get('amplitude'),
                               2 * model.get('amplitude'))

        qualifier = '_ebv_%.2f_rv_%.2f' % (ebv, r_v)

        samples, fitmod = sncosmo.mcmc_lc(lc, model, vparams,
                                          bounds=bounds,
                                          nwalkers=config['nwalkers'],
                                          nburn=config['nburn'],
                                          nsamples=config['nsamples'])

        models = [copy(fitmod) for sample in samples]
        pdicts = [dict(zip(vparams, sample)) for sample in samples]
        for m, d in zip(models, pdicts):
            m.set(**d)
        
        
    result = np.vstack(result)
    mods = []
    for samp in results:
        
    names = model._param_names
    
    
    
gathered = comm.gather(result, root=0)
names = comm.gather(names, root=0)

if rank == 0:
    samples = np.vstack(filter(lambda a: a is not None, gathered))
    names = 
    
    
