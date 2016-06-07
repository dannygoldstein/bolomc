
# Create the run files for a grid of LC fits

import os
import glob

exe = os.path.join('/'.join(os.path.abspath(__file__).split('/')[:-1]), 
                   'bolomc/model.py')

# Parameters

#LOGFILE = None # Use stdout.
NBURN = 5000 # Number of burn-in iterations.
NSAMP = 50 # Number of sampling iterations.
NL = 10 * 2 
#NL = None # Number of wavelength knots (regular grid).
NWALKERS = 1000 # Number of walkers in the ensemble.
NTHREADS = 4 # Number of threads for MCMC sampling. 
#EXCLUDE_BANDS = [] # Fit all bandpasses given. 
#DUST_TYPE = 'OD94' # Host galaxy dust reddening law.
#RV_BINTYPE = 'gmm' # Host galaxy Rv prior type. 
SPLINT_ORDER = 3 # Spline interpolation order.
NPH = 10 * 2 


basecmd = """
#!/bin/bash
{exe:s} run \\
    {lc_filename:s} \\
    {nph:d} \\
    {outfile:s} \\
    --logfile {logfile:s} \\
    --fc_fname {fc_fname:s}"""

try:
    os.mkdir('run')
except:
    pass

stringify = lambda l: " ".join(l)

lcfiles = glob.glob('data/*/*')
for fn in lcfiles:
    runf_basename = fn.split('/')[-1].split('opt')[0]
    fname = 'run/%s.run_hires' % runf_basename
    outfile = '%s.h5_hires' % runf_basename
    logfile = '%s.log_hires' % runf_basename
    lc_filename = os.path.abspath(fn)
    fc_fname = '%s.fc.pkl_hires' % runf_basename
    with open(fname, 'w') as f:
        f.write(basecmd.format(exe=exe, lc_filename=lc_filename, 
                               outfile=outfile, logfile=logfile,
                               nph=NPH, fc_fname=fc_fname))
        try:
            f.write(' \\\n    --nburn {nburn:d}'.format(nburn=NBURN))
        except NameError:
            pass
        try: 
            f.write(' \\\n    --nsamp {nsamp:d}'.format(nsamp=NSAMP))
        except NameError:
            pass
        try:
            f.write(' \\\n    --nl {nl:d}'.format(nl=NL))
        except NameError:
            pass
        try:
            f.write(' \\\n    --nwalkers {nwalkers:d}'.format(nwalkers=NWALKERS))
        except NameError:
            pass
        try:
            f.write(' \\\n    --nthreads {nthreads:d}'.format(nthreads=NTHREADS))
        except NameError:
            pass
        try:
            f.write(' \\\n    --exclude_bands {exclude_bands:s}'.format(exclude_bands=stringify(EXCLUDE_BANDS)))
        except NameError:
            pass
        try:
            f.write(' \\\n    --dust_type {dust_type:s}'.format(dust_type=DUST_TYPE))
        except NameError:
            pass
        try:
            f.write(' \\\n    --rv_bintype {rv_bintype:s}'.format(rv_bintype=RV_BINTYPE))
        except NameError:
            pass
        try:
            f.write(' \\\n    --splint_order {splint_order:d}'.format(splint_order=SPLINT_ORDER))
        except NameError:
            pass
        f.write('\n')
