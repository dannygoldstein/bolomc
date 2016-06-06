
import matplotlib
matplotlib.use("Agg")

import h5py
import sncosmo
import bolomc
import sys
import pickle


f = h5py.File('sine.h52')

cs = f['samples'] if f['current_stage'][()] else f['burn']
i = cs['last_index_filled'][()]

models = []

for k in range(50):
    params = cs['params'][i, k]
    fc = pickle.load(open('sinefc.pkl2','rb'))
    vec = bolomc.ParamVec(params, fc.nph, fc.nl)
    model = fc._create_model(vec)
    models.append(model)

fig = sncosmo.plot_lc(data=fc.lc, model=models)
fig.savefig('sine3.pdf')

#sncosmo.animate_source(model.source, fname="degraded.mp4")

#mod2 = fc._create_model(bolomc.ParamVec(cs['params'][i-200,0], fc.nph, fc.nl))
#mod3 = fc._create_model(bolomc.ParamVec(cs['params'][i-150,20], fc.nph, fc.nl))

"""control = params.copy()
control[4:] = 1.

vec_control = bolomc.ParamVec(control, fc.nph, fc.nl)
gp_control = bolomc.reconstruct_gp(fc, vec_control)
control_mod = fc._create_model(vec_control, gp_control)

fig = sncosmo.plot_lc(model=models+[control_mod], data=fc.lc)
fig.savefig('%s_%d_%d.pdf' % (fc.lc.meta['name'], i, 0))

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('ticks')

"""
""" This code doesn't work anymore
bolo = cs['bolo'][i, 0]
control_bolo = fc.bolo(vec_control, compute_luminosity=True)
phase = sncosmo.get_source('hsiao', version='3.0')._phase
fig, ax = plt.subplots()
ax.plot(phase, bolo, 'k')
ax.plot(phase, control_bolo, 'r')
sns.despine(ax=ax)
fig.savefig('bolo%s_%d_%d.pdf' % (fc.lc.meta['name'], i, 0))
"""
