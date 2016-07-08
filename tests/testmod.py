import matplotlib 
matplotlib.use('Agg')
from copy import copy
import sncosmo
import numpy as np
from bolomc import bump
from bolomc import burns
from bolomc.distributions import TruncNorm

lc = sncosmo.read_lc('../data/CSP_Photometry_DR2/SN2005elopt+nir_photo.dat', format='csp')

model = sncosmo.Model(bump.BumpSource(),
                      effect_names=['host','mw'],
                      effect_frames=['rest','obs'],
                      effects=[sncosmo.OD94Dust(), sncosmo.F99Dust()])

model2 = copy(model)
model2.set(UV_bump_amp=1.,#blue_bump_amp=0.2,
           blue_bump_amp=-0.2, 
           i1_bump_amp=0.1,
           i2_bump_amp=-0.2,
           y1_bump_amp=-0.2, 
           y2_bump_amp=0.2,
           y3_bump_amp=-0.1,
           j1_bump_amp=-0.2, 
           j2_bump_amp=0.2,
           h1_bump_amp=-0.2, 
           h2_bump_amp=0.2,
           k1_bump_amp=-0.2, 
           k2_bump_amp=0.2)

bump_map = ['blue',
            'blue',
            'blue',
            'blue',
            'blue',
            'i',
            'y',
            'j',
            'h',
            'k']


fig = sncosmo.plot_lc(model=[model, model2], bands=np.unique(lc['filter']))

for i,ax in enumerate(fig.axes):
    try:
        line1, line2 = ax.lines[:2]
        x = line1.get_xdata()
        y2 = line2.get_ydata()
        y = line1.get_ydata()
        ax.plot(x, y2-y, color=line1.get_color(),
                ls=':')
        name = bump_map[i]
        bumps = filter(lambda bump: name in bump.name, model.source.bumps)
        for color, bump in zip(['k','g','r'], bumps):
            mu = bump._gaussian.mu
            sigma = bump._gaussian.sigma
            for x in [mu - sigma, mu, mu + sigma]:
                ax.axvline(x, color=color, linestyle=':')
    except Exception as e:
        print e
        pass
    
        
fig.savefig('newbump.pdf')
        
        
