import pickle
import sncosmo
import numpy as np
from copy import copy
from bolomc import burns, bump, bolo

__whatami__ = "Post-process bolometric LC constructions."
__author__ = "Danny Goldstein <dgold@berkeley.edu>"

def models(sample_file, thin=1, nkeep=200):
    lc, config, data = pickle.load(open(sample_file, 'rb'))
    try:
        dust_type = sncosmo.OD94Dust if config['dust_type'] == 'od94' \
                    else sncosmo.F99Dust
    except:
        dust_type = sncosmo.OD94Dust
    field_names = data.dtype.names
    data = data[::thin]
    data = np.random.choice(data, size=nkeep, replace=False)

    # create models
    mods = []
    model = bump.bump_model(dust_type)
    for row in data:
        d = dict(zip(field_names, row))
        cur_mod = copy(model)
        cur_mod.set(**d)
        mods.append(cur_mod)
    return (lc, config, mods)
