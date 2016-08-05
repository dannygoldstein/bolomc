import pickle
import sncosmo
import numpy as np
from copy import copy
from bolomc import burns, bump, bolo

__whatami__ = "Post-process bolometric LC constructions."
__author__ = "Danny Goldstein <dgold@berkeley.edu>"

def models(sample_file, dust_type, thin=1, nkeep=200):
    data = pickle.load(open(sample_file, 'rb'))
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
    return mods
