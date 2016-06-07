import h5py
import matplotlib
matplotlib.use("Agg")
import pickle
from bolomc import plotting
import numpy as np

import sys

name = sys.argv[1]

fc = pickle.load(open(name + '.fc.pkl','rb'))

p = fc.xstar_p
l = fc.xstar_l

f = h5py.File(name + '.h5')

W = f['burn']['params'][1, 0][2:].reshape(fc.nph, fc.nl)

fig = plotting.plot_wsurf(p, l, W)

fig.savefig(name+'.surf.pdf')
