import h5py
import matplotlib
matplotlib.use("Agg")
import pickle
from bolomc import plotting
import numpy as np

fc = pickle.load(open('sinefc.pkl2','rb'))

p = fc.xstar_p
l = fc.xstar_l

f = h5py.File('sine.h52')

W = f['burn']['params'][1, 0][2:].reshape(fc.nph, fc.nl)

fig = plotting.plot_wsurf(p, l, W)

fig.savefig('surf.pdf')
