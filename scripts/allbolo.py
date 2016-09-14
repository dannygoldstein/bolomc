import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from book import plot_all_bolometric_light_curves
from samples import models
import glob

plt.rcParams['font.family'] = 'monospace'
outfiles = glob.glob('run/*.out')
all_models = map(models, outfiles)
all_models = [tup[2] for tup in all_models]
fig = plot_all_bolometric_light_curves(all_models)
fig.tight_layout()
fig.savefig('allbolo.pdf')

