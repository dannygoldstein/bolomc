import os
import sncosmo
import numpy as np
import seaborn as sns
from bolomc import model
from matplotlib import pyplot as plt

from numpy import testing

from util_test import *

LC = path('SN2005el')
fc = model.FitContext(LC)

fc2 = model.FitContext(path('SN2009F'))
fc3 = model.FitContext(path('SN2004gc'))

sns.set_style('ticks')

fig, ax = plt.subplots(figsize=(8,6))

outdir = 'output/testBolo'
os.mkdir(outdir)

def testBolo10():
    
    bolos = [fc.one_iteration() for i in range(10)]
    
    for b in bolos:
        ax.plot(fc.hsiao._phase, b, color='k')

    ax.set_xlabel('phase (days)')
    ax.set_ylabel('flux (cgs)')

    sns.despine()
    
    fig.savefig(os.path.join(outdir, 'testBolo10.pdf'))

def test2lcs():
    
    bolos2 = [fc2.one_iteration() for i in range(10)]
    
    for b in bolos2:
        ax.plot(fc.hsiao._phase, b, color='r')
        
    fig.savefig(os.path.join(outdir, 'test2lcs.pdf'))

def test3lcs():
    
    bolos3 = [fc3.one_iteration() for i in range(10)]
    
    for b in bolos3:
        ax.plot(fc.hsiao._phase, b, color='b')
        
    fig.savefig(os.path.join(outdir, 'test3lcs.pdf'))
