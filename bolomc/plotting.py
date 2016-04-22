#!/usr/bin/env python 

__author__ = 'Danny Goldstein <dgold@berkeley.edu>'
__whatami__ = 'Plotting tools for bolomc.'

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import numpy as np

def plot_wsurf(pgrid, wgrid, warp, vmin=0, vmax=2, lc=None):
    """Produce a heatmap of a spectral warping surface. pgrid is the 1D
    array of phase values, wgrid is the 1d array of wavelength values,
    and warp is the 2D array of warping function values evaluated on
    the grid.

    """

    fig, ax = plt.subplots(figsize=(5,10))
    m = cm.get_cmap('viridis')
    
    # Plot the surface. 
    res = ax.pcolorfast(wgrid, pgrid, warp.T, 
                        cmap=m, vmin=vmin, 
                        vmax=vmax)
    
    if lc is not None:
        ax.plot(lc['wave_eff'], lc['mjd'], 'k+')
    
    ax.invert_yaxis()
    ax.set_xlabel('wavelength (AA)')
    ax.set_ylabel('phase (days)')

    fig.colorbar(res)
    return fig


def plot_wslices(pgrid, wgrid, warp):
    """Plot slices of realized warping functions. pgrid is the 1D
    array of phase values, wgrid is the 1d array of wavelength values,
    and warp is the 2D array of warping function values evaluated on
    the grid."""
    
    warp = np.atleast_3d(warp)
    figs = []
    
    warp = np.rollaxis(warp, 1)

    for p, row in zip(pgrid, warp):
        fig, ax = plt.subplots(figsize=(10.5,8))
        medlc = np.median(row, axis=0)
        medm1 = np.percentile(row, 50 - 68./2, axis=0)
        medp1 = np.percentile(row, 50 + 68./2, axis=0)
        ax.plot(wgrid, medlc, color='r')
        ax.fill_between(wgrid, medm1, medp1, color='r', alpha=0.2)
        ax.set_title('phase=%.5ed' % p)
        ax.set_xlabel('wavelength (AA)')
        ax.set_ylabel('ratio')
        figs.append(fig)
    return figs

        
        
    
        
