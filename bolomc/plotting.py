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
    
    warp = np.rollaxis(warp, 2)

    for p, row in zip(pgrid, warp):
        fig, ax = plt.subplots(figsize=(10.5,8))
        for realization in row:
            ax.plot(wgrid, realization, 'k-', alpha=0.2)
        ax.plot(wgrid, row.mean(0), 'r-', linewidth=1.3)
        ax.set_title('phase=%.5ed' % p)
        ax.set_xlabel('wavelength (AA)')
        ax.set_ylabel('ratio')
        figs.append(fig)
    return figs

        
        
    
        
