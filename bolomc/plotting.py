#!/usr/bin/env python 

__author__ = 'Danny Goldstein <dgold@berkeley.edu>'
__whatami__ = 'Plotting tools for bolomc.'

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
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

def iqr(x, axis=None):
    """Compute the interquartile range of x."""
    q75, q25 = np.percentile(x, [75 ,25], axis=axis)
    return q75 - q25

def nbins(x):
    """Compute the number of bins for histogramming x using the freedman
    diaconis rule-of-thumb.
    """
    
    # Compute the bandwith.
    h = 2 * iqr(x) * x.size**(-1/3.)
    
    # Convert the bandwith into a number of bins. 
    return int((x.max() - x.min()) / h)

def center(bins):
    """Take bin edges and return the bin centers."""
    return (bins[:-1] + bins[1:]) / 2.

def plot_chains(chains, param_names=None, filename=None):
    """Plot the paths of MCMC chains in parameter space. Chains should
    have shape npar, nwal, nt."""
    
    s = chains.shape
    x = np.arange(1, s[-1] + 1)
    
    # Set up the figure.
    fig = plt.figure(figsize=(10, 7.5))

    # Lay down a grid structure in the figure. 
    g = GridSpec(1, 4)
    
    # Chain axis.
    ca = fig.add_subplot(g[:3])

    # Marginal axis (shares a y-axis with the chain plot).
    ma = fig.add_subplot(g[-1], sharey=ca)
    
    # If the figure is to be saved, initialize the backend. 
    
    if filename is not None:
        pdf = PdfPages(filename)
    figs = []
    
    for k, p in enumerate(chains):
        # Create this figure. 
        fig, (ca, ma) = plt.subplots(ncols=2, figsize=(10.5, 8)) 
        
        # Determine the bins for the marginal histogram. 
        bins = np.linspace(p.min(), p.max(), nbins(p) + 1)
        
        # For each parameter, 
        for i, w in enumerate(p):
            
            # plot the chains of the walkers on top of each other. 
            ca.plot(x, w, 'k', lw=1)
        
        # Compute the marginal histogram...
        n, bins = np.histogram(p, bins=bins)
        
        # ...and plot it. 
        bin_centers = center(bins)
        ma.plot(n, bin_centers, 'k') # x and y are switched 

        # Label the plot. 
        ca.set_xlabel('iteration')
        
        # The plot gets a y label if parameter names have been
        # specified.
        if param_names is not None:
            ca.set_ylabel(param_names[k])

        # Save. 
        if filename is not None:
            pdf.savefig(fig)
        figs.append(fig)
    
    # Clean up. 
    if filename is not None:
        pdf.close()
    return figs
