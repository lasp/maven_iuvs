import glob
import numpy as np
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib import colorbar
from matplotlib.collections import PolyCollection
from .paths import idl_cmap_directory
import warnings


def getcmap(no,reverse=False,vmin=0,vmax=1):
    if idl_cmap_directory == '':
        warnings.warn('No IDL Colorbars directory defined, using Magma')
        cm = mpl.cm.magma()
    else:
        fnames = glob.glob(idl_cmap_directory+str(no).zfill(3)+'*')
        data = np.loadtxt(fnames[0],delimiter=',')
        if reverse:
            data=np.flip(data,axis=0)
        datalength=len(data)
        dmin=int(datalength*vmin)
        dmax=int(datalength*vmax)
        if dmax==datalength:
            dmax=datalength-1
        if dmin==datalength:
            dmin=datalength-1
        data=data[dmin:dmax+1]
        
        cm = LinearSegmentedColormap.from_list('my_cmap',data)
    return cm


def detector_image(fits,integration=0,
                   fig=None,ax=None,
                   norm=None, cmap=109,
                   scale="linear",
                   arange=None,
                   prange=None):
    new_ax=False
    if ax==None:    
        new_ax=True
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1, 1, 1)
    if type(cmap) != int:
        raise ValueError
    else:
        cmap = getcmap(cmap)
        
    ax.set_xlim([0, 1024])
    ax.set_ylim([0, 1024])

    #get the data
    data=fits['detector_dark_subtracted'].data[integration]
    
    #figure out the binning
    spapixlo=fits['Binning'].data['SPAPIXLO'][0]
    spapixhi=fits['Binning'].data['SPAPIXHI'][0]
    spepixlo=fits['Binning'].data['SPEPIXLO'][0]
    spepixhi=fits['Binning'].data['SPEPIXHI'][0]
    if not (set((spapixhi[:-1]+1)-spapixlo[1:])=={0} and set((spepixhi[:-1]+1)-spepixlo[1:])=={0}):
        raise ValueError
    
    spepixrange=np.concatenate([[spepixlo[0]],spepixhi+1])
    spapixrange=np.concatenate([[spapixlo[0]],spapixhi+1])
    
    spepixwidth=spepixrange[1:]-spepixrange[:-1]
    spapixwidth=spapixrange[1:]-spapixrange[:-1]
    
    npixperbin=np.outer(spapixwidth,spepixwidth)
    
    data=data/npixperbin
    
    #figure out what norm to use
    if norm == None:
        if prange == None:
            prange = [0, 100]
        if arange == None:
            arange = [np.percentile(data, prange[0]),
                      np.percentile(data, prange[1])]
        if scale == "linear":
            norm = mpl.colors.Normalize(vmin=arange[0],
                                               vmax=arange[1])
        elif scale == "sqrt":
            norm = mpl.colors.PowerNorm(gamma=0.5,
                                               vmin=arange[0],
                                               vmax=arange[1])
        elif scale == "log":
            norm = mpl.colors.LogNorm(vmin=arange[0],
                                             vmax=arange[1])
        else:
            raise ValueError
    
    ax.patch.set_color('#666666')
    ax.patch.set_alpha(1.0)
    pcm = ax.pcolormesh(spepixrange, spapixrange,
                        data,
                        norm=norm,
                        cmap=cmap)

    #add the colorbar axes
    ax_pos = ax.get_position()
    cax_width_frac = 0.07
    cax_margin = 0.02
    cax = fig.add_axes((ax_pos.x1+cax_margin*ax_pos.width,ax_pos.y0,cax_width_frac*ax_pos.width,ax_pos.height))
    fig.colorbar(pcm, cax=cax)
    if scale=="linear":
        cax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    
    if new_ax:
        #fig.show()
        return fig
    else:
        return


