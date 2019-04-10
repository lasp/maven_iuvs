import glob
import numpy as np
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar
from matplotlib.collections import PolyCollection
from iuvs.startup import idl_cmap_directory
import warnings

def getcmap(no):
    if idl_cmap_directory == '':
        warnings.warn('No IDL Colorbars directory defined, using Magma')
        cm = mpl.cm.magma()
    else:
        fnames = glob.glob(idl_cmap_directory+str(no).zfill(3)+'*')
        cm = LinearSegmentedColormap.from_list('my_cmap',
                                               np.loadtxt(fnames[0],
                                                          delimiter=','))
    return cm


def plotdetector(data,
                 xbins=[], ybins=[],
                 norm=[], cmap=109,
                 scale="linear",
                 arange=[],
                 prange=[]):
    if type(data) != np.ndarray:
        data = np.array(data)
    if len(xbins) == 0:
        xbins = np.arange(data.shape[0]+1)
    if len(ybins) == 0:
        ybins = np.arange(data.shape[1]+1)
    if norm == []:
        if prange == []:
            prange = [0, 100]
        if arange == []:
            arange = [np.percentile(data, prange[0]),
                      np.percentile(data, prange[1])]
        if scale == "linear":
            norm = colors.Normalize(vmin=arange[0],
                                    vmax=arange[1])
        elif scale == "sqrt":
            norm = colors.PowerNorm(gamma=0.5,
                                    vmin=arange[0],
                                    vmax=arange[1])
        elif scale == "log":
            norm = colors.LogNorm(vmin=arange[0],
                                  vmax=arange[1])
        else:
            raise ValueError
    if type(cmap) != int:
        raise ValueError
    else:
        cmap = getcmap(cmap)

    fig = plt.figure(figsize=(5, 5))

    ax = fig.add_subplot(1, 1, 1)
    ax.set_aspect(1.)
    ax.set_xlim([0, 1023])
    ax.set_ylim([0, 1023])

    xplot, yplot = np.meshgrid(xbins, ybins)

    pcm = ax.pcolormesh(xplot, yplot,
                        data,
                        norm=norm,
                        cmap=cmap)

    ax_divider = make_axes_locatable(ax)
    cax1 = ax_divider.append_axes("right", size="7%", pad="2%")
    cb1 = colorbar(pcm, cax=cax1)

    fig.show()

    return fig


def get_periapse_files(orbno):
    # ,dir=filedirectory)
    orbfilenames = getfiles('*periapse-orbit'+str(orbno).zfill(5)+'*fuv*')

    orbfiles = [fits.open(f) for f in orbfilenames]
    orbfiles = [f for f in orbfiles if len(f['Primary'].shape) == 3]
    orbfilenames = [f['Primary'].header['FILENAME']
                    for f in orbfiles if len(f['Primary'].shape) == 3]


def get_slit_corner_alts(myfits, swathnumber=0):
    fitalts = myfits['PixelGeometry'].data['PIXEL_CORNER_MRH_ALT']
    altpadfrac = 0.1  # pad the vertical extent of the pixels by 20%
    slitpadfrac = 0.0
    slitaltcorners = [
        [
            [
                [(1+slitpadfrac)*(1-(i_slitalt+1)/(1.0*len(integration))-0.5) +
                 swathnumber,
                 slitalt[2]+altpadfrac*(slitalt[2]-slitalt[3])],
                [(1+slitpadfrac)*(1-(i_slitalt+1)/(1.0*len(integration))-0.5) +
                 swathnumber,
                 slitalt[3]+altpadfrac*(slitalt[3]-slitalt[2])],
                [(1+slitpadfrac)*(1-(i_slitalt)/(1.0*len(integration))-0.5) +
                 swathnumber,
                 slitalt[1]+altpadfrac*(slitalt[1]-slitalt[0])],
                [(1+slitpadfrac)*(1-(i_slitalt)/(1.0*len(integration))-0.5) +
                 swathnumber,
                 slitalt[0]+altpadfrac*(slitalt[0]-slitalt[1])]
            ]
            for i_slitalt, slitalt in enumerate(integration)]
        for integration in fitalts]
    return np.array(slitaltcorners)


def periapse_swath_plot(lyafit, slitalts):
    fig, ax = plt.subplots(1, figsize=(20, 6))
    tickstyle = {'labelsize': 30}
    labelstyle = {'fontsize': 30, 'labelpad': 20}

    lyabright = [l[0] for l in lyafit]

    flatslitalts = np.concatenate(np.concatenate(slitalts))
    flatlyabright = np.concatenate(np.concatenate(lyabright))

    p = PolyCollection(flatslitalts,
                       array=flatlyabright)
    p.set_cmap(getcmap(109))
    p.set_norm(mpl.colors.Normalize(vmin=0, vmax=20.))
    p.set_edgecolor(None)
    ax.add_collection(p)

    ax.autoscale_view()
    ax.set_facecolor('black')
    ax.margins(0)
    ax.grid(0)
    ax.set_xticks(())
    ax.tick_params(**tickstyle)
    ax.set_ylabel('Altitude [km]', **labelstyle)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cax.tick_params(**tickstyle)

    cbar = plt.colorbar(p, cax=cax)
    cbar.set_label('Intensity [kR]', **labelstyle)

    plt.show()
