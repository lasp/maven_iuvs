from .plot_defaults import *
import numpy as np
import matplotlib

def setup_colorbar(fig,outcorona_axis,colorscale=109,brightness_range=[0,20],ticks=None,plot_diff=False):
    figsize       = fig.get_size_inches()
    figure_width  = figsize[0]
    figure_height = figsize[1]
    
    outcorona_bbox = outcorona_axis.get_position()
    outcorona_frac=0.6
    colorbar_axis = fig.add_axes((outcorona_bbox.x1+0.35*outcorona_bbox.width,
                                  outcorona_bbox.y0+outcorona_bbox.height*(1-outcorona_frac)/2,
                                  0.35*outcorona_bbox.width,
                                  outcorona_bbox.height*outcorona_frac))
    style_axis(colorbar_axis)
    colorbar_axis.yaxis.tick_right()
    colorbar_axis.yaxis.set_label_position("right")
    colorbar_axis.set_ylim(brightness_range)
    if ticks==None:
        colorbar_axis.yaxis.set_major_locator(matplotlib.ticker.LinearLocator(numticks=5))

    if not plot_diff:
        labeltext = 'Brightness [kR]'
    else:
        labeltext = 'Model - Data [kR]'
        
    colorbar_axis.set_ylabel(labeltext,size=fontsize,labelpad=tickpad+2)
    
    from ..graphics import getcmap
    if plot_diff and colorscale==109:
        colorscale = 67
    
    colormap=getcmap(colorscale)
    norm=matplotlib.colors.Normalize(vmin=brightness_range[0],vmax=brightness_range[1])
    
    bright_vals=np.linspace(brightness_range[0],brightness_range[1],100)[::-1]
    colorbar_axis.imshow(np.transpose([bright_vals,bright_vals]),extent=(0,1,0,1),transform=colorbar_axis.transAxes,cmap=colormap,norm=norm,aspect='auto')
    
    return colorbar_axis, colormap, norm
