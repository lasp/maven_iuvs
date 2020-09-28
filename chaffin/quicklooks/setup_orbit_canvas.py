from .plot_defaults import *
import numpy as np

def setup_orbit_canvas(fig, orbtimedict, panel_x_start_frac=2.5/6):
    #set up canvas
    figsize       = fig.get_size_inches()
    figure_width  = figsize[0]
    figure_height = figsize[1]
    panel_aspect_ratio = 5/3.5 #legacy
    panel_height  = figure_height
    panel_width   = panel_height/panel_aspect_ratio
    
    panel_width_frac = panel_width / figure_width
    
    #get the orbit images for the middle of the orbit and periapsis
    from ..graphics import maven_orbit_image
    orbimg_arr, orbit_coords = maven_orbit_image(orbtimedict['orbit_middle_utc'],show=False,view_from_orbit_normal=True)
    from PIL import Image
    orbimg = Image.fromarray(orbimg_arr[0:695,190:-190])
    periorbimg_arr = maven_orbit_image(orbtimedict['peri_middle_utc'],view_from_periapsis=True,show=False)[0]
    
    #figure out where to put the axes on the figure so the orbit shows up in the right place
    periapse_distance = 1+500/3395 # not actual peri/apoapsis, nudged a bit to get the plot to aligned OK
    apoapse_distance = 3.
    maven_semimajor_axis = (periapse_distance + apoapse_distance) / 2
    maven_semiminor_axis = np.sqrt(periapse_distance*apoapse_distance)
    
    orbxpx,orbypx = orbimg.size
    orbxratio = 0.4*panel_width_frac #fraction of the panel width occupied by the orbit
    orbit_ax_width = orbxratio/(2*maven_semiminor_axis)*2*orbit_coords['extent']
    orbit_ax_height = figure_width/figure_height*orbit_ax_width
    orbit_ax_x_start = (panel_width_frac-orbit_ax_width)/2 + panel_x_start_frac
    periapse_y = 0.425
    orbit_ax_y_start = periapse_y-orbit_ax_height*(orbit_coords['extent']-periapse_distance)/(2*orbit_coords['extent'])
    
    orbit_ax = fig.add_axes((orbit_ax_x_start,orbit_ax_y_start,orbit_ax_width,orbit_ax_height))
    style_axis(orbit_ax)
    orbit_ax.set_axis_off()
    #set axes limits to match orbit image
    orbit_ax.set_ylim([-orbit_coords['extent'],orbit_coords['extent']])
    orbit_ax.set_xlim([-orbit_coords['extent'],orbit_coords['extent']])
    orbit_ax.imshow(orbimg_arr,extent=(0,1,0,1),transform=orbit_ax.transAxes,aspect='auto')
    
    peri_orbit_width_rMars = 2
    peri_orbit_width = orbit_ax_width*peri_orbit_width_rMars/(orbit_coords['extent'])
    peri_orbit_height_rMars = 0.5
    peri_orbit_height = figure_width/figure_height * peri_orbit_height_rMars/peri_orbit_width_rMars * peri_orbit_width
    peri_orbit_x_offset = orbit_ax_width*(orbit_coords['extent']-peri_orbit_width_rMars)/orbit_coords['extent']/2
    
    peri_orbit_y_offset = -0.18
    orbit_peri_ax = fig.add_axes((orbit_ax_x_start+peri_orbit_x_offset,
                                  periapse_y-peri_orbit_height+peri_orbit_y_offset,
                                  peri_orbit_width,
                                  peri_orbit_height))
    
    style_axis(orbit_peri_ax,color='#444444')
    orbit_peri_ax.set_xlim([-peri_orbit_width_rMars,peri_orbit_width_rMars])
    orbit_peri_ax.set_ylim([-peri_orbit_height/orbit_ax_height*orbit_coords['extent'],peri_orbit_height/orbit_ax_height*orbit_coords['extent']])
    orbit_peri_ax.imshow(periorbimg_arr,extent=(-orbit_coords['extent'],orbit_coords['extent'],-orbit_coords['extent'],orbit_coords['extent']),transform=orbit_peri_ax.transData,aspect='auto')
    #orbit_peri_ax.set_axis_off()
    
    #approximate bounding box of stuff drawn on orbit plot
    orbit_bbox = (orbit_ax_x_start+(orbit_coords['extent']-maven_semiminor_axis)/(2*orbit_coords['extent'])*orbit_ax_width,
                  orbit_ax_x_start+(orbit_coords['extent']+maven_semiminor_axis)/(2*orbit_coords['extent'])*orbit_ax_width,
                  periapse_y-peri_orbit_height+peri_orbit_y_offset,
                  orbit_ax.get_position().y1)
    
    #return the figure and the bounding box of the orbit
    return orbit_ax, orbit_bbox, orbit_coords, orbit_peri_ax
