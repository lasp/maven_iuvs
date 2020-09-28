from .plot_defaults import *
import numpy as np
from .orbit_annotations import draw_obs_arrow
from .pixel_swath_quantities import pixel_swath_quantities
from .populate_limb_plot import populate_limb_plot

def quicklook_outlimb(orbno=None, observations=None, orbit_ax=None, orbit_coords=None, map_ax=None, to_iau_mat=None, colormap=None, cmapnorm=None, outlimb_axes=None, fig=None):
    outlimb_obs = [obs for obs in observations if obs['obsid']=='outlimb' and obs['segment']=='outbound' and not obs['echelle']]
    n_outlimb = len(outlimb_obs)
    
    #logic to handle how many we have based on orbit number
    if (orbno < 3100):
        #only one outlimb on these orbits
        new_start_pos = outlimb_axes[3].get_position()
        outlimb_axes[0].set_position((new_start_pos.x0,new_start_pos.y0,new_start_pos.width,new_start_pos.height))
        [ax.remove() for ax in outlimb_axes[1:]]
        outlimb_axes=[outlimb_axes[0]]
    if (3100<=orbno and orbno <3477):
        #only two outlimb were commanded on these orbits
        new_start_pos = outlimb_axes[2].get_position()
        outlimb_axes[0].set_position((new_start_pos.x0,new_start_pos.y0,new_start_pos.width,new_start_pos.height))
        outlimb_axes[1].remove()
        outlimb_axes[2].remove()
        outlimb_axes[4].remove()
        outlimb_axes[5].remove()
        outlimb_axes=[outlimb_axes[0],outlimb_axes[3]]
    if (3477<=orbno and orbno<4725):
        #four outlimbs on these orbits
        outlimb_axes[-1].remove()
        outlimb_axes[-2].remove()
        outlimb_axes=outlimb_axes[:-2]
    if ((9245<=orbno and orbno<9265) or (9565<=orbno and orbno<9700)):
        #some aerobraking orbits have 9 outlimb observations, let's add some extra frames to deal with these
        
        #get the geometry of the existing axes
        outlimb_box_left     = outlimb_axes[ 0].get_position().x0
        outlimb_box_right    = outlimb_axes[-1].get_position().x1
        outlimb_box_bottom   = outlimb_axes[ 0].get_position().y0
        outlimb_box_height   = outlimb_axes[ 0].get_position().height
        outlimb_axes_padding = outlimb_axes[ 1].get_position().x0 - outlimb_axes[ 0].get_position().x1
        
        #get rid of the existing axes
        [ax.remove() for ax in outlimb_axes]
        
        #make 9 new axes that fit into the same space
        n_axes = 9
        outlimb_axis_width = (outlimb_box_right-outlimb_box_left-(n_axes-1)*outlimb_axes_padding)/n_axes
        outlimb_axis_space = outlimb_axis_width+outlimb_axes_padding
        
        outlimb_axes=[]
        for idx in range(9):
            outlimb_x_start = outlimb_box_left+idx*outlimb_axis_space
            ax = fig.add_axes((outlimb_x_start,outlimb_box_bottom,outlimb_axis_width,outlimb_box_height))
            style_axis(ax)
            outlimb_axes.append(ax)
    
    #all other orbits are supposed to have six outlimb observations, we don't have to change anything
    
    n_axes = len(outlimb_axes)
    
    if (n_outlimb>n_axes):
        #we have more observations than axes! print a warning:
        outlimb_axes[0].text(0,-0.02,'more outlimb obs than axes --- showing only first '+str(n_axes),
                            ha='left',va='top',transform=outlimb_axes[0].transAxes,
                            color=orbit_annotation_warning,size=orbit_annotation_fontsize,clip_on=False)
    
    #draw labels on orbit, and geometry of observations
    #get reduced data
    all_pixel_x=[]
    all_pixel_y=[]
    all_brightness=[]
    for idx, obs in enumerate(outlimb_obs):
        obsid_label = obs['label']
        axis_idx = int(obsid_label.replace('OL',''))-1
        
        outlimb_axes[axis_idx].set_title(obsid_label,
                                         fontsize=orbit_annotation_fontsize,
                                         c=(orbit_annotation_color if obs['filename'] != '' else orbit_annotation_warning),
                                         pad=0)
        
        if obs['filename'] !="":
            #draw arrow on orbit
            draw_obs_arrow(obs['fits'], orbit_coords['camera_right'], orbit_coords['camera_up'], orbit_ax)
            #draw box on map axis
            #overlaps with orbit, confusing
            #draw_map_obs(obs,map_ax,to_iau_mat)
            
            if axis_idx < n_axes:
                #figure out pixel corner coordinates and plot data
                outlimb_axes[axis_idx].patch.set_alpha(1)
                pixel_x,pixel_y,brightness=pixel_swath_quantities(obs['fits'])
                all_pixel_x.append(pixel_x)
                all_pixel_y.append(pixel_y)
                all_brightness.append(brightness)
                
    
    populate_limb_plot(outlimb_obs, outlimb_axes, orbit_coords, all_pixel_x, all_pixel_y, all_brightness, colormap, cmapnorm)
    
    return outlimb_axes

def quicklook_inlimb(orbno=None, observations=None, orbit_ax=None, orbit_coords=None, map_ax=None, to_iau_mat=None, colormap=None, cmapnorm=None, inlimb_axes=None, fig=None):
    inlimb_obs = [obs for obs in observations if obs['obsid']=='inlimb' and obs['segment']=='inbound' and not obs['echelle']]
    n_inlimb = len(inlimb_obs)
    
    #logic to handle how many we have based on orbit number
    if (orbno < 3442 and orbno!=3190):
        #inlimb data were not taken until orbit !3200
        [ax.remove() for ax in inlimb_axes]
        return
    if (orbno == 3190 or (3442<=orbno and orbno <3477)):
        #only two inlimb were commanded on these orbits
        new_start_pos = inlimb_axes[-2].get_position()
        inlimb_axes[-1].set_position((new_start_pos.x0,new_start_pos.y0,new_start_pos.width,new_start_pos.height))
        inlimb_axes[-2].remove()
        inlimb_axes[-4].remove()
        inlimb_axes[-5].remove()
        inlimb_axes[-6].remove()
        inlimb_axes=[inlimb_axes[-3],inlimb_axes[-1]]
    if (3477<=orbno and orbno<4725):
        #four inlimbs on these orbits
        inlimb_axes[0].remove()
        inlimb_axes[1].remove()
        inlimb_axes=inlimb_axes[2:] 
    if (9565<=orbno and orbno<9658):
        #some aerobraking orbits have 9 inlimb observations, let's add some extra frames to deal with these
        
        #get the geometry of the existing axes
        inlimb_box_left     = inlimb_axes[-1].get_position().x0
        inlimb_box_right    = inlimb_axes[ 0].get_position().x1
        inlimb_box_bottom   = inlimb_axes[ 0].get_position().y0
        inlimb_box_height   = inlimb_axes[ 0].get_position().height
        inlimb_axes_padding = inlimb_axes[ 0].get_position().x0 - inlimb_axes[ 1].get_position().x1
        
        #get rid of the existing axes
        [ax.remove() for ax in inlimb_axes]
        
        #make 9 new axes that fit into the same space
        n_axes = 9
        inlimb_axis_width = (inlimb_box_right-inlimb_box_left-(n_axes-1)*inlimb_axes_padding)/n_axes
        inlimb_axis_space = inlimb_axis_width+inlimb_axes_padding
        
        inlimb_axes=[]
        for idx in range(9):
            inlimb_x_start = inlimb_box_right-idx*inlimb_axis_space-inlimb_axis_width
            ax = fig.add_axes((inlimb_x_start,inlimb_box_bottom,inlimb_axis_width,inlimb_box_height))
            style_axis(ax)
            inlimb_axes.append(ax)

    #all other orbits are supposed to have six inlimb observations, we don't have to change anything
    n_axes = len(inlimb_axes)
    
    if n_inlimb > n_axes:
        #we have more observations than axes! print a warning:
        inlimb_axes[-1].text(0,-0.02,'more inlimb obs than axes --- showing only first '+str(n_axes),
                             ha='left',va='top',transform=inlimb_axes[-1].transAxes,
                             color=orbit_annotation_warning,size=orbit_annotation_fontsize,clip_on=False)
        n_inlimb_show = 6
    
    #draw labels on orbit, draw geometry of observations
    #get reduced data
    all_pixel_x=[]
    all_pixel_y=[]
    all_brightness=[]
    for idx, obs in enumerate(inlimb_obs):
        obsid_label = obs['label']
        axis_idx = int(obsid_label.replace('IL',''))-1
        
        inlimb_axes[axis_idx].set_title(obsid_label,
                                        fontsize=orbit_annotation_fontsize,
                                        c=(orbit_annotation_color if obs['filename'] != '' else orbit_annotation_warning),
                                        pad=0)

        if obs['filename'] !="":
            #draw arrow on orbit
            draw_obs_arrow(obs['fits'], orbit_coords['camera_right'], orbit_coords['camera_up'], orbit_ax)
            #draw box on map axis
            #overlaps with orbit, confusing
            #draw_map_obs(obs,map_ax,to_iau_mat)

            if axis_idx < n_axes:
                inlimb_axes[axis_idx].patch.set_alpha(1)
                #get the pixel coordinates to figure out if we should plot in altitude or integration number
                pixel_x,pixel_y,brightness=pixel_swath_quantities(obs['fits'])
                all_pixel_x.append(pixel_x)
                all_pixel_y.append(pixel_y)
                all_brightness.append(brightness)

    populate_limb_plot(inlimb_obs, inlimb_axes, orbit_coords, all_pixel_x, all_pixel_y, all_brightness, colormap, cmapnorm)

    return inlimb_axes
