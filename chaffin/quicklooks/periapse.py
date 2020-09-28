from .plot_defaults import *
import numpy as np
from .orbit_annotations import draw_obs_arrow
from .draw_map_obs import draw_map_obs
import spiceypy as spice
import matplotlib
from .pixel_swath_quantities import pixel_swath_quantities
from .populate_limb_plot import populate_limb_plot

def quicklook_periapse(orbno=None, observations=None, orbit_coords=None, map_ax=None, to_iau_mat=None, colormap=None, cmapnorm=None, orbit_peri_ax=None, periapse_axes=None):
    peri_obs = [obs for obs in observations if obs['obsid']=='periapse' and obs['segment']=='periapse']
    n_peri = len(peri_obs)
    
    #orbits after ~9080 routinely have 14 periapse limb scans
    #orbits before this have 12
    if orbno < 9081:
        #delete the first and last axis, move labels to second axis
        periapse_axes[-1].remove()
        new_start_pos = periapse_axes[1].get_position()
        periapse_axes[1].remove()
        periapse_axes[0].set_position((new_start_pos.x0,new_start_pos.y0,new_start_pos.width,new_start_pos.height))
        periapse_axes=np.concatenate([[periapse_axes[0]],periapse_axes[2:-1]])
    
    n_peri_show=len(periapse_axes)
    
    #check for high voltage on, appropriate altitude range
    
    # if we have more than the expected number, draw nothing except a warning
    if n_peri > n_peri_show:
        periapse_axes[0].text(0,-0.02,'more than expected number of periapse files found, not showing any',
                              color=orbit_annotation_warning,size=orbit_annotation_fontsize,
                              transform=periapse_axes[0].transAxes,clip_on=False,va='top',ha='left')
        n_peri_show = 0
    
    #determine what direction we're looking so we know what side to put the labels on
    obs_with_file = [obs for obs in peri_obs if obs['filename']!='']
    peri_obs_direction = 'port'
    if len(obs_with_file)>0:
        scpos_peri_2d, lospos_peri_2d = draw_obs_arrow(obs_with_file[0]['fits'], orbit_coords['camera_right'], orbit_coords['camera_pos_norm'], orbit_peri_ax, get_geom=True)
        if lospos_peri_2d[1] < scpos_peri_2d[1]:
            peri_obs_direction = 'starbord'

    label_offset = 0.04 if peri_obs_direction=='starbord' else -0.04
    label_vertical_align = 'bottom' if peri_obs_direction=='starbord' else 'top'
    
    #draw labels and lines of sight on pariapse_orbit, show data
    #get reduced data
    all_pixel_x=[]
    all_pixel_y=[]
    all_brightness=[]
    for idx, obs in enumerate(peri_obs):
        obsid_label = obs['label']
        axis_idx = int(obsid_label.replace('P',''))-1
        
        #determine what label color to use
        file_exists=True
        orbit_label_color = '#666666'
        plot_label_color=orbit_annotation_color
        if obs['filebasename']=='':
            #there is no file matching this command, label it missing
            file_exists=False
            orbit_label_color = '#AA6666'
            plot_label_color=orbit_annotation_warning

        if idx < n_peri_show:
            periapse_axes[axis_idx].set_title(obsid_label,fontsize=orbit_annotation_fontsize,c=plot_label_color,pad=0)
            
        #label the orbit where the observation happened
        #get the spacecraft position for the integration times
        etlist=np.linspace(obs['et_start'],obs['et_end'],100)
        statelist=np.array(spice.spkezr('MAVEN',etlist,'MAVEN_MSO','NONE','Mars')[0])
        scpos3d=statelist[:,0:3]
        scvel3d=statelist[:,3:6]
        scpos_peri_2d=np.array([[np.dot(v,orbit_coords['camera_right'])/3395,np.dot(v,orbit_coords['camera_pos_norm'])/3395] for v in scpos3d])
        scpos_peri_2d=np.mean(scpos_peri_2d,axis=0)
        text_label = orbit_peri_ax.text(scpos_peri_2d[0],scpos_peri_2d[1]+label_offset,obsid_label,fontsize=orbit_annotation_fontsize,c=orbit_label_color,va=label_vertical_align,ha='center',rotation=270)
        text_label.set_path_effects([matplotlib.patheffects.withStroke(linewidth=0.35, foreground='k')])
            
        #plot observation if it happened
        if file_exists and not obs['echelle']:
            scpos_peri_2d, lospos_peri_2d = draw_obs_arrow(obs['fits'], orbit_coords['camera_right'], orbit_coords['camera_pos_norm'], orbit_peri_ax, color=orbit_label_color)
            alt_info = orbit_peri_ax.text(0.9,0.025,'arrows point\nat closest alt\nto ~'+str(default_target_altitude)+' km',
                                          transform=orbit_peri_ax.transAxes,
                                          fontsize=orbit_annotation_fontsize,
                                          c=orbit_annotation_color,va='bottom',ha='center')
    
            #draw observation on map
            draw_map_obs(obs,map_ax,to_iau_mat)
            
            #figure out pixel corner coordinates and plot data
            if idx < n_peri_show:
                pixel_x,pixel_y,brightness=pixel_swath_quantities(obs['fits'])
                periapse_axes[axis_idx].patch.set_alpha(1)
                all_pixel_x.append(pixel_x)
                all_pixel_y.append(pixel_y)
                all_brightness.append(brightness)
        else:
            axes_text=''
            if obs['echelle']:
                axes_text = 'ECH'
            else:
                axes_text = 'no\nfile'
            periapse_axes[axis_idx].text(0.5,0.5,axes_text,fontsize=orbit_annotation_fontsize,c=plot_label_color,transform=periapse_axes[axis_idx].transAxes,ha='center',va='center')
    
    populate_limb_plot(peri_obs, periapse_axes, orbit_coords, all_pixel_x, all_pixel_y, all_brightness, colormap, cmapnorm)

    return periapse_axes
