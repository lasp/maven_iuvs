from .plot_defaults import *
import numpy as np
from .orbit_annotations import draw_obs_arrow
from .pixel_swath_quantities import pixel_swath_quantities
from .populate_limb_plot import draw_integration_plot_alt_labels

def quicklook_inoutdisk(orbno=None, observations=None, orbit_ax=None, orbit_coords=None, map_ax=None, to_iau_mat=None, colormap=None, cmapnorm=None, outdisk_axis=None, indisk_axis=None):
    outdisk_obs = [obs for obs in observations if obs['obsid']=='outdisk' and obs['segment']=='outbound' and not obs['echelle']]
    indisk_obs = [obs for obs in observations if obs['obsid']=='indisk' and obs['segment']=='inbound' and not obs['echelle']]
        
    #remove the axes we don't need based on orbit number
    #indisk observations come in chunks:
    if not ((orbno==3190) or (3440<orbno and orbno<3480) or (6175<orbno and orbno<6900)):
        outdisk_axis.set_ylabel(indisk_axis.get_ylabel(),size=fontsize,labelpad=tickpad)
        indisk_axis.set_axis_off()
    
    #outdisk observations occur until orbit ~8400
    if orbno>8050:
        outdisk_axis.set_axis_off()
    
    #place the data on the axes
    if len(outdisk_obs)>0:
        #draw labels on orbit, and geometry of observations
        outdisk_brightness=[]
        outdisk_alts=[]
        for idx, obs in enumerate(outdisk_obs):
            if obs['filename']!='':
                draw_obs_arrow(obs['fits'], orbit_coords['camera_right'], orbit_coords['camera_up'], orbit_ax, target_index=0.5, arrow_length=0.4)
                from ..integration import get_lya
                outdisk_brightness.append(get_lya(obs['filename']))
                
                mid_pixel_index = outdisk_brightness[-1].shape[1]//2
                slit_center_alt = obs['fits']['PixelGeometry'].data['PIXEL_CORNER_MRH_ALT'][:,mid_pixel_index,4] 
                outdisk_alts.append(slit_center_alt)
        if len(outdisk_brightness)>0:
            outdisk_brightness = np.concatenate(outdisk_brightness)
            outdisk_x=1-np.linspace(0,1,outdisk_brightness.shape[1]+1) # flip for consistency with peri/limb/corona
            outdisk_x=np.repeat(outdisk_x[np.newaxis,:],outdisk_brightness.shape[0]+1,axis=0)
            outdisk_y=np.linspace(0,1,outdisk_brightness.shape[0]+1)
            outdisk_y=np.repeat(outdisk_y[:,np.newaxis],outdisk_brightness.shape[1]+1,axis=1)
           
            outdisk_obsid_label = "OD"
            outdisk_axis.set_title(outdisk_obsid_label,fontsize=orbit_annotation_fontsize,c=orbit_annotation_color,pad=0)
            outdisk_axis.patch.set_alpha(1)
        
            outdisk_axis.autoscale(False)
            outdisk_axis.set_xlim([0,1])
            outdisk_axis.set_ylim([0,1])
            pcol = outdisk_axis.pcolormesh(outdisk_x,outdisk_y,outdisk_brightness,norm=cmapnorm,cmap=colormap,linewidth=pcolormesh_edge_width)
            pcol.set_edgecolor('face')

            #label the limb and max altitude if we cross the limb
            outdisk_alts=np.concatenate(outdisk_alts)
            mid_pixel_index = outdisk_brightness.shape[1]//2
            mid_pixel_x = np.mean(outdisk_x[mid_pixel_index:mid_pixel_index+2])
            draw_integration_plot_alt_labels(outdisk_alts, mid_pixel_x, outdisk_y, outdisk_axis, n_labels=0)

            #draw box on map axis
            #overlaps with orbit, confusing
            #[draw_map_obs(obs,map_ax,to_iau_mat) for obs in outdisk_obs]
        else:
            #no observation files found, change the plot label color
            outdisk_axis.set_title(outdisk_obsid_label,fontsize=orbit_annotation_fontsize,c=orbit_annotation_warning,pad=0)

    if len(indisk_obs)>0:
        indisk_brightness=[]
        indisk_alts=[]
        for idx, obs in enumerate(indisk_obs):
            if obs['filename']!='':
                draw_obs_arrow(obs['filename'], orbit_coords['camera_right'], orbit_coords['camera_up'], orbit_ax, target_index=0.5, arrow_length=0.4)
                from ..integration import get_lya
                indisk_brightness.append(get_lya(obs['filename']))
                
                mid_pixel_index = outdisk_brightness[-1].shape[1]//2
                slit_center_alt = obs['fits']['PixelGeometry'].data['PIXEL_CORNER_MRH_ALT'][:,mid_pixel_index,4] 
                outdisk_alts.append(slit_center_alt)
        if len(indisk_brightness)>0:
            indisk_brightness = np.concatenate(indisk_brightness)
            indisk_x=1-np.linspace(0,1,indisk_brightness.shape[1]+1) # flip for consistency with peri/limb/corona
            indisk_x=np.repeat(indisk_x[np.newaxis,:],indisk_brightness.shape[0]+1,axis=0)
            indisk_y=1-np.linspace(0,1,indisk_brightness.shape[0]+1) #flip so earlier integrations show up higher on the plot, consistent with observation type
            indisk_y=np.repeat(indisk_y[:,np.newaxis],indisk_brightness.shape[1]+1,axis=1)

            indisk_obsid_label = "ID"
            indisk_axis.set_title(indisk_obsid_label,fontsize=orbit_annotation_fontsize,c=orbit_annotation_color,pad=0)
            indisk_axis.patch.set_alpha(1)
            
            indisk_axis.autoscale(False)
            indisk_axis.set_xlim([0,1])
            indisk_axis.set_ylim([0,1])
            pcol = indisk_axis.pcolormesh(indisk_x,indisk_y,indisk_brightness,norm=cmapnorm,cmap=colormap,linewidth=pcolormesh_edge_width)
            pcol.set_edgecolor('face')
            
            #label the limb and max altitude if we cross the limb
            indisk_alts=np.concatenate(indisk_alts)
            mid_pixel_index = indisk_brightness.shape[1]//2
            mid_pixel_x = np.mean(indisk_x[mid_pixel_index:mid_pixel_index+2])
            draw_integration_plot_alt_labels(indisk_alts, mid_pixel_x, indisk_y, indisk_axis, n_labels=0)
            
            #draw box on map axis           
            #overlaps with orbit, confusing
            #[draw_map_obs(obs,map_ax,to_iau_mat) for obs in outdisk_obs]
        else:
            #no observation files found, change the plot label color
            indisk_axis.set_title(indisk_obsid_label,fontsize=orbit_annotation_fontsize,c=orbit_annotation_warning,pad=0)
    
    return (outdisk_axis, indisk_axis)
