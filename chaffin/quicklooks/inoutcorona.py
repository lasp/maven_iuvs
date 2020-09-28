from .plot_defaults import *
import numpy as np
from .orbit_annotations import draw_obs_arrow
from .pixel_swath_quantities import pixel_swath_quantities
from ..geometry import get_pixel_vec_mso

def quicklook_outcorona(orbno=None, observations=None, orbit_ax=None, orbit_coords=None, map_ax=None, to_iau_mat=None, colormap=None, cmapnorm=None, outcorona_axis=None):
    #some files are incorrectly labeled, so we need to check if these are really outcorona files
    outcorona_obs = [obs for obs in observations if (obs['obsid']=='outcorona' or obs['obsid']=='outspace') and obs['segment']=='outbound' and not obs['echelle'] and obs['fits']!=None]
    
    #select only files that look across the orbit and are ~inertially pointed
    pixel_vecs=[get_pixel_vec_mso(obs['fits']) for obs in outcorona_obs]
    pixel_vecs=[v[:,v.shape[1]//2,4] for v in pixel_vecs]
    pixel_vec_mean=[np.mean(v,axis=0) for v in pixel_vecs]
    pixel_vec_dev=[np.mean(np.dot(v,m)) for v,m in zip(pixel_vecs,pixel_vec_mean)]
    outcorona_alignment=[1-np.abs(np.dot(v,orbit_coords['camera_right'])) for v in pixel_vec_mean]
    outcorona_obs=[obs for a,d,obs in zip(outcorona_alignment,pixel_vec_dev,outcorona_obs) if (a<0.05 and (1-d)<0.01)]
    #do something with failed cases?
    
    obsid_label = outcorona_obs[0]['label'] if len(outcorona_obs)>0 else ''
    obsid_label_color = orbit_annotation_color
    if len(outcorona_obs)>0:
        if outcorona_obs[0]['filename'] == '':
            obsid_label_color = orbit_annotation_warning
    
    outcorona_axis.set_title(obsid_label,
                             fontsize=orbit_annotation_fontsize,
                             c=obsid_label_color,
                             pad=0)
    
    #draw labels on orbit, and geometry of observations
    for idx, obs in enumerate(outcorona_obs):
        if obs['filename'] != '':
            #draw arrow on orbit
            arrow_location = 0.5 if obs['obsid']=='outcorona' else 0.8
            draw_obs_arrow(obs['fits'], orbit_coords['camera_right'], orbit_coords['camera_up'], orbit_ax, target_index=arrow_location, arrow_length=0.4)   
            #draw box on map axis
            #overlaps with orbit, confusing
            #draw_map_obs(obs,map_ax,to_iau_mat)
            
            
            #figure out pixel corner coordinates and plot data
            outcorona_axis.patch.set_alpha(1)
            pixel_x,pixel_y,brightness=pixel_swath_quantities(obs['fits'])
            pcol = outcorona_axis.pcolormesh(pixel_x,pixel_y,brightness,norm=cmapnorm,cmap=colormap,linewidth=pcolormesh_edge_width)
            pcol.set_edgecolor('face')
        
    return outcorona_axis

def quicklook_incorona(orbno=None, observations=None, orbit_ax=None, orbit_coords=None, map_ax=None, to_iau_mat=None, colormap=None, cmapnorm=None, incorona_axis=None):
    incorona_obs = [obs for obs in observations if (obs['obsid']=='incorona' or obs['obsid']=='inspace') and obs['segment']=='inbound' and not obs['echelle'] and obs['fits']!=None]
    
    #select only files that look mostly across the orbit
    #select only files that look across the orbit and are ~inertially pointed
    pixel_vecs=[get_pixel_vec_mso(obs['fits']) for obs in incorona_obs]
    pixel_vecs=[v[:,v.shape[1]//2,4] for v in pixel_vecs]
    pixel_vec_mean=[np.mean(v,axis=0) for v in pixel_vecs]
    pixel_vec_dev=[np.mean(np.dot(v,m)) for v,m in zip(pixel_vecs,pixel_vec_mean)]
    incorona_alignment=[1-np.abs(np.dot(v,orbit_coords['camera_right'])) for v in pixel_vec_mean]
    incorona_obs=[obs for a,d,obs in zip(incorona_alignment,pixel_vec_dev,incorona_obs) if (a<0.05 and (1-d)<0.01)]
    #do something with failed cases?
    
    obsid_label = incorona_obs[0]['label'] if len(incorona_obs)>0 else ''
    obsid_label_color = orbit_annotation_color
    if len(incorona_obs)>0:
        if incorona_obs[0]['filename'] == '':
            obsid_label_color = orbit_annotation_warning
    
    incorona_axis.set_title(obsid_label,
                            fontsize=orbit_annotation_fontsize,
                            c=obsid_label_color,
                            pad=0)
    
    #draw labels on orbit, and geometry of observations
    for idx, obs in enumerate(incorona_obs):   
        if obs['filename'] != '':
            #draw arrow on orbit
            arrow_location = 0.5 if obs['obsid']=='incorona' else 0.2
            draw_obs_arrow(obs['fits'], orbit_coords['camera_right'], orbit_coords['camera_up'], orbit_ax, target_index=arrow_location, arrow_length=0.4)   
            #draw box on map axis
            #overlaps with orbit, confusing
            #draw_map_obs(obs,map_ax,to_iau_mat)

            #figure out pixel corner coordinates and plot data
            incorona_axis.patch.set_alpha(1)
            pixel_x,pixel_y,brightness=pixel_swath_quantities(obs['fits'])
            pcol = incorona_axis.pcolormesh(pixel_x,pixel_y,brightness,norm=cmapnorm,cmap=colormap,linewidth=pcolormesh_edge_width)
            pcol.set_edgecolor('face')
        
    return incorona_axis
