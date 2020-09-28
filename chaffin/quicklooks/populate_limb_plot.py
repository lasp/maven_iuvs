from .plot_defaults import *
import numpy as np
import matplotlib

def draw_integration_plot_alt_labels(slit_center_alt, mid_pixel_x, pixel_y, axis, n_labels=3, label_limb=True):
        obs_shape=slit_center_alt.shape
        
        #determine the hard limb crossing, if it exists
        limb_index=[]
        limb_label=[]
        indices_to_label=np.array([])
        if label_limb:
            if np.any(slit_center_alt==0):
                if slit_center_alt[0]==0:
                    limb_crossing_idx = np.where(slit_center_alt==0)[0][-1]+1
                else:
                    limb_crossing_idx = np.where(slit_center_alt==0)[0][ 0]-1

                if limb_crossing_idx>0 and limb_crossing_idx<len(slit_center_alt):
                    limb_index = [limb_crossing_idx]
                    limb_label  = ['limb'           ]

                    #also label the max alt if we cross the limb
                    if slit_center_alt[0]==0:
                        indices_to_label = np.append(indices_to_label, obs_shape[0]-1)
                    else:
                        indices_to_label = np.append(indices_to_label, 0)
                
        #now add additional indices to label based on the call
        indices_to_label = np.append(indices_to_label,(np.round(np.linspace(0, obs_shape[0]-1,n_labels))))
        indices_to_label = np.unique(indices_to_label).astype(int) #only label each entry once
        
        #figure out what label to print at each index
        alt_labels=(np.round(slit_center_alt[indices_to_label])).astype(int).astype(str)
        alt_labels=[a+' km' for a in alt_labels]

        #add the limb label
        indices_to_label = np.append(indices_to_label,limb_index).astype(int)
        alt_labels       = np.append(      alt_labels,limb_label).astype(str)
        
        #draw the labels!
        for label_idx, alt_idx in enumerate(indices_to_label):
            alt_pixel_y=np.mean(pixel_y[alt_idx:alt_idx+2])

            #prevent the labels from running into the top/bottom of the frame
            edge_tolerance_low=0.03
            if alt_pixel_y<edge_tolerance_low:
                alt_pixel_y=edge_tolerance_low
            edge_tolerange_hi=0.96
            if alt_pixel_y>edge_tolerange_hi:
                alt_pixel_y=edge_tolerange_hi

            text_label = axis.text(mid_pixel_x,alt_pixel_y,
                                   alt_labels[label_idx],
                                   fontsize=orbit_annotation_fontsize,
                                   c=orbit_annotation_color,
                                   alpha=0.5,
                                   va='center',ha='center')
            text_label.set_path_effects([matplotlib.patheffects.withStroke(linewidth=0.35, alpha=0.5, foreground='k')])

def populate_limb_plot(observations, axes, orbit_coords, all_pixel_x, all_pixel_y, all_brightness, colormap, cmapnorm):
    if len(all_pixel_y)==0:
        #no data for this orbit, we've done all we can
        return
    
    #otherwise, let's see if we should plot in altitude or integration number
    switch_to_int_frac=0.6
    all_pixel_y_quant = np.concatenate(all_pixel_y)
    
    if (np.quantile(all_pixel_y_quant,1-switch_to_int_frac)>axes[0].get_ylim()[1] 
        or
        np.quantile(all_pixel_y_quant,  switch_to_int_frac)<axes[0].get_ylim()[0]):
        
        #too much of the data would be off the altitude axes, switch to plotting by integration number
        integration_plot=True
        
        leftmost_axis = None
        leftmost_start=1.0
        for ax in axes:
            ax_x0=ax.get_position().x0
            if ax_x0<leftmost_start:
                leftmost_start=ax_x0
                leftmost_axis=ax
        
        leftmost_axis.set_ylabel('Integration',size=fontsize,labelpad=tickpad,color=orbit_annotation_warning)
        leftmost_axis.yaxis.set_ticks([])
    else:
        #we're good, go ahead and plot the data normally
        integration_plot=False
        
    data_idx=0
    for idx, obs in enumerate(observations):
        obsid_label = obs['label']
        axis_idx = int(obsid_label.replace('IL','').replace('OL','').replace('P',''))-1
        if obs['filename'] !="" and not obs['echelle']:
            if axis_idx < len(axes):
                
                pixel_x=all_pixel_x[data_idx]
                altitudes=all_pixel_y[data_idx]
                brightness=all_brightness[data_idx]
                
                if integration_plot:
                    #rescale the axes and pixel_y
                    axes[axis_idx].set_ylim(0,1)
                    pixel_y=np.linspace(0,1,brightness.shape[0]+1)
                    pixel_y=np.repeat(pixel_y[:,np.newaxis],brightness.shape[1]+1,axis=1)
                    
                    reverse_y=False
                    #determine the direction of motion of the LOS
                    if 'limb' in obs['obsid']:
                        pixel_dist = obs['fits']['PixelGeometry'].data['PIXEL_CORNER_LOS'][:,:,4]
                        pixel_dist = np.repeat(pixel_dist[:,:,np.newaxis],3,axis=2)
                        pixel_vec  = np.transpose(obs['fits']['PixelGeometry'].data['PIXEL_VEC'][:,:,:,4],[0,2,1])
                        sc_vec     = obs['fits']['SpacecraftGeometry'].data['V_SPACECRAFT']
                        sc_vec     = np.repeat(sc_vec[:,np.newaxis,:], pixel_vec.shape[1], axis=1)
                        tp_vec     = sc_vec + pixel_dist*pixel_vec
                        if np.dot(np.mean(tp_vec[-1]-tp_vec[0],axis=0),orbit_coords['camera_up'])<0:
                            #swath is scanning away from apoapsis, reverse the pixel_y scale so higher altitudes are on top
                            reverse_y=True
                    elif obs['obsid']=='periapse':
                        #see if we can determine anything from the altitudes, otherwise leave things unchanged
                        altdiff = altitudes[1:]-altitudes[:-1]
                        if np.min(altdiff)<0 and not np.max(altdiff)>0:
                            #we're scanning down
                            reverse_y=True
                        #all other cases result in alt scanning up or something indeterminate
                    if reverse_y:
                        pixel_y = 1.0-pixel_y
                else:
                    pixel_y=altitudes
                
                #plot the data
                pcol = axes[axis_idx].pcolormesh(pixel_x,pixel_y,brightness,
                                                 norm=cmapnorm,cmap=colormap,linewidth=pcolormesh_edge_width)
                pcol.set_edgecolor('face')
                
                if integration_plot:
                    #put some altitudes on top of the integration plot for reference
                    mid_pixel_index = brightness.shape[1]//2
                    mid_pixel_x = np.mean(pixel_x[mid_pixel_index:mid_pixel_index+2])
                    slit_center_alt = obs['fits']['PixelGeometry'].data['PIXEL_CORNER_MRH_ALT'][:,mid_pixel_index,4]   
                    draw_integration_plot_alt_labels(slit_center_alt, mid_pixel_x, pixel_y, axes[axis_idx], n_labels=3)
                data_idx+=1
    
    return

