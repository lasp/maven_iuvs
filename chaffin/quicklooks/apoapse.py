from .plot_defaults import *
import numpy as np
from ..geometry import get_pixel_vec_mso,pixelcorner_avg

def quicklook_apoapse(fig=None,orbno=None, observations=None, orbit_ax=None, orbit_coords=None, map_ax=None, to_iau_mat=None, colormap=None, cmapnorm=None, apoapse_axes=None):
    apoapse_obs = [obs for obs in observations if obs['obsid']=='apoapse' and obs['n_int']>1 and obs['segment']=='apoapse' and not obs['echelle'] and not obs['filename']=='']
    #there's nothing to do for an apoapse observation without a file except put it on the orbit, which we already did.
    
    #if there are no apoapse observations, turn off all the axis frames and return
    if len(apoapse_obs)==0:
        apoapse_axes[0].set_axis_off
        apoapse_axes[2].set_axis_off
        return apoapse_axes
    
    #otherwise, continue drawing the observations  

    #get the MSO pixel vecs for each observation
    all_pixel_vecs_mso=[]
    for idx, obs in enumerate(apoapse_obs):
        pixel_vecs_mso=get_pixel_vec_mso(obs['fits'])
        all_pixel_vecs_mso.append(pixel_vecs_mso)
        
    #
    # plot 1: image of the planet and swaths
    #
    
    #draw bounding box of each swath on top of planet
    all_pixel_corners_2d=[]
    for idx,obs in enumerate(apoapse_obs):
        myfits=obs['fits']

        #we need to pad the spacecraft pos to the same dimensions as the pixel array
        spacecraft_pos_mso_reshaped=np.repeat(np.repeat(myfits['SpacecraftGeometry'].data["V_SPACECRAFT_MSO"][:,np.newaxis,np.newaxis,:],
                                                        all_pixel_vecs_mso[idx].shape[1],axis=1),
                                              all_pixel_vecs_mso[idx].shape[2],axis=2)
        
        
        #to get the position of the pixel corner we could use the distance to the tangent point
        pixel_corner_distance_mso=np.repeat(myfits['PixelGeometry'].data['PIXEL_CORNER_LOS'][:,:,:,np.newaxis],3,axis=3)*all_pixel_vecs_mso[idx]
        
        #or project out to the plane perpendicular to the look direction
        #pixel_corner_distance_mso=-np.dot(spacecraft_pos_mso_reshaped,orbit_coords['camera_up'])/np.dot(all_pixel_vecs_mso[idx],orbit_coords['camera_up'])
        #                         ^minus sign required here because pixel_vec points in opposite direction of spacecraft pos
        #pixel_corner_distance_mso=np.repeat(pixel_corner_distance_mso[:,:,:,np.newaxis],3,axis=3)*all_pixel_vecs_mso[idx]
        
        #now project into the coordinate frame
        pixel_los_pos_mso=spacecraft_pos_mso_reshaped+pixel_corner_distance_mso
        pixel_los_pos_2d_x=np.dot(pixel_los_pos_mso,-orbit_coords['camera_right'])/3395
        pixel_los_pos_2d_y=np.dot(pixel_los_pos_mso,orbit_coords['camera_pos']/np.linalg.norm(orbit_coords['camera_pos']))/3395
        
        #get the average corners for pcolormesh
        pixel_corners_2d_x, pixel_corners_2d_y=pixelcorner_avg(pixel_los_pos_2d_x,pixel_los_pos_2d_y)
        pixel_corners_2d = np.transpose([pixel_corners_2d_x,pixel_corners_2d_y],(1,2,0))
        all_pixel_corners_2d.append(pixel_corners_2d)
        
        #draw the swath plot
        swath_boundary_2d=np.concatenate([pixel_corners_2d[:,0],pixel_corners_2d[-1,:],pixel_corners_2d[::-1,-1],pixel_corners_2d[0,::-1],[pixel_corners_2d[0,0]]])
        
        map_axis=apoapse_axes[1]
        map_axis.plot(swath_boundary_2d[:,0],swath_boundary_2d[:,1],color=axis_color,lw=frame_line_width,clip_on=False)
        
        #figure out where to put the plot label
        mid_pixel_index=pixel_corners_2d.shape[1]//2
        mid_pixel_range=np.arange(mid_pixel_index-2,mid_pixel_index+3)
        text_anchor=[0,0]
        if np.mean(pixel_corners_2d[-1,:,1])>np.mean(pixel_corners_2d[0,:,1]):
            if pixel_corners_2d[-1,-1,0]>pixel_corners_2d[-1,0,0]:
                #secondare if structure is here to support left/right alignment if desired
                text_anchor[0]=np.mean(pixel_corners_2d[-1,mid_pixel_range,0])
                text_anchor[1]=np.min(pixel_corners_2d[-1,mid_pixel_range,1])
            else:
                text_anchor[0]=np.mean(pixel_corners_2d[-1,mid_pixel_range,0])
                text_anchor[1]=np.min(pixel_corners_2d[-1,mid_pixel_range,1])
        else:
            if pixel_corners_2d[0,-1,0]>pixel_corners_2d[0,0,0]:
                text_anchor[0]=np.mean(pixel_corners_2d[0,mid_pixel_range,0])
                text_anchor[1]=np.min(pixel_corners_2d[0,mid_pixel_range,1])
            else:
                text_anchor[0]=np.mean(pixel_corners_2d[0,mid_pixel_range,0])
                text_anchor[1]=np.min(pixel_corners_2d[0,mid_pixel_range,1])            
        
        map_axis.text(text_anchor[0]+0.02,text_anchor[1]-0.02, obs['label'], fontsize=orbit_annotation_fontsize, c=orbit_annotation_color, ha='center',va='top')
        
        #print LT to check orientation of swaths
        #n_int, n_slit, n_waves = myfits['Primary'].data.shape
        #print("A"+str(idx+1).zfill(2)+": LT"+str(myfits['PixelGeometry'].data['PIXEL_LOCAL_TIME'][n_int//2,n_slit//2]))
        
        #draw box on map axis
        #needs better limb detection to look good
        #draw_map_obs(obs,map_ax,to_iau_mat)
        
    
    #
    # plot 2: all swaths laid out in rectangular format side-by-side
    #
    # get lyman alpha brightness of all files
    from ..integration import get_lya
    brightness=[get_lya(obs['fits']) for obs in apoapse_obs]
    
    #use the pixel locations to define the along slit / along integration coordinate system
    scale_by_map_plot        = False
    scale_by_pixel_vec_angle = False
    scale_by_mirror_angle    = True
    
    if scale_by_map_plot+scale_by_pixel_vec_angle+scale_by_mirror_angle != 1:
        print('select only one scaling method in quicklook_apoapse')
        raise
    
    mid_swath_index=len(all_pixel_vecs_mso)//2 #helps to set origins by the middle swath since sometimes first has problems
    if scale_by_map_plot:    
        #scale the swaths by apparent size in the map panel
        alongslitvec=np.mean([v/np.linalg.norm(v) for v in np.concatenate([v[:,-1]-v[:,0] for v in all_pixel_corners_2d])],axis=0)
        alongslitvec=alongslitvec/np.linalg.norm(alongslitvec)
        alongmirrorvec=np.mean([v/np.linalg.norm(v) for v in np.concatenate([v[-1,:]-v[0,:] for v in all_pixel_corners_2d])],axis=0)
        alongmirrorvec=alongmirrorvec-alongslitvec*np.dot(alongslitvec,alongmirrorvec)
        alongmirrorvec=alongmirrorvec/np.linalg.norm(alongmirrorvec)

        pixel_vec_origin=all_pixel_corners_2d[0][0,0]


        start_along_mirror=np.array([np.dot(v[0,mid_pixel_index]-pixel_vec_origin,alongmirrorvec) for v in all_pixel_corners_2d])
        end_along_mirror=np.array([np.dot(v[-1,mid_pixel_index]-pixel_vec_origin,alongmirrorvec) for v in all_pixel_corners_2d])
        length_along_mirror=np.abs(end_along_mirror-start_along_mirror)
        swath_ax_scale=2.75 # number of mars radii represented by the entire axis
    elif scale_by_pixel_vec_angle:
        #scale the swaths by angular extent based on the mirror angles
        alongslitvec=np.median([v/np.linalg.norm(v) for v in np.concatenate([v[:,-1,-1]-v[:,0,-1] for v in all_pixel_vecs_mso])],axis=0)
        alongslitvec=alongslitvec/np.linalg.norm(alongslitvec)
        alongmirrorvec=np.median([v/np.linalg.norm(v) for v in np.concatenate([v[-1,:,-1]-v[0,:,-1] for v in all_pixel_vecs_mso])],axis=0)
        alongmirrorvec=alongmirrorvec-alongslitvec*np.dot(alongslitvec,alongmirrorvec)
        alongmirrorvec=alongmirrorvec/np.linalg.norm(alongmirrorvec)

        pixel_vec_origin=all_pixel_vecs_mso[mid_swath_index][0,0,4]
        
        #or scale the swaths by angular extent based on the mirror angles
        start_along_mirror=np.array([np.rad2deg(np.dot(v[0,mid_pixel_index,0]-pixel_vec_origin,alongmirrorvec)) for v in all_pixel_vecs_mso])
        end_along_mirror=np.array([np.rad2deg(np.dot(v[-1,mid_pixel_index,3]-pixel_vec_origin,alongmirrorvec)) for v in all_pixel_vecs_mso])
        length_along_mirror=np.abs(end_along_mirror-start_along_mirror)
        
        swath_ax_scale=50 if orbno < 8500 else 60 # number of degrees represented by the entire axis (larger during + after aerobraking)
    elif scale_by_mirror_angle:
        fov_angles=[obs['fits']['Integration'].data['FOV_DEG'] for obs in apoapse_obs]
        
        start_along_mirror=np.array([a[0] for a in fov_angles])
        end_along_mirror=np.array([a[-1]+np.median(a[:-1]-a[1:]) for a in fov_angles])
        length_along_mirror=np.abs(end_along_mirror-start_along_mirror)
        length_along_mirror=[0.1 if a==0 else a for a in length_along_mirror] #zero size axes confuse matplotlib, need to give this a bit of space
        
        swath_ax_scale=50 if orbno < 8500 else 60 # number of degrees represented by the entire axis (larger during + after aerobraking)
   
    #print(start_along_mirror)
    #print(end_along_mirror)
    #print(length_along_mirror)
    
    #determine what groups the swaths belong to 
    direction_change=(start_along_mirror[1:]-end_along_mirror[:-1]+0.05)*(end_along_mirror[mid_swath_index]-start_along_mirror[mid_swath_index])
    #print(direction_change)
    groupsplits=[i+1 for i,d in enumerate(direction_change) if d<0]
    groupsplits=np.insert(groupsplits,0,0,axis=0)
    groupsplits=np.append(groupsplits,len(apoapse_obs))
    #print(groupsplits)
    
    #put the swaths on the plot
    axis_bbox=apoapse_axes[0].get_position()
        
    #ensure the swath ordering is the same as in plot 1:
    swathgroups=[]
    for groupidx in range(len(groupsplits)-1):
        swathidx_list=np.arange(groupsplits[groupidx],groupsplits[groupidx+1])
        y_location = [np.nanmedian(all_pixel_corners_2d[idx][:,:,1]) for idx in swathidx_list]
        correct_order = np.argsort(y_location)[::-1] #swaths are drawn on this plot from top to bottom
        swathgroups.append(swathidx_list[correct_order])
    
    #print(swathgroups)
    
    #draw the new axes
    swath_pad_x=0.02 # fraction of axis width
    swath_pad_y=0.075 # fraction of axis width
    swath_x_width=np.min([0.125*8/len(swathgroups),0.15])
    swath_x_space=swath_x_width+swath_pad_x
    swath_x_start=0.5-((len(groupsplits)-1)*swath_x_width+(len(groupsplits)-2)*swath_pad_x)/2

    aposwath_axes=[]
    for groupidx in range(len(groupsplits)-1):
        totalextent=np.sum(length_along_mirror[groupsplits[groupidx]:groupsplits[groupidx+1]])
        y_extent=totalextent/swath_ax_scale+swath_pad_y*(groupsplits[groupidx+1]-groupsplits[groupidx])
        swath_y_top=1.0-(1.0-y_extent)/2
        for swathidx in swathgroups[groupidx]:
            swath_height=length_along_mirror[swathidx]
            swath_height=swath_height/swath_ax_scale
            ax = fig.add_axes((axis_bbox.x0+(swath_x_start+groupidx*swath_x_space)*axis_bbox.width,
                              axis_bbox.y0+(swath_y_top-swath_height-0.5*swath_pad_y)*axis_bbox.height,
                              swath_x_width*axis_bbox.width,
                              swath_height*axis_bbox.height))
            style_axis(ax)
            ax.patch.set_alpha(1)
            ax.set_title(apoapse_obs[swathidx]['label'],fontsize=orbit_annotation_fontsize,c=orbit_annotation_color,pad=0)
            
            swath_x_values=np.linspace(0,1,brightness[swathidx].shape[1]+1)
            swath_y_values=np.linspace(0,1,brightness[swathidx].shape[0]+1)
            
            #we need to flip both axes if the scan starts on the bottom and moves upward:
            if np.nanmean(all_pixel_corners_2d[swathidx][-1,:,1]-all_pixel_corners_2d[swathidx][0,:,1])<0:
                swath_y_values=1-swath_y_values
            if np.nanmean(all_pixel_corners_2d[swathidx][:,-1,0]-all_pixel_corners_2d[swathidx][:,0,0])<0:
                swath_x_values=1-swath_x_values
            
            ax.pcolormesh(swath_x_values,
                          swath_y_values,
                          brightness[swathidx],
                          transform=ax.transAxes,
                          norm=cmapnorm,cmap=colormap,linewidth=pcolormesh_edge_width)
            aposwath_axes.append(ax)
            swath_y_top=swath_y_top-swath_height-swath_pad_y
            #print(swathidx," ",swath_height)
            
                #remember to use same linewidth command etc as above
    
    
    apoapse_axes[0].set_axis_off()
    
    #
    # plot 3: swaths overlaid in same geometry as plot 1
    #
    #now plot the overlaid swaths

    #import pdb; pdb.set_trace()
    
    for idx,obs in enumerate(apoapse_obs):
        plot_pixel_corners = all_pixel_corners_2d[idx]
        plot_brightness = brightness[idx]
        
        #sometimes pixel_vec is missing for the first or last integration
        bad_indices=np.array([idx for idx,vals in enumerate(plot_pixel_corners) if np.any(np.isnan(vals))])

        if len(bad_indices)!=0:
#            import pdb; pdb.set_trace()
            bad_indices=np.concatenate([[-1],bad_indices,[len(plot_pixel_corners)]])
            bad_indices=np.unique(bad_indices)
            
            #find the largest range of continuous data 
            max_continuous_range_index = ((bad_indices[1:]-1)-(bad_indices[:-1]+1)).argmax()
            start_good_idx=bad_indices[max_continuous_range_index]+1
            end_good_idx  =bad_indices[max_continuous_range_index+1]

            plot_pixel_corners=plot_pixel_corners[start_good_idx:end_good_idx]
            plot_brightness   =plot_brightness   [start_good_idx:end_good_idx-1]

        if len(plot_brightness)>0:        
            apoapse_axes[2].pcolormesh(plot_pixel_corners[:,:,0],
                                       plot_pixel_corners[:,:,1],
                                       plot_brightness,
                                       norm=cmapnorm,cmap=colormap,linewidth=pcolormesh_edge_width,clip_on=False)
    
    apoapse_axes[2].text(0,0.1,'swath boundaries drawn at\nlocation of minimum ray height',
                         transform=apoapse_axes[2].transAxes,
                         fontsize=orbit_annotation_fontsize,color=orbit_annotation_color,va='top',ha='center')
    
    apoapse_axes[2].set_axis_off()
        
    return apoapse_axes
