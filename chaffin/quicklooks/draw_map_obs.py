from .plot_defaults import *
import numpy as np
import spiceypy as spice
from .draw_map import longitude_mod

def draw_map_obs(obs, map_ax,to_iau_mat):
    #put the observation on the lat/long plot with a text label
    #print(obs['filebasename'])
    
    if 'space' in obs['obsid']:
        #this observation is not pointed at the planet, don't draw a box
        return
    
    
    #we need to get the mso tangent point positions and convert them to iau lat/lon at the appropriate time
    from ..geometry import get_pixel_vec_mso, pixelcorner_avg
    pixel_vec_mso = get_pixel_vec_mso(obs['fits'])
    pixel_vec_mso_corners=pixelcorner_avg(pixel_vec_mso[:,:,:,0],pixel_vec_mso[:,:,:,1],pixel_vec_mso[:,:,:,2])
    pixel_vec_mso_corners=np.transpose(pixel_vec_mso_corners,[1,2,0])
    
    if obs['obsid']!='apoapse':
        #we can proceed assuming Chris found all the right geometry

        #we need to pad the spacecraft pos to the same dimensions as the pixel array
        spacecraft_pos_mso_reshaped=np.repeat(np.repeat(obs['fits']['SpacecraftGeometry'].data["V_SPACECRAFT_MSO"][:,np.newaxis,np.newaxis,:],
                                                        pixel_vec_mso.shape[1],axis=1),
                                              pixel_vec_mso.shape[2],axis=2)

        #to get the position of the pixel corner we could use the distance to the tangent point
        pixel_corner_distance_mso=np.repeat(obs['fits']['PixelGeometry'].data['PIXEL_CORNER_LOS'][:,:,:,np.newaxis],3,axis=3)*pixel_vec_mso
        pixel_los_pos_mso=spacecraft_pos_mso_reshaped+pixel_corner_distance_mso
        pixel_corners_mso=pixelcorner_avg(pixel_los_pos_mso[:,:,:,0],pixel_los_pos_mso[:,:,:,1],pixel_los_pos_mso[:,:,:,2])
        pixel_corners_mso=np.transpose(pixel_corners_mso,[1,2,0])
        
        #now we need to convert to IAU_MARS so we can get lat/lons
        pixel_corners_iau=np.transpose(np.tensordot(to_iau_mat,pixel_corners_mso,axes=([1],[2])),[2,1,0])

        #get the complete boundary at all altitudes
        surface_boundary_iau=np.concatenate([pixel_corners_iau[:,0],pixel_corners_iau[-1,:],pixel_corners_iau[::-1,-1],pixel_corners_iau[0,::-1]])
    else:
        #TODO: replace this whole section with something based off of SPICE limb extraction routine
        
        #we need to subsample the pixel_vec to find the lat/lons on the limb
        avg_angle_along_slit=np.rad2deg(np.average(np.arccos(np.sum(pixel_vec_mso_corners[1:,:]*pixel_vec_mso_corners[:-1,:],axis=2))))
        avg_angle_along_mirror=np.rad2deg(np.average(np.arccos(np.sum(pixel_vec_mso_corners[:,1:]*pixel_vec_mso_corners[:,:-1],axis=2))))

        #determine how much to subsample to get to the desired resolution
        dangle = 10
        zoom_along_slit  = np.ceil(avg_angle_along_slit/dangle)
        zoom_along_mirror= np.ceil(avg_angle_along_mirror/dangle)

        #zoom in on the mso pixel_vecs
        from scipy.ndimage import zoom
        pixel_vec_mso_corners_zoom = zoom(pixel_vec_mso_corners,(zoom_along_slit,zoom_along_mirror,1))

        #we also need to subsample the ETs for the spice intersection routine
        etlist=obs['fits']['Integration'].data['ET']
        etlist=np.append(etlist,etlist[-1]+obs['fits']['Engineering'].data['CADENCE'][0]/1000)
        etlist_zoom=zoom(etlist,zoom_along_mirror)

        #wrapper around SPICE to get MSO intersect locations so we cna remap using the selected time for longitude display
        def get_mso_surface_intersection(et,pixel_vec_mso):
            try:
                intersect=spice.sincpt('ELLIPSOID','Mars',et,'IAU_MARS','CN+S','MAVEN','MAVEN_MSO',pixel_vec_mso)[0]
                to_mso_mat=spice.pxform('IAU_MARS','MAVEN_MSO',et)
                intersect=np.matmul(to_mso_mat,intersect)
            except spice.exceptions.NotFoundError:
                intersect=[np.nan,np.nan,np.nan]
            return intersect

        #this takes at least seconds to run, da
        pixel_corners_mso=np.array([[get_mso_surface_intersection(etlist_zoom[0],v) for v in slitvec] for et,slitvec in zip(etlist_zoom,np.transpose(pixel_vec_mso_corners_zoom,[1,0,2]))])

        #now we can remap to iau_mars using our preferred transform
        pixel_corners_iau=np.transpose(np.tensordot(to_iau_mat,pixel_corners_mso,axes=([1],[2])),[2,1,0])

        surface_indices=np.array(np.transpose(np.where(~np.isnan(pixel_corners_iau[:,:,2]))))
        surface_indices=[(a,np.sort(surface_indices[np.where(surface_indices[:,0]==a)][:,1])[[0,-1]]) for a in set(surface_indices[:,0])]
        surface_boundary_indices=np.concatenate([[[s[0],s[1][0]] for s in surface_indices],#first integration edge
                                                 [[surface_indices[-1][0],s] for s in np.arange(surface_indices[-1][1][0],surface_indices[-1][1][1]+1)], #first slit edge
                                                 [[s[0],s[1][1]] for s in surface_indices[::-1]], #second integration edge
                                                 [[surface_indices[0][0],s] for s in np.arange(surface_indices[0][1][1],surface_indices[0][1][0]-1,-1)] #second slit edge
                                                ])

        surface_boundary_iau=[pixel_corners_iau[a,b,:] for a,b in surface_boundary_indices]

    # OK, now we can draw the polygon representing the observation boundary
    # to do this without breaks let's rotate to a coordinate system centered on the observation

    mean_iau=np.mean(surface_boundary_iau,axis=0)
    mean_iau=mean_iau/np.linalg.norm(mean_iau)
    mean_lat=np.rad2deg(np.arcsin(mean_iau[2]))
    mean_lon=np.rad2deg(np.arctan2(mean_iau[1],mean_iau[0]))
    
    along_slit_vec=np.mean(pixel_vec_mso_corners[-1]-pixel_vec_mso_corners[0],axis=0)
    along_slit_vec=along_slit_vec/np.linalg.norm(along_slit_vec)
    along_slit_vec=np.matmul(to_iau_mat,along_slit_vec)
    pole_vec=np.cross(mean_iau,along_slit_vec)
    pole_lat=np.rad2deg(np.arcsin(pole_vec[2]))
    pole_lon=np.rad2deg(np.arctan2(pole_vec[1],pole_vec[0]))
    
    if mean_lat > 0:
        pole_lat = 90 - mean_lat
        pole_lon = longitude_mod(mean_lon+180)
        central_lon = 0
    else:
        pole_lat = 90 + mean_lat
        pole_lon = mean_lon
        central_lon = 180

    proj_x = mean_iau
    proj_z = np.array([np.cos(np.deg2rad(pole_lat))*np.cos(np.deg2rad(pole_lon)),
                       np.cos(np.deg2rad(pole_lat))*np.sin(np.deg2rad(pole_lon)),
                       np.sin(np.deg2rad(pole_lat))])
    proj_y = np.cross(proj_z,proj_x)
        
    import cartopy.crs as ccrs
    rotated_pole = ccrs.RotatedPole(pole_latitude=pole_lat,
                                    pole_longitude=pole_lon,
                                    central_rotated_longitude=central_lon)

    surface_boundary_normalized=np.array([v/np.linalg.norm(v) for v in surface_boundary_iau])
    surface_boundary_rotated_pole=np.array([np.matmul([proj_x,proj_y,proj_z],v) for v in surface_boundary_normalized])

    rotated_pole_lat = np.rad2deg(np.arcsin(surface_boundary_rotated_pole[:,2]))
    rotated_pole_lon = np.rad2deg(np.arctan2(surface_boundary_rotated_pole[:,1],surface_boundary_rotated_pole[:,0]))
    rotated_pole_lonlat = np.transpose([rotated_pole_lon,rotated_pole_lat])

    import matplotlib.patches as mpatches

    if obs['obsid']!='apoapse':
        fill = True
    else:
        fill = False
    
    poly_rotated = mpatches.Polygon(rotated_pole_lonlat, 
                                    closed=True, 
                                    ec='#888888',
                                    fc='#888888',
                                    fill=fill, 
                                    lw=orbit_annotation_linewidth, 
                                    transform=rotated_pole, 
                                    zorder=3) #zorder = 3 puts these above the shadow but blow the orbit annotations
    map_ax.add_patch(poly_rotated)
    
#     if obs['obsid']=='periapse':
#         #label the swaths
#         text_label = map_ax.text(np.mean(rotated_pole_lon),
#                                  np.mean(rotated_pole_lat),
#                                  obs['label'],
#                                  fontsize=orbit_annotation_fontsize,
#                                  c='#666666',
#                                  va='center',
#                                  ha='center',
#                                  rotation=0,
#                                  transform=rotated_pole)
#         text_label.set_path_effects([matplotlib.patheffects.withStroke(linewidth=0.35, foreground='k'))]
