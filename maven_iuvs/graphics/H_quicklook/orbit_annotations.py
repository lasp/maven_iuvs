from .plot_defaults import *
import numpy as np
import spiceypy as spice
import matplotlib

def add_orbit_annotations(orbfileinfo, orbit_ax, orbit_coords):
    for obs in orbfileinfo:
        #determine what label color to use
        label_color = orbit_annotation_color
        if obs['filebasename']=='':
            #there is no file matching this command, label it missing
            label_color = orbit_annotation_warning
        
        
        #get the spacecraft position for the integration times
        etlist=np.linspace(obs['et_start'],obs['et_end'],100)
        statelist=np.array(spice.spkezr('MAVEN',etlist,'MAVEN_MSO','NONE','Mars')[0])
        scpos3d=statelist[:,0:3]
        scvel3d=statelist[:,3:6]

        scpos2d=np.array([[np.dot(v,orbit_coords['camera_right'])/3395,np.dot(v,orbit_coords['camera_up'])/3395] for v in scpos3d])
        scvel2d=np.array([[np.dot(v,orbit_coords['camera_right'])/3395,np.dot(v,orbit_coords['camera_up'])/3395] for v in scvel3d])
        scvel2d=np.array([v/np.linalg.norm(v) for v in scvel2d])
        scperp2d=np.array([[v[1],-v[0]] for v in scvel2d])

        #add barbs at the beginning and end of the observation
        scplot2d=scpos2d+0.125*scperp2d
        scplot2d=np.insert(scplot2d,0,scplot2d[0]-0.065*scperp2d[0],0)
        scplot2d=np.append(scplot2d,[scplot2d[-1]-0.065*scperp2d[-1]],axis=0)

        #orbit_ax.plot(*np.transpose(scpos2d),lw=orbit_annotation_linewidth,c=orbit_annotation_color,clip_on=False)
        orbit_ax.plot(*np.transpose(scplot2d),lw=orbit_annotation_linewidth,c=label_color,clip_on=False)

        #add a label
        midpoint=np.argmin([np.linalg.norm(v-(scplot2d[0]+scplot2d[-1])/2) for v in scplot2d[1:-1]])    
        midpoint_pos=scplot2d[midpoint+1]
        midpoint_perp=scperp2d[midpoint+1]
        labelpos=midpoint_pos+0.02*midpoint_perp

        textangle=np.rad2deg(np.arctan2(midpoint_perp[1],midpoint_perp[0]))
        text_horizontal_alignment='left'
        
        if 'in' in obs['segment']:
            textangle+=180
            text_horizontal_alignment='right'

        print_label=obs['label']
        if obs['echelle'] and not obs['obsid']=='periapse':
            print_label='ECH'
            if obs['obsid'] == 'comm':
                print_label = 'ECHc'
            if obs['obsid'] == 'relay':
                print_label = 'ECHr'

            
        orbit_ax.text(labelpos[0],labelpos[1],print_label,
                      c=label_color,fontsize=orbit_annotation_fontsize,
                      va='center_baseline',ha=text_horizontal_alignment,
                      rotation=textangle,rotation_mode='anchor')


def draw_obs_arrow(myfits, 
                   orbit_x, orbit_y, 
                   orbit_ax, 
                   target_altitude=default_target_altitude, 
                   target_index=None, 
                   arrow_length=None, #length of arrow to draw in units of r_Mars 
                   color='#666666', 
                   get_geom=False):
    
    if target_index!=None:
        if target_index<1:
            target_index=int(np.round(target_index*myfits['Primary'].data.shape[0]))
        slit_center_closest_to_target_pos = target_index
    else:
        slit_center_closest_to_target_pos  = np.argmin((myfits['PixelGeometry'].data['PIXEL_CORNER_MRH_ALT'][:,3,4]-target_altitude)**2)
    
    slit_center_closest_to_target_spacecraft_vec = myfits['SpacecraftGeometry'].data['V_SPACECRAFT_MSO'][slit_center_closest_to_target_pos]
    slit_center_closest_to_target_et = myfits['Integration'].data['ET'][slit_center_closest_to_target_pos]
    slit_center_closest_to_target_rmat = spice.pxform('IAU_MARS','MAVEN_MSO',slit_center_closest_to_target_et)
    slit_center_closest_to_target_pixel_vec_iau_mars = myfits['PixelGeometry'].data['PIXEL_VEC'][slit_center_closest_to_target_pos,:,3,4]
    slit_center_closest_to_target_pixel_vec_mso = np.matmul(slit_center_closest_to_target_rmat,slit_center_closest_to_target_pixel_vec_iau_mars)
    
    slit_center_closest_to_target_dist = myfits['PixelGeometry'].data['PIXEL_CORNER_LOS'][slit_center_closest_to_target_pos,3,4]
    slit_center_closest_to_target_los_vec = slit_center_closest_to_target_spacecraft_vec + slit_center_closest_to_target_dist * slit_center_closest_to_target_pixel_vec_mso

    scpos_2d = np.array([np.dot(slit_center_closest_to_target_spacecraft_vec,orbit_x),
                         np.dot(slit_center_closest_to_target_spacecraft_vec,orbit_y)])/3395
    headpos_2d = np.array([np.dot(slit_center_closest_to_target_los_vec,orbit_x),
                           np.dot(slit_center_closest_to_target_los_vec,orbit_y)])/3395
    pixelvec_2d = np.array([np.dot(slit_center_closest_to_target_pixel_vec_mso,orbit_x),
                            np.dot(slit_center_closest_to_target_pixel_vec_mso,orbit_y)])/3395
    pixelvec_2d = pixelvec_2d/np.linalg.norm(pixelvec_2d)
    
    if arrow_length!=None:
        headpos_2d=scpos_2d+arrow_length*pixelvec_2d
    
    arrow_patch = matplotlib.patches.FancyArrowPatch(posA=scpos_2d,posB=headpos_2d,
                                                     lw=orbit_annotation_linewidth,
                                                     arrowstyle=matplotlib.patches.ArrowStyle("-|>",
                                                                                              head_length=0.5,       
                                                                                              head_width=0.5),
                                                     shrinkA=0,
                                                     shrinkB=0,
                                                     path_effects=[matplotlib.patheffects.withStroke(linewidth=0.5, foreground='k')],
                                                     color=color)
    if not get_geom:
        orbit_ax.add_patch(arrow_patch)
    
    return scpos_2d, headpos_2d
