from .plot_defaults import *
import numpy as np
import spiceypy as spice


def setup_info_canvas(fig, orbtimedict, panel_x_start_frac=0):
    #set up canvas
    figsize       = fig.get_size_inches()
    figure_width  = figsize[0]
    figure_height = figsize[1]
    panel_aspect_ratio = 5/2.5 #legacy
    panel_height  = figure_height
    panel_width   = panel_height/panel_aspect_ratio
    
    panel_width_frac = panel_width / figure_width

    ax = fig.add_axes([panel_x_start_frac,0,panel_width_frac,1])
    ax.set_axis_off()

    #orbit name
    ax.text(0.01,0.99,'Orbit',size=20,c='#DDDDDD',ha='left',va='top')
    ax.text(0.7,0.99,str(orbtimedict['orbno']),size=20,c='#DDDDDD',ha='right',va='top')
    
    import datetime
    starttimestring=spice.et2utc(orbtimedict['orbit_start_et'],'C',0)[:-3]
    starttimestring=datetime.datetime.strptime(starttimestring,"%Y %b %d %H:%M").strftime("%Y %b %d %H:%M")
    endtimestring=spice.et2utc(orbtimedict['orbit_end_et'],'C',0)[:-3]
    endtimestring=datetime.datetime.strptime(endtimestring,"%Y %b %d %H:%M").strftime("%Y %b %d %H:%M")
    
    ax.text(0.02,0.94,starttimestring+' - '+endtimestring,size=6,c='#DDDDDD',ha='left',va='top')
    from ..time import Ls
    Ls,MY = Ls(orbtimedict['orbit_middle_utc'],return_marsyear=True)
    ax.text(0.02,0.92,'L'+r'$_\mathrm{s} \sim $'+str(np.around(Ls,1))+", MY "+str(MY),size=6,c='#DDDDDD',ha='left',va='top')
    
    #warning based on caveats file to maybe implement someday
    caveats_text=''
    #caveats_text=get_caveats_text(orbtimedict['orbno'])
    ax.text(0.98,0.92,caveats_text,size=6,c='#DDAAAA',ha='right',va='top')
    
    #orbit view from +x and -y
    orbit_pad_x=0.02*panel_width_frac
    orbit_ax_width=(panel_width_frac-orbit_pad_x*3)/2
    orbit_ax_height=figure_width/figure_height*orbit_ax_width
    orbit_ax_y_start=0.90-orbit_ax_height
    orbit_ax_x_start=panel_x_start_frac+orbit_pad_x
    orbit_x_ax=fig.add_axes((orbit_ax_x_start,
                             orbit_ax_y_start,
                             orbit_ax_width,
                             orbit_ax_height))
    orbit_x_ax.autoscale(False)
    #orbit_x_ax.set_axis_off()
    style_axis(orbit_x_ax,color='#222222')
    
    orbit_y_ax=fig.add_axes((orbit_ax_x_start+orbit_ax_width+orbit_pad_x,
                             orbit_ax_y_start,
                             orbit_ax_width,
                             orbit_ax_height))
    orbit_y_ax.autoscale(False)
    #orbit_x_ax.set_axis_off()
    style_axis(orbit_y_ax,color='#222222')
    
    #get the images
    from ..graphics import maven_orbit_image
    orbimg_arr_x, orbit_coords_x = maven_orbit_image(orbtimedict['orbit_middle_utc'],show=False,camera_pos=[1,0,0],camera_up=[0,0,1])
    orbimg_arr_y, orbit_coords_y = maven_orbit_image(orbtimedict['orbit_middle_utc'],show=False,camera_pos=[0,1,0],camera_up=[0,0,1])

    def includemars(lim):
        if lim[0]>-1.1:
            lim[0]=-1.1
        if lim[1]<1.1:
            lim[1]=1.1
        return lim
    
    orb_extent_x = includemars(np.array([np.min(orbit_coords_x['orbit_coords'][0]),np.max(orbit_coords_x['orbit_coords'][0])]))
    orb_extent_y = includemars(np.array([np.min(orbit_coords_x['orbit_coords'][1]),np.max(orbit_coords_x['orbit_coords'][1])]))
    orb_extent_z = includemars(np.array([np.min(orbit_coords_x['orbit_coords'][2]),np.max(orbit_coords_x['orbit_coords'][2])]))
    
    xyzextent = [np.max(v)-np.min(v) for v in [orb_extent_x,orb_extent_y,orb_extent_z]]
    
    extent_scale=[]
    if np.argmax(xyzextent)==0:
        extent_scale=orb_extent_x
    if np.argmax(xyzextent)==1:
        extent_scale=orb_extent_y
    if np.argmax(xyzextent)==2:
        extent_scale=orb_extent_z
    
    def scale_to(extent,extent_scale,orbit_image_pad=1.1):
        return (extent-np.mean(extent))*orbit_image_pad*(extent_scale[1]-extent_scale[0])/(extent[1]-extent[0])+np.mean(extent)

    orbit_x_ax.set_xlim(scale_to(orb_extent_y,extent_scale))
    orbit_x_ax.set_ylim(scale_to(orb_extent_z,extent_scale))

    orbit_x_ax.imshow(orbimg_arr_x,
                      extent=(-orbit_coords_x['extent'],
                              orbit_coords_x['extent'],
                              -orbit_coords_x['extent'],
                              orbit_coords_x['extent']),
                      transform=orbit_x_ax.transData,
                      aspect='auto')
    
    orbit_y_ax.set_xlim(np.flip(scale_to(orb_extent_x,extent_scale)))
    orbit_y_ax.set_ylim(scale_to(orb_extent_z,extent_scale))
    orbit_y_ax.imshow(orbimg_arr_y,
                      extent=(orbit_coords_x['extent'],
                              -orbit_coords_x['extent'],
                              -orbit_coords_x['extent'],
                              orbit_coords_x['extent']),
                      transform=orbit_y_ax.transData,
                      aspect='auto')
    
    #lat/lon view of orbit track
    map_pad_left=0.2/figure_width
    map_pad_right=orbit_pad_x
    map_ax_width=panel_width_frac-map_pad_left-map_pad_right
    map_ax_height=0.5*figure_width/figure_height*map_ax_width
    map_ax_y_start=orbit_ax_y_start-0.075/figure_height-map_ax_height
    map_ax_x_start=panel_x_start_frac+map_pad_left
    map_ax=fig.add_axes((map_ax_x_start,
                         map_ax_y_start,
                         map_ax_width,
                         map_ax_height))
    map_ax.autoscale(False)
    style_axis(map_ax)
    
    #plots of EUVM, SWIA, MCS dust
    anc_pad_left=0.4/figure_width
    anc_pad_right=map_pad_right
    anc_ax_width=panel_width_frac-anc_pad_left-anc_pad_right
    anc_sep_y=0.15/figure_height
    anc_pad_y=0.05/figure_height
    anc_ax_height=0.35*map_ax_height
    anc_ax_x_start=panel_x_start_frac+anc_pad_left
    euvm_ax=fig.add_axes((anc_ax_x_start,
                          map_ax_y_start-anc_sep_y-anc_ax_height,
                          anc_ax_width,
                          anc_ax_height))
    style_axis(euvm_ax)
    swia_ax=fig.add_axes((anc_ax_x_start,
                          map_ax_y_start-anc_sep_y-2*anc_ax_height-anc_pad_y,
                          anc_ax_width,
                          anc_ax_height))
    style_axis(swia_ax)
    mcs_ax=fig.add_axes((anc_ax_x_start,
                         map_ax_y_start-anc_sep_y-3*anc_ax_height-2*anc_pad_y,
                         anc_ax_width,
                         anc_ax_height))
    style_axis(mcs_ax)
    
    
    #list of all orbit files (ideally in a copy-paste text box)
    filelist_ax=fig.add_axes((orbit_ax_x_start,
                              orbit_ax_x_start*figure_width/figure_height,
                              panel_width_frac-2*orbit_pad_x,
                              0.5/figure_height))
    style_axis(filelist_ax,color='#222222')
    #filelist_ax.set_axis_off()

    return (orbit_x_ax, orbit_y_ax,
            map_ax,
            euvm_ax, swia_ax, mcs_ax,
            filelist_ax)
