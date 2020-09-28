from .plot_defaults import *
import numpy as np

def setup_orbit_axes(fig, orbit_ax, orbit_coords, orbtimedict=None, orbit_bbox=None, panel_x_start_frac=2.5/6):
    #set up canvas
    figsize       = fig.get_size_inches()
    figure_width  = figsize[0]
    figure_height = figsize[1]
    panel_aspect_ratio = 5/3.5 #legacy
    panel_height  = figure_height
    panel_width   = panel_height/panel_aspect_ratio
    
    panel_width_frac = panel_width / figure_width
    
    #periapse
    allperiapse_width=panel_width_frac*np.array([0.1,0.90])+panel_x_start_frac # fraction of the horizontal to be occupied by periapse viz, overall
    allperiapse_x_start=allperiapse_width[0]
    periapsepad=0.1 # fraction of the width of each swath plot to place in between plots

    periapse_below_fraction=[0.1,0.875] # fraction of area below orbit image to be occupied by plots
    periapse_height=(periapse_below_fraction[1]-periapse_below_fraction[0])*orbit_bbox[2] #fraction of total vertical to be occupied by periapse vis
    periapse_y_start=periapse_below_fraction[0]*orbit_bbox[2] #total figure fractional start position of periapse y

    n_periapse=14
    periapse_width=(allperiapse_width[1]-allperiapse_width[0])/(n_periapse+(n_periapse-1)*periapsepad)
    periapse_x_space=(1+periapsepad)*periapse_width
    periapse_axes=[]
    for i in range(n_periapse):
        ax = fig.add_axes((allperiapse_x_start+i*periapse_x_space,periapse_y_start,periapse_width,periapse_height))
        style_axis(ax)

        if i>0:
            ax.yaxis.set_ticks([])

        #set typical range
        ax.set_xlim([0,1])#swath
        ax.set_ylim([60,275])
        if i==0:
            ax.yaxis.set_ticks([100,150,200,250])
            ax.set_ylabel('Altitude [km]',size=fontsize,labelpad=tickpad)

        periapse_axes.append(ax)

    #apoapse
    n_apo_views=3
    apo_equal_width=0.3*panel_width_frac
    widen_first=0.1 #fraction to widen first axis
    apo_view_width=np.ones(n_apo_views)*(apo_equal_width*(1-widen_first/(n_apo_views-1)))
    apo_view_width[0]=apo_equal_width*(1+widen_first)
    
    apo_view_height=figure_width/figure_height*apo_equal_width

    allapo_width=panel_width_frac*np.array([0.025,0.975])+panel_x_start_frac
    allapo_x_start=allapo_width[0]

    apo_y_start = ((1.0+orbit_bbox[3])-apo_view_height)/2

    apo_x_pad = ((allapo_width[1]-allapo_width[0])-n_apo_views*apo_equal_width)/(n_apo_views-1)*0.5
    apoapse_axes = []
    apo_x_start=allapo_x_start
    for i in range(n_apo_views):       
        ax = fig.add_axes((apo_x_start,apo_y_start,apo_view_width[i],apo_view_height))
        apo_x_start+=apo_view_width[i]
        apo_x_start+=apo_x_pad
        if i==0:
            apo_x_start+=apo_x_pad*1.2/0.8
        style_axis(ax)
        #turn off autoscaling
        ax.set_autoscale_on(False)
        ax.set_axis_off()
        apoapse_axes.append(ax)

    if orbtimedict!=None:
        #insert an image of Mars from apoapsis on the middle apoapsis axis
        from ..graphics import maven_orbit_image
        apoview_img=maven_orbit_image(orbtimedict['apo_middle_utc'],
                                      camera_pos=3*orbit_coords['camera_up'],
                                      camera_up=orbit_coords['camera_pos'],
                                      show_orbit=False,extent=3,label_poles=True,show=False)[0]
        map_axis=apoapse_axes[1]
        apoview_extent_x=3*map_axis.get_position().width/orbit_ax.get_position().width
        apoview_extent_y=3*map_axis.get_position().height/orbit_ax.get_position().height
        pos=map_axis.get_position()
        map_axis.set_xlim(-apoview_extent_x,apoview_extent_x)
        map_axis.set_ylim(-apoview_extent_y,apoview_extent_y)
        map_axis.imshow(apoview_img,extent=(-3,3,-3,3),transform=map_axis.transData,aspect='auto')
        map_axis.text(0.5,1.0,'isometric view\nfrom apoapsis',
                      transform=map_axis.transAxes,                      
                      fontsize=orbit_annotation_fontsize,color=orbit_annotation_color,va='bottom',ha='center')
        map_axis.set_axis_off()
        
        #set up the overlaid swath axis also
        apoapse_axes[2].set_xlim(-apoview_extent_x,apoview_extent_x)
        apoapse_axes[2].set_ylim(-apoview_extent_y,apoview_extent_y)


        
    #inlimb
    # define the position and size of these using the periapse axes
    n_inlimb=6

    inlimb_copy_start=5
    inlimb_copy_increment=-1
    inlimb_copy_repeat=6

    inlimb_x_offset=-0.5*periapse_width
    inlimb_y_offset=periapse_height*1.85
    inlimb_y_spacing=periapse_height*1.1

    inlimb_axes=[]
    for i in range(n_inlimb):
        peribbox=periapse_axes[inlimb_copy_start+inlimb_copy_increment*(i%inlimb_copy_repeat)].get_position()

        ax = fig.add_axes((peribbox.x0+inlimb_x_offset,peribbox.y0+inlimb_y_offset+(i//inlimb_copy_repeat)*inlimb_y_spacing,peribbox.width,peribbox.height))
        style_axis(ax)

        if i!=n_inlimb-1:
            ax.yaxis.set_ticks([])

        #set typical range
        ax.set_xlim([0,1])#swath
        ax.set_ylim([60,275])
        if i==n_inlimb-1:
            ax.yaxis.set_ticks([100,150,200,250])
            ax.set_ylabel('Altitude [km]',size=fontsize,labelpad=tickpad)

        inlimb_axes.append(ax)

    #incorona/space
    # only one of these!
    # copy from inlimb
    incorona_inlimb_copy=-3
    inlimb_bbox=inlimb_axes[incorona_inlimb_copy].get_position()

    incorona_y_frac = 0.85
    incorona_height = incorona_y_frac*(apoapse_axes[0].get_position().y0-inlimb_bbox.y1)
    incorona_y_start = (1-incorona_y_frac)/2.0*(apoapse_axes[0].get_position().y0-inlimb_bbox.y1)

    incorona_x_offset=-0*inlimb_bbox.width

    incorona_axis = fig.add_axes((inlimb_bbox.x0+incorona_x_offset,inlimb_bbox.y1+incorona_y_start,inlimb_bbox.width,incorona_height))
    style_axis(incorona_axis)

    incorona_axis.set_xlim([0,1])#swath
    incorona_axis.set_ylim([250,4250])
    incorona_axis.yaxis.set_ticks(np.arange(500,4250,500))
    incorona_axis.set_ylabel('Altitude [km]',size=fontsize,labelpad=tickpad)

    #outlimb

    n_outlimb=n_inlimb

    outlimb_copy_start=-6
    outlimb_copy_increment=1
    outlimb_copy_repeat=6

    outlimb_x_offset=1.5*periapse_width
    outlimb_y_offset=inlimb_y_offset
    outlimb_y_spacing=inlimb_y_spacing

    outlimb_axes=[]
    for i in range(n_outlimb):
        peribbox=periapse_axes[outlimb_copy_start+outlimb_copy_increment*(i%outlimb_copy_repeat)].get_position()

        ax = fig.add_axes((peribbox.x0+outlimb_x_offset,peribbox.y0+outlimb_y_offset+(i//outlimb_copy_repeat)*outlimb_y_spacing,peribbox.width,peribbox.height))
        style_axis(ax)

        if i>0:
            ax.yaxis.set_ticks([])

        #set typical range
        ax.set_xlim([0,1])#swath
        ax.set_ylim([60,275])
        if i==0:
            ax.yaxis.set_ticks([100,150,200,250])
            ax.set_ylabel('Altitude [km]',size=fontsize,labelpad=tickpad)

        outlimb_axes.append(ax)
        
    #outdisk
    copy_bbox=outlimb_axes[0].get_position()
    outdisk_axis = fig.add_axes((copy_bbox.x0-2.5*copy_bbox.width,copy_bbox.y0,copy_bbox.width,copy_bbox.height))
    style_axis(outdisk_axis)
    
    #indisk
    copy_bbox=outdisk_axis.get_position()
    indisk_axis = fig.add_axes((copy_bbox.x0-periapse_x_space,copy_bbox.y0,copy_bbox.width,copy_bbox.height))
    style_axis(indisk_axis)
    indisk_axis.set_ylabel('Integration',size=fontsize,labelpad=tickpad)
    
    #outcorona/space
    # only one of these!
    # copy from outlimb
    outcorona_outlimb_copy=3
    outlimb_bbox=outlimb_axes[outcorona_outlimb_copy].get_position()

    outcorona_y_frac = incorona_y_frac
    outcorona_height = incorona_height
    outcorona_y_start = incorona_y_start

    outcorona_x_offset=-1.25*incorona_x_offset

    outcorona_axis = fig.add_axes((outlimb_bbox.x0+outcorona_x_offset,outlimb_bbox.y1+outcorona_y_start,outlimb_bbox.width,outcorona_height))
    style_axis(outcorona_axis)

    outcorona_axis.set_xlim([0,1])#swath
    outcorona_axis.set_ylim([250,4250])
    outcorona_axis.yaxis.set_ticks(np.arange(500,4250,500))
    outcorona_axis.set_ylabel('Altitude [km]',size=fontsize,labelpad=tickpad)
    
    return periapse_axes, apoapse_axes, indisk_axis, inlimb_axes, incorona_axis, outdisk_axis, outlimb_axes, outcorona_axis
