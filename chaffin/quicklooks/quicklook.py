import matplotlib.pyplot as plt
from .get_orbfiles_and_times import get_orbfiles_and_times
from .setup_info_canvas import setup_info_canvas
from .draw_map import draw_map
from .draw_anc import draw_anc
from .draw_file_list import draw_file_list, get_file_list
from .setup_orbit_canvas import setup_orbit_canvas
from .setup_orbit_axes import setup_orbit_axes
from .setup_colorbar import setup_colorbar
from .orbit_annotations import add_orbit_annotations
from .periapse import quicklook_periapse
from .inoutdisk import quicklook_inoutdisk
from .inoutlimb import quicklook_inlimb, quicklook_outlimb
from .inoutcorona import quicklook_incorona, quicklook_outcorona
from .apoapse import quicklook_apoapse
import os

def H_corona_quicklook(orbno,brightness_range=[0,20],save=False,savedir='/home/mike/Documents/MAVEN/IUVS/iuvs_python/quicklooks_png/'):
    observations, orbtimedict = get_orbfiles_and_times(orbno)
    
    fig=plt.figure(figsize=(6,5),dpi=300,facecolor='k');
    fig.subplots_adjust(bottom=0,top=1,left=0,right=1)
    [ax.remove() for ax in fig.axes];

    (orbit_x_ax, orbit_y_ax, map_ax, euvm_ax, swia_ax, mcs_ax, filelist_ax) = setup_info_canvas(fig,orbtimedict)
    map_ax, to_iau_mat = draw_map(fig,map_ax,orbtimedict['orbit_middle_utc']);
    draw_anc(euvm_ax, swia_ax, mcs_ax, orbtimedict);
    draw_file_list(observations, filelist_ax)

    (orbit_ax, orbit_bbox, orbit_coords, orbit_peri_ax) = setup_orbit_canvas(fig, orbtimedict)

    myaxes = setup_orbit_axes(fig, orbit_ax, orbit_coords, orbit_bbox=orbit_bbox, orbtimedict=orbtimedict)
    periapse_axes, apoapse_axes, indisk_axis, inlimb_axes, incorona_axis, outdisk_axis, outlimb_axes, outcorona_axis = myaxes

    colorbar_axis, colormap, cmapnorm = setup_colorbar(fig,outcorona_axis,brightness_range=brightness_range)

    add_orbit_annotations(observations,orbit_ax,orbit_coords)

    periapse_axes = quicklook_periapse(orbno=orbtimedict['orbno'], observations=observations, 
                                       orbit_coords=orbit_coords, 
                                       map_ax=map_ax, to_iau_mat=to_iau_mat, 
                                       colormap=colormap, cmapnorm=cmapnorm, 
                                       orbit_peri_ax=orbit_peri_ax, periapse_axes=periapse_axes)

    outdisk_axis, indisk_axis = quicklook_inoutdisk(orbno=orbtimedict['orbno'], observations=observations, 
                                                    orbit_ax=orbit_ax, orbit_coords=orbit_coords, 
                                                    map_ax=map_ax, to_iau_mat=to_iau_mat, 
                                                    colormap=colormap, cmapnorm=cmapnorm, 
                                                    outdisk_axis=outdisk_axis, indisk_axis=indisk_axis)

    outlimb_axes = quicklook_outlimb(orbno=orbtimedict['orbno'], observations=observations, 
                                     orbit_ax=orbit_ax, orbit_coords=orbit_coords, 
                                     map_ax=map_ax, to_iau_mat=to_iau_mat, 
                                     colormap=colormap, cmapnorm=cmapnorm, 
                                     outlimb_axes=outlimb_axes, fig=fig)

    inlimb_axes = quicklook_inlimb(orbno=orbtimedict['orbno'], observations=observations, 
                                   orbit_ax=orbit_ax, orbit_coords=orbit_coords, 
                                   map_ax=map_ax, to_iau_mat=to_iau_mat, 
                                   colormap=colormap, cmapnorm=cmapnorm, 
                                   inlimb_axes=inlimb_axes, fig=fig)

    outcorona_axis = quicklook_outcorona(orbno=orbtimedict['orbno'], observations=observations, 
                                         orbit_ax=orbit_ax, orbit_coords=orbit_coords, 
                                         map_ax=map_ax, to_iau_mat=to_iau_mat, 
                                         colormap=colormap, cmapnorm=cmapnorm, 
                                         outcorona_axis=outcorona_axis)

    incorona_axis = quicklook_incorona(orbno=orbtimedict['orbno'], observations=observations, 
                                       orbit_ax=orbit_ax, orbit_coords=orbit_coords,
                                       map_ax=map_ax, to_iau_mat=to_iau_mat, 
                                       colormap=colormap, cmapnorm=cmapnorm, 
                                       incorona_axis=incorona_axis)

    apoapse_axes = quicklook_apoapse(fig=fig,
                                     orbno=orbtimedict['orbno'], observations=observations, 
                                     orbit_ax=orbit_ax, orbit_coords=orbit_coords, 
                                     map_ax=map_ax, to_iau_mat=to_iau_mat, 
                                     colormap=colormap, cmapnorm=cmapnorm, 
                                     apoapse_axes=apoapse_axes)
    import datetime

    if save:
        fname=os.path.join(savedir,'orbit'+str(orbno).zfill(5))

        #     #save to PDF with metadata
        #     from matplotlib.backends.backend_pdf import PdfPages

        #     pdf_fname=fname+'.pdf'
        #     with PdfPages(pdf_fname) as pdf:
        #         pdf.savefig(figure=fig,dpi=600)
        #         d = pdf.infodict()
        #         d['Title'] = 'Orbit '+str(orbno)+' Lyman Alpha Overview'
        #         d['Author'] = 'Mike Chaffin'
        #         d['Subject'] = get_filename_list(observations)
        #         d['Keywords'] = ''
        #         d['CreationDate'] = datetime.datetime.today()
        #         d['ModDate'] = datetime.datetime.today()
        
        #save to png with metadata
        pngfname=fname+'.png'
        fig.savefig(pngfname,dpi=600)

        # Use PIL to save some image metadata
        from PIL import Image
        from PIL import PngImagePlugin

        im = Image.open(pngfname)
        meta = PngImagePlugin.PngInfo()
        meta.add_text("Title", 'Orbit '+str(orbno)+' Lyman Alpha Overview')
        meta.add_text("Author", 'Mike Chaffin')
        meta.add_text("Description", get_file_list(observations))
        meta.add_text("Creation Time", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        im.save(pngfname, "png", pnginfo=meta)

    return fig
