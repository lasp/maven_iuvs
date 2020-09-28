import glob
import numpy as np
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib import colorbar
from matplotlib.collections import PolyCollection
from .paths import idl_cmap_directory
import warnings


def getcmap(no,reverse=False,vmin=0,vmax=1):
    if idl_cmap_directory == '':
        warnings.warn('No IDL Colorbars directory defined, using Magma')
        cm = mpl.cm.magma()
    else:
        fnames = glob.glob(idl_cmap_directory+str(no).zfill(3)+'*')
        data = np.loadtxt(fnames[0],delimiter=',')
        if reverse:
            data=np.flip(data,axis=0)
        datalength=len(data)
        dmin=int(datalength*vmin)
        dmax=int(datalength*vmax)
        if dmax==datalength:
            dmax=datalength-1
        if dmin==datalength:
            dmin=datalength-1
        data=data[dmin:dmax+1]
        
        cm = LinearSegmentedColormap.from_list('my_cmap',data)
    return cm


def detector_image(fits,integration=0,
                   fig=None,ax=None,
                   norm=None, cmap=109,
                   scale="linear",
                   arange=None,
                   prange=None):
    new_ax=False
    if ax==None:    
        new_ax=True
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1, 1, 1)
    if type(cmap) != int:
        raise ValueError
    else:
        cmap = getcmap(cmap)
        
    ax.set_xlim([0, 1024])
    ax.set_ylim([0, 1024])

    #get the data
    data=fits['detector_dark_subtracted'].data[integration]
    
    #figure out the binning
    spapixlo=fits['Binning'].data['SPAPIXLO'][0]
    spapixhi=fits['Binning'].data['SPAPIXHI'][0]
    spepixlo=fits['Binning'].data['SPEPIXLO'][0]
    spepixhi=fits['Binning'].data['SPEPIXHI'][0]
    if not (set((spapixhi[:-1]+1)-spapixlo[1:])=={0} and set((spepixhi[:-1]+1)-spepixlo[1:])=={0}):
        raise ValueError
    
    spepixrange=np.concatenate([[spepixlo[0]],spepixhi+1])
    spapixrange=np.concatenate([[spapixlo[0]],spapixhi+1])
    
    spepixwidth=spepixrange[1:]-spepixrange[:-1]
    spapixwidth=spapixrange[1:]-spapixrange[:-1]
    
    npixperbin=np.outer(spapixwidth,spepixwidth)
    
    data=data/npixperbin
    
    #figure out what norm to use
    if norm == None:
        if prange == None:
            prange = [0, 100]
        if arange == None:
            arange = [np.percentile(data, prange[0]),
                      np.percentile(data, prange[1])]
        if scale == "linear":
            norm = mpl.colors.Normalize(vmin=arange[0],
                                               vmax=arange[1])
        elif scale == "sqrt":
            norm = mpl.colors.PowerNorm(gamma=0.5,
                                               vmin=arange[0],
                                               vmax=arange[1])
        elif scale == "log":
            norm = mpl.colors.LogNorm(vmin=arange[0],
                                             vmax=arange[1])
        else:
            raise ValueError
    
    ax.patch.set_color('#666666')
    ax.patch.set_alpha(1.0)
    pcm = ax.pcolormesh(spepixrange, spapixrange,
                        data,
                        norm=norm,
                        cmap=cmap)

    #add the colorbar axes
    ax_pos = ax.get_position()
    cax_width_frac = 0.07
    cax_margin = 0.02
    cax = fig.add_axes((ax_pos.x1+cax_margin*ax_pos.width,ax_pos.y0,cax_width_frac*ax_pos.width,ax_pos.height))
    fig.colorbar(pcm, cax=cax)
    if scale=="linear":
        cax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    
    if new_ax:
        #fig.show()
        return fig
    else:
        return


class line_fit_plot:
    
    n_int=0
    n_spa=0
    
    fig=None
    figure_size_x=0
    figure_size_y=0
    
    detector_image_axes=None
    counts_axes=None
    residual_axes=None
    thumbnail_axes=None
    correct_muv=False
    
    
    def __init__(self, myfits, n_int, n_spa, correct_muv):
        self.n_int=n_int
        self.n_spa=n_spa
        self.correct_muv=correct_muv
        
        #set up plot axes
        bin_plot_size = 2 #plot size, square, in
        column_margin=0.1 #in
        row_margin=0.6 #in
        
        n_detector_images_per_int=2
        detector_image_margin=0.65 #in
        
        image_lineplot_margin=0.5 #in, space between detector images and line plots

        counts_plot_frac = 2.5 # ratio of height of counts plot to height of residual plot
        counts_residual_margin=0.05 #in, fraction of plot height to use as margin
        residual_plot_height = (1-counts_residual_margin)/(1+counts_plot_frac)*bin_plot_size
        counts_plot_height = counts_plot_frac*residual_plot_height
        
        #figure out how much space to save on top
        thumbnail_ratio=0.05
        thumbnail_plot_height = n_int*bin_plot_size*thumbnail_ratio
        thumbnail_plot_width  = n_spa*bin_plot_size*thumbnail_ratio
        
        thumbnail_margin=[0.5,0.1]#bottom, top
        
        header_height = thumbnail_margin[1] + thumbnail_plot_height + thumbnail_margin[0]
        header_height = np.max([2,header_height])
        
        margins_x=[0.5,0.1]#left, right
        margins_y=[0.25,header_height]#bottom, top
        
        self.figure_size_x = margins_x[0] + (1+self.correct_muv)*(bin_plot_size+detector_image_margin) + image_lineplot_margin + n_spa*(bin_plot_size+column_margin) - column_margin + margins_x[1]
        self.figure_size_y = margins_y[1] + n_int*(bin_plot_size+row_margin) - row_margin + margins_y[0]
        
        #print(self.figure_size_x)
        #print(self.figure_size_y)
        
        dpi=np.min([100,2**16/self.figure_size_x,2**16/self.figure_size_y])
        
        #print(dpi)
        
        self.fig = plt.figure(figsize=(self.figure_size_x, self.figure_size_y), dpi=dpi)
        
        #make axes for thumbnail
        self.thumbnail_axes = self.fig.add_axes((margins_x[0]/self.figure_size_x,
                                       (self.figure_size_y-thumbnail_margin[1]-thumbnail_plot_height)/self.figure_size_y,
                                       thumbnail_plot_width/self.figure_size_x,
                                       thumbnail_plot_height/self.figure_size_y))
        
        #print some basic info about the files
        file_text_start=1+1/thumbnail_plot_width
        file_info_text='FUV integration report\n'
        file_info_text+=myfits['Primary'].header['FILENAME']+'\n'
        file_info_text+='MCP_VOLT: '+str(myfits['Observation'].data['MCP_VOLT'][0])
        self.thumbnail_axes.text(file_text_start,1,file_info_text,ha='left',va='top',transform=self.thumbnail_axes.transAxes,clip_on=False)
        
        #make axes for each integration and bin
        self.detector_image_axes = np.reshape([None]*(1+self.correct_muv)*n_int,(n_int,1+self.correct_muv))
        self.counts_axes         = np.reshape([None]*n_spa*n_int,(n_int,n_spa))
        self.residual_axes       = np.reshape([None]*n_spa*n_int,(n_int,n_spa))
        row_start_y = self.figure_size_y - margins_y[1] - bin_plot_size
        detector_image_row_start_y = row_start_y + residual_plot_height + counts_residual_margin*bin_plot_size
        for iint in range(n_int):
            plot_start_x = margins_x[0]
            self.detector_image_axes[iint][0] = self.fig.add_axes((plot_start_x/self.figure_size_x,
                                                         row_start_y/self.figure_size_y,
                                                         bin_plot_size/self.figure_size_x,
                                                         bin_plot_size/self.figure_size_y))
            if self.correct_muv:
                plot_start_x += bin_plot_size + detector_image_margin
                self.detector_image_axes[iint][1] = self.fig.add_axes((plot_start_x/self.figure_size_x,
                                                             row_start_y/self.figure_size_y,
                                                             bin_plot_size/self.figure_size_x,
                                                             bin_plot_size/self.figure_size_y))
                plot_start_x += bin_plot_size + detector_image_margin + image_lineplot_margin
            for ispa in range(n_spa):
                self.counts_axes[iint][ispa] = self.fig.add_axes((plot_start_x/self.figure_size_x,
                                                         detector_image_row_start_y/self.figure_size_y,
                                                         bin_plot_size/self.figure_size_x,
                                                         counts_plot_height/self.figure_size_y))
                self.residual_axes[iint][ispa] = self.fig.add_axes((plot_start_x/self.figure_size_x,
                                                         row_start_y/self.figure_size_y,
                                                         bin_plot_size/self.figure_size_x,
                                                         residual_plot_height/self.figure_size_y))
                plot_start_x += bin_plot_size + column_margin
            row_start_y-= (bin_plot_size+row_margin)
            detector_image_row_start_y-= (bin_plot_size+row_margin)
            
    def plot_detector(self, myfits, iint, myfits_muv=None):
        #plot the detector image
        self.detector_image_axes[iint][0].text(0.0,0.5,'integration '+str(iint),ha='right',va='center',rotation=90,transform=self.detector_image_axes[iint][0].transAxes,clip_on=False)
        self.detector_image_axes[iint][0].text(0.5,1.0,'FUV DN/pix',ha='center',va='bottom',transform=self.detector_image_axes[iint][0].transAxes,clip_on=False)
        self.detector_image_axes[iint][0].xaxis.set_ticks([])
        self.detector_image_axes[iint][0].yaxis.set_ticks([])
        detector_image(myfits,iint,fig=self.fig,ax=self.detector_image_axes[iint][0],scale='log',arange=[1,1e5])
        if iint==0:
            self.detector_image_axes[iint][0].text(0.0,-0.025,'spatial bins run from bottom to top\n(small to large keyhole)',size=6,ha='left',va='top',transform=self.detector_image_axes[iint][0].transAxes,clip_on=False)
        if self.correct_muv:
            self.detector_image_axes[iint][1].text(0.5,1.0,'MUV DN/pix',ha='center',va='bottom',transform=self.detector_image_axes[iint][1].transAxes,clip_on=False)
            self.detector_image_axes[iint][1].xaxis.set_ticks([])
            self.detector_image_axes[iint][1].yaxis.set_ticks([])
            detector_image(myfits_muv,iint,fig=self.fig,ax=self.detector_image_axes[iint][1],scale='log',arange=[1,1e5],cmap=98)
            
    def plot_line_fits(self,
                       iint, ispa,
                       fitwaves,
                       fitDN, background_fit, line_fit,
                       DNguess, DN_fit, thislinevalue):
        data_color='#1f78b4'
        fit_color='#a6cee3'
        background_color='#888888'

        # plot line shapes and fits
        self.counts_axes[iint][ispa].text(0.5,1.0,'int '+str(iint)+' spa '+str(ispa),ha='center',va='bottom',transform=self.counts_axes[iint][ispa].transAxes,clip_on=False)
        
        self.counts_axes[iint][ispa].step(fitwaves, fitDN,          color=data_color)
        self.counts_axes[iint][ispa].step(fitwaves, background_fit, color=background_color)
        self.counts_axes[iint][ispa].step(fitwaves, line_fit,       color=fit_color)
        
        self.counts_axes[iint][ispa].text(0.025,0.975,'DN guess = '+str(int(DNguess)),size=6,ha='left',va='top',transform=self.counts_axes[iint][ispa].transAxes,clip_on=False)
        self.counts_axes[iint][ispa].text(0.025,0.9  ,'fit DN = '+str(int(np.round(DN_fit))),size=6,ha='left',va='top',transform=self.counts_axes[iint][ispa].transAxes,clip_on=False)
        self.counts_axes[iint][ispa].text(0.025,0.825,'cal = '+str(np.round(thislinevalue,2))+" kR",size=6,ha='left',va='top',transform=self.counts_axes[iint][ispa].transAxes,clip_on=False)
        self.counts_axes[iint][ispa].xaxis.set_ticks([])
        self.counts_axes[iint][ispa].ticklabel_format(axis='y',style='sci',scilimits=(0,0))

        # plot deviations
        self.residual_axes[iint][ispa].step(fitwaves, (fitDN-line_fit)/np.sum(fitDN),color=data_color)
        self.residual_axes[iint][ispa].set_ylim(-0.06, 0.06)
        self.residual_axes[iint][ispa].yaxis.set_ticks([-0.05, 0, 0.05])
        self.residual_axes[iint][ispa].yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1,decimals=0))  

        if ispa==0:
            self.counts_axes[iint][ispa].set_ylabel('Counts [DN/bin]')
            self.counts_axes[iint][ispa].text(0.025,0.5,'Data',color=data_color,ha='left',va='top',transform=self.counts_axes[iint][ispa].transAxes,clip_on=False)
            self.counts_axes[iint][ispa].text(0.025,0.4,'Fit',color=fit_color,ha='left',va='top',transform=self.counts_axes[iint][ispa].transAxes,clip_on=False)
            self.residual_axes[iint][ispa].text(0.025,0.075,'(fit-data)/(fit DN)',size=6,ha='left',va='bottom',transform=self.residual_axes[iint][ispa].transAxes,clip_on=False)
        else:
            self.counts_axes[iint][ispa].yaxis.set_major_formatter(mpl.ticker.NullFormatter())
            self.residual_axes[iint][ispa].yaxis.set_major_formatter(mpl.ticker.NullFormatter())

#                 if iint!=n_int-1:
#                     self.residual_axes[iint][ispa].xaxis.set_major_formatter(mpl.ticker.NullFormatter()) 

    def finish_plot(self,lineDNmax, linevalues):
        #use the same scale for all the counts axes, based on the largest value
        for iint in range(self.n_int):
            for ispa in range(self.n_spa):    
                self.counts_axes[iint][ispa].set_ylim(0, 1.05*lineDNmax)
                
        #plot the values on the thumbnail axis
        norm = mpl.colors.Normalize(vmin=0,vmax=20)
        pcm = self.thumbnail_axes.pcolormesh(np.arange(self.n_spa+1)-0.5, np.arange(self.n_int+1)-0.5, linevalues,cmap=getcmap(109),norm=norm)
        self.thumbnail_axes.invert_yaxis()
        #add a colorbar
        ax_pos = self.thumbnail_axes.get_position()
        cax_width = 0.2/self.figure_size_x
        cax_margin = 0.05/self.figure_size_x
        cax = self.fig.add_axes((ax_pos.x1+cax_margin,ax_pos.y0,cax_width,ax_pos.height))
        self.fig.colorbar(pcm, cax=cax)

    
def fig2rgb_array(fig):
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    return np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
    
def maven_orbit_image(time, 
                      camera_pos=[1,0,0], camera_up=[0,0,1], extent=3, 
                      parallel_projection=True, view_from_orbit_normal=False, view_from_periapsis=False,
                      show_maven=False, show_orbit=True, label_poles=None,
                      show=True, transparent_background=False, background_color=(0,0,0)):
    from mayavi import mlab
    mlab.options.offscreen = True
    from tvtk.api import tvtk # python wrappers for the C++ vtk ecosystem
    from tvtk.common import configure_input_data
    import spiceypy as spice
    
    orbitcolor=np.array([222,45,38])/255 # a nice red
    orbitcolor=tuple(orbitcolor)
    
    # create a figure window (and scene)
    mlab_pix = 1000
    mfig = mlab.figure(size=(mlab_pix, mlab_pix),bgcolor=background_color)

    mfig.scene.disable_render = True

    # load and map the texture
    import os
    from .paths import anc_dir
    image_file = os.path.join(anc_dir,'marssurface_2.jpg')
    img = tvtk.JPEGReader()
    img.file_name = image_file
    texture = tvtk.Texture(input_connection=img.output_port, interpolate=1)

    # use a TexturedSphereSource, a.k.a. getting our hands dirty
    R = 1
    Nrad = 180
    sphere = tvtk.TexturedSphereSource(radius=R, theta_resolution=Nrad,
                                       phi_resolution=Nrad)
    sphere_mapper = tvtk.PolyDataMapper(input_connection=sphere.output_port)
    mars = tvtk.Actor(mapper=sphere_mapper, texture=texture)
    mars.property.ambient=0.2 # so the nightside is slightly visible
    mars.property.specular=0.15 #make it shinier near dayside
    
    #now apply the rotation matrix
    #tvtk only thinks about rotations with Euler angles, so we need to use a SPICE routine to get these
    myet=spice.str2et(time)
    rmat=spice.pxform('IAU_MARS','MAVEN_MSO',myet)
    trmat=spice.pxform('MAVEN_MSO','IAU_MARS',myet)#we need to use transpose because m2eul assumes the matrix defines a coordinate system rotation, the inverse of the matrix to rotate vectors
    rangles=np.rad2deg(spice.m2eul(trmat,2,1,3))#2,1,3 because vtk performs rotations in the order z,x,y and SPICE wants these in REVERSE order
    mars.orientation=rangles[[1,0,2]]#orientation must be specified as x,y,z rotations in that order even though they are applied in the order above
    #OK, that was hard, but now we're good!

    mfig.scene.add_actor(mars)

    #make a lat/lon grid
    line_t = np.linspace(0,2*np.pi,100)
    line_r = 1.0
    longrid=np.arange(0,360,30)
    grid_linewidth=0.25*mlab_pix/1000
    line_x=[]
    line_y=[]
    line_z=[]
    line_o=[]
    for lon in longrid:
        line_x.append(line_r*np.cos(np.deg2rad(lon))*np.cos(line_t))
        line_x.append([0])
        line_y.append(line_r*np.sin(np.deg2rad(lon))*np.cos(line_t))
        line_y.append([0])
        line_z.append(line_r*np.sin(line_t))
        line_z.append([0])
        line_o.append(np.ones_like(line_t))
        line_o.append([0])

    latgrid=np.arange(-90,90,30)[1:]
    for lat in latgrid:
        line_x.append(line_r*np.cos(np.deg2rad(lat))*np.cos(line_t))
        line_x.append([0])
        line_y.append(line_r*np.cos(np.deg2rad(lat))*np.sin(line_t))
        line_y.append([0])
        line_z.append(line_r*np.sin(np.deg2rad(lat))*np.ones_like(line_t))
        line_z.append([0])
        line_o.append(np.ones_like(line_t))
        line_o.append([0])

    line_x=np.concatenate(line_x)
    line_y=np.concatenate(line_y)
    line_z=np.concatenate(line_z)
    line_o=np.concatenate(line_o)
    line_x,line_y,line_z=np.transpose(np.array([np.matmul(rmat,[x,y,z]) for x,y,z in zip(line_x,line_y,line_z)]))
    mlab.plot3d(line_x,line_y,line_z,line_o,transparent=True,color=(0,0,0),tube_radius=None,line_width=grid_linewidth)


    #add the spacecraft orbit

    #for the given time, we determine the orbit length
    maven_state=spice.spkezr('MAVEN',myet,'MAVEN_MME_2000','NONE','MARS')[0]
    marsmu=spice.bodvrd('MARS','GM',1)[1][0]
    maven_elements=spice.oscltx(maven_state,myet,marsmu)
    orbit_period=1.001*maven_elements[-1]

    #get the times corresponding to the half-orbit ahead and behind
    orbit_subdivisions=2000
    etlist=myet-orbit_period/2+orbit_period*np.linspace(0,1,num=orbit_subdivisions)

    #get the position of the orbit in MSO
    statelist=spice.spkezr('MAVEN',etlist,'MAVEN_MSO','NONE','MARS')[0]
    statelist=np.append(statelist,[statelist[0]],axis=0)
    poslist=np.transpose(statelist)[:3]/3395 #scale to Mars radius = 1

    #plot the orbit
    maven_x,maven_y,maven_z=poslist
    if show_orbit:
        mlab.plot3d(maven_x,maven_y,maven_z,color=orbitcolor,tube_radius=None,line_width=3*mlab_pix/1000)

    #add a dot indicating the location of the Sun
    #this only makes sense with a perspective transform... with orthographic coordinates we're always too far away
    if not parallel_projection:
        sun_distance = 10
        sun_sphere = tvtk.SphereSource(center=(sun_distance,0,0),radius=1*np.pi/180*sun_distance, theta_resolution=Nrad, phi_resolution=Nrad)
        sun_sphere_mapper = tvtk.PolyDataMapper(input_connection=sun_sphere.output_port)
        sun_sphere = tvtk.Actor(mapper=sun_sphere_mapper)
        sun_sphere.property.ambient=1.0
        sun_sphere.property.lighting=False
        #mfig.scene.add_actor(sun_sphere)

        #put a line along the x-axis towards the sun
        sunline_x=np.arange(0,5000,1)
        #mlab.plot3d(sunline_x,0*sunline_x,0*sunline_x,color=(1.0,1.0,1.0),tube_radius=None,line_width=6)


    #define the coordinate system with respect to the camera
    if view_from_periapsis:
        #now we need to get the position of apoapsis and the orbit normal
        rlist=[np.linalg.norm(p) for p in np.transpose(poslist)]
        apoidx=np.argmax(rlist)
        apostate=spice.spkezr('MAVEN',etlist[apoidx],'MAVEN_MSO','NONE','MARS')[0]
        camera_pos=-apostate[:3]
        camera_pos=5*camera_pos/np.linalg.norm(camera_pos)
        camera_up=np.cross(apostate[:3],apostate[-3:])
        camera_up=camera_up/np.linalg.norm(camera_up)
        parallel_projection=True

    if view_from_orbit_normal:
        #now we need to get the position of apoapsis and the orbit normal
        rlist=[np.linalg.norm(p) for p in np.transpose(poslist)]
        apoidx=np.argmax(rlist)
        apostate=spice.spkezr('MAVEN',etlist[apoidx],'MAVEN_MSO','NONE','MARS')[0]
        camera_up=apostate[:3]
        camera_up=camera_up/np.linalg.norm(camera_up)
        camera_pos=np.cross(apostate[:3],apostate[-3:])
        camera_pos=5*camera_pos/np.linalg.norm(camera_pos)
        parallel_projection=True

    camera_pos_norm = camera_pos/np.linalg.norm(camera_pos)
    camera_up = camera_up-camera_pos_norm*np.dot(camera_pos_norm,camera_up)
    camera_up = camera_up/np.linalg.norm(camera_up)
    camera_right = np.cross(-camera_pos_norm,camera_up)


    #set location of camera and orthogonal projection
    camera=mlab.gcf().scene.camera
    if parallel_projection:
        camera_pos=5*camera_pos_norm
        camera.parallel_projection = True
        camera.parallel_scale=extent # half box size
    else:
        camera.parallel_projection = False
        camera.view_angle=50

    camera.position = np.array(camera_pos)
    camera.focal_point = (0,0,0)
    camera.view_up = camera_up
    camera.clipping_range = (0.01,5000)


    #the only way to set a light in mayavi/vtk is with respect to the camera position
    #default lights are uniform and don't fall off with distance, which is what we want
    mfig.scene.light_manager.light_mode = "vtk"
    sun = mfig.scene.light_manager.lights[0]
    sun.activate = True
    sun_vec=(1,0,0)

    #get the elevation/azimuth coordinates
    #elevation is [-90/+90] +90 is from the direction of camera_up
    #azimuth is [-180/+180] +90 is to the right of the image; +/-180 is behind, pointing at the camera -90 is to the left

    #we need to put the sun in scene coordinates
    sun_scene = np.matmul([camera_right, camera_up, camera_pos_norm],sun_vec)
    
    #elevation is the angle is latitude measured wrt the y-axis of the scene
    sun_elevation = np.rad2deg(np.arcsin(np.dot(sun_scene,[0,1,0])))
    #azimuth is the angle in the x-z plane clockwise from the z-axis
    sun_azimuth = np.rad2deg(np.arctan2(sun_scene[0],sun_scene[2]))
    
    sun.azimuth = sun_azimuth
    sun.elevation = sun_elevation
    
    sun.intensity = 1.0-mars.property.ambient

    mfig.scene.disable_render = False
    #mfig.scene.anti_aliasing_frames = 0 # can uncomment to make rendering faster and uglier
    mlab.show()
    mode = 'rgba' if transparent_background else 'rgb'
    img=mlab.screenshot(mode=mode,antialiased=True)
    mlab.close()

    fig,ax=plt.subplots(1,1,dpi=400*mlab_pix/1000,figsize=(2.5,2.5))
    ax.imshow(img)

    #put an arrow along the orbit direction
    arrow_width=5
    arrow_length=1.5*arrow_width
    # put it at the closest point on the orbit to the viewer
    if show_orbit:
        arrowidx=np.argmax([np.dot(camera_pos/np.linalg.norm(camera_pos),p) for p in np.transpose(poslist)])
        arrowetlist=etlist[arrowidx]+5*60*np.array([0,1])
        arrowstatelist=spice.spkezr('MAVEN',arrowetlist,'MAVEN_MSO','NONE','MARS')[0]
        arrowdir=arrowstatelist[1][:3]-arrowstatelist[0][:3]
        arrowdirproj=[np.dot(camera_right,arrowdir),np.dot(camera_up,arrowdir)]
        arrowdirproj=arrowdirproj/np.linalg.norm(arrowdirproj)

        arrowloc=np.transpose(poslist)[arrowidx]
        arrowlocproj=np.array([np.dot(camera_right,arrowloc),np.dot(camera_up,arrowloc)])
        arrowlocdisp=(arrowlocproj+extent)/extent/2
        arrow = ax.annotate('',
	                xytext=arrowlocdisp-0.05*arrowdirproj,
	                xy=arrowlocdisp+0.05*arrowdirproj,
	                xycoords='axes fraction',
	                textcoords='axes fraction',
	                arrowprops=dict(facecolor=orbitcolor,edgecolor='none',width=0,headwidth=arrow_width,headlength=arrow_length))

    if view_from_periapsis: 
        #we need to redraw the arrow so it's always in the same place
        if show_orbit:
            arrow.remove()
            arrowidx=np.argmax([np.dot((camera_right+camera_pos/np.linalg.norm(camera_pos))/np.linalg.norm(camera_right+camera_pos/np.linalg.norm(camera_pos)),p) for p in np.transpose(poslist)])
            arrowetlist=etlist[arrowidx]+5*60*np.array([0,1])
            arrowstatelist=spice.spkezr('MAVEN',arrowetlist,'MAVEN_MSO','NONE','MARS')[0]
            arrowdir=arrowstatelist[1][:3]-arrowstatelist[0][:3]
            arrowdirproj=[np.dot(camera_right,arrowdir),np.dot(camera_up,arrowdir)]
            arrowdirproj=arrowdirproj/np.linalg.norm(arrowdirproj)

            arrowloc=np.transpose(poslist)[arrowidx]
            arrowlocproj=np.array([np.dot(camera_right,arrowloc),np.dot(camera_up,arrowloc)])
            arrowlocdisp=(arrowlocproj+extent)/extent/2
            arrow = ax.annotate('',
		                xytext=arrowlocdisp-0.05*arrowdirproj,
		                xy=arrowlocdisp+0.05*arrowdirproj,
		                xycoords='axes fraction',
		                textcoords='axes fraction',
		                arrowprops=dict(facecolor=orbitcolor,edgecolor='none',width=0,headwidth=arrow_width,headlength=arrow_length))
        label_poles = True


    if view_from_orbit_normal: 
        #we need to redraw the arrow so it's always in the same place
        if show_orbit:
            arrow.remove()
            arrowidx=np.argmax([np.dot((camera_right-camera_up)/np.linalg.norm(camera_right-camera_up),p) for p in np.transpose(poslist)])
            arrowetlist=etlist[arrowidx]+5*60*np.array([0,1])
            arrowstatelist=spice.spkezr('MAVEN',arrowetlist,'MAVEN_MSO','NONE','MARS')[0]
            arrowdir=arrowstatelist[1][:3]-arrowstatelist[0][:3]
            arrowdirproj=[np.dot(camera_right,arrowdir),np.dot(camera_up,arrowdir)]
            arrowdirproj=arrowdirproj/np.linalg.norm(arrowdirproj)

            arrowloc=np.transpose(poslist)[arrowidx]
            arrowlocproj=np.array([np.dot(camera_right,arrowloc),np.dot(camera_up,arrowloc)])
            arrowlocdisp=(arrowlocproj+extent)/extent/2
            arrow = ax.annotate('',
		                xytext=arrowlocdisp-0.05*arrowdirproj,
		                xy=arrowlocdisp+0.05*arrowdirproj,
		                xycoords='axes fraction',
		                textcoords='axes fraction',
		                arrowprops=dict(facecolor=orbitcolor,edgecolor='none',width=0,headwidth=arrow_width,headlength=arrow_length))
        label_poles = True
	
    if label_poles==None:
        label_poles=False

    if label_poles:
        import matplotlib.patheffects as path_effects
        
        #label the north and south pole
        npolepos=np.matmul(rmat,[0,0,1])
        npoleposproj=np.array([np.dot(camera_right,npolepos),np.dot(camera_up,npolepos)])
        npoleposdisp=(npoleposproj+extent)/extent/2
        npolevis=not (np.linalg.norm([npoleposproj])<1 and np.dot(camera_pos,npolepos)<0)
        if npolevis:
            npolelabel = ax.text(*npoleposdisp, 'N',transform=ax.transAxes,color='#888888',ha='center',va='center',size=4,zorder=1)
            npolelabel.set_path_effects([path_effects.withStroke(linewidth=0.75, foreground='k')])


        spolepos=np.matmul(rmat,[0,0,-1])
        spoleposproj=np.array([np.dot(camera_right,spolepos),np.dot(camera_up,spolepos)])
        spoleposdisp=(spoleposproj+extent)/extent/2
        spolevis=not (np.linalg.norm([spoleposproj])<1 and np.dot(camera_pos,spolepos)<0)
        if spolevis:
            spolelabel = ax.text(*spoleposdisp, 'S',transform=ax.transAxes,color='#888888',ha='center',va='center',size=4,zorder=1)
            spolelabel.set_path_effects([path_effects.withStroke(linewidth=0.75, foreground='k')])

    #add a mark for periapsis and apoapsis
    if show_orbit:
        #find periapsis/apoapsis
        rlist=[np.linalg.norm(p) for p in np.transpose(poslist)]
        periidx=np.argmin(rlist)
        peripos=np.transpose(poslist)[periidx]
        periposproj=np.array([np.dot(camera_right,peripos),np.dot(camera_up,peripos)])
        periposdisp=(periposproj+extent)/extent/2
        perivis=not (np.linalg.norm([periposproj])<1 and np.dot(camera_pos,peripos)<0)

        apoidx=np.argmax(rlist)
        apopos=np.transpose(poslist)[apoidx]
        apoposproj=np.array([np.dot(camera_right,apopos),np.dot(camera_up,apopos)])
        apoposdisp=(apoposproj+extent)/extent/2
        apovis=not (np.linalg.norm([apoposproj])<1 and np.dot(camera_pos,apopos)<0)

        #periapsis and apoapsis
        import matplotlib.patches as mpatches
        if perivis:
            peri = mpatches.CirclePolygon(periposdisp, 0.015, resolution=4, transform=ax.transAxes, fc=orbitcolor, lw=0,zorder=10)
            ax.add_patch(peri)
            ax.text(*periposdisp, 'P',transform=ax.transAxes,color='k',ha='center',va='center_baseline',size=4,zorder=10) 

        if apovis:
            apo = mpatches.CirclePolygon(apoposdisp, 0.015, resolution=4, transform=ax.transAxes, fc=orbitcolor,lw=0,zorder=10)
            ax.add_patch(apo)
            ax.text(*apoposdisp, 'A',transform=ax.transAxes,color='k',ha='center',va='center',size=4,zorder=10) 


    #add a dot for the spacecraft location
    if show_maven:
        mavenpos=spice.spkezr('MAVEN',myet,'MAVEN_MSO','NONE','MARS')[0][:3]/3395
        mavenposproj=np.array([np.dot(camera_right,mavenpos),np.dot(camera_up,mavenpos)])
        mavenposdisp=(mavenposproj+extent)/extent/2
        mavenvis=not (np.linalg.norm([mavenposproj])<1 and np.dot(camera_pos,mavenpos)<0)
        if mavenvis:
            maven = mpatches.Circle(mavenposdisp, 0.012, transform=ax.transAxes, fc=orbitcolor,lw=0,zorder=11)
            ax.add_patch(maven)
            ax.text(*mavenposdisp, 'M', transform=ax.transAxes, color='k', ha='center', va='center_baseline', size=4,zorder=11) 

    #suppress all whitespace
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    ax.set_axis_off()
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator()) 
    
    fig.canvas.draw()
    
    rgb_array=fig2rgb_array(fig)
    
    if show==False:
        plt.close()
    
    return_coords={'extent':extent, 'scale':'3395 km', 'camera_pos':camera_pos, 'camera_pos_norm':camera_pos_norm, 'camera_up':camera_up, 'camera_right':camera_right,'orbit_coords':poslist}
    
    return rgb_array, return_coords

    
def maven_orbit_summary(time, show_maven=False):
    fig,ax=plt.subplots(1,1,dpi=400,figsize=(5,5))
    ax.set_xlim(0,2)
    ax.set_ylim(0,2)
    ax.set_aspect(1)

    #make background black
    ax.imshow(np.zeros([2,2,3]),extent=(0,1,0,1),transform=ax.transAxes)
    
    xview, orbinfo=maven_orbit_image(time,camera_pos=[1,0,0], camera_up=[0,0,1],show=False,show_maven=show_maven)
    ax.imshow(xview,extent=(0,1,0,1))
    yview, orbinfo=maven_orbit_image(time,camera_pos=[0,1,0], camera_up=[0,0,1],show=False,show_maven=show_maven)
    ax.imshow(yview,extent=(1,2,0,1))
    zview, orbinfo=maven_orbit_image(time,camera_pos=[0,0,1], camera_up=[-1,0,0],show=False,show_maven=show_maven)
    ax.imshow(zview,extent=(0,1,1,2))
    
    #suppress all whitespace
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    ax.set_axis_off()
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    
    from .time import Ls
    import spiceypy as spice
    ax.text(0.02,0.98,time+"\nL"+r'$_\mathrm{s}$'+" = "+str(int(np.round(Ls(spice.str2et('2019 Jul 02')))))+r'$^\circ$',transform=ax.transAxes,color='w',ha='left',va='top')
