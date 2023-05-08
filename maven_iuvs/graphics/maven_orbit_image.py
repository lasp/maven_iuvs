"""Routines to create an image of Mars and the MAVEN orbit at a
specified time"""

import os as _os
import numpy as np
import spiceypy as spice
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import matplotlib.patches as mpatches

from mayavi import mlab
from tvtk.api import tvtk

from maven_iuvs import anc_dir
from maven_iuvs.graphics import fig2rgb_array
from maven_iuvs.time import Ls


def maven_orbit_image(time,
                      camera_pos=[1, 0, 0],
                      camera_up=[0, 0, 1],
                      extent=3,

                      parallel_projection=True,

                      view_from_orbit_normal=False,
                      view_from_periapsis=False,

                      show_maven=False,
                      show_orbit=True,
                      label_poles=None,

                      show=True,
                      transparent_background=False,
                      background_color=(0, 0, 0)):
    """Creates an image of Mars and the MAVEN orbit at a specified time.

    Parameters
    ----------
    time : str
        Time to diplay, in a string format interpretable by spiceypy.str2et.

    camera_pos : length 3 iterable
        Position of camera in MSO coordinates.
    camera_up : length 3 iterable
        Vector defining the image vertical.
    extent : float
        Half-width of image in Mars radii.

    parallel_projection : bool
        Whether to display an isomorphic image from the camera
        position. If False, goofy things happen. Defaults to True.

    view_from_orbit_normal : bool
        Override camera_pos with a camera position along MAVEN's orbit
        normal. Defaults to False.
    view_from_periapsis : bool
        Override camera_pos with a camera position directly above
        MAVEN's periapsis. Defaults to False.

    show_maven : bool
        Whether to draw a circle showing the position of MAVEN at the
        specified time. Defaults to False.
    show_orbit : bool
        Whether to draw the MAVEN orbit. Defaults to True.
    label_poles : bool
        Whether to draw an 'N' and 'S' above the visible poles of Mars.

    show : bool
        Whether to show the image when called, or supress
        display. Defaults to True.
    transparent_background : bool
        If True, the image background is transparent, otherwise it is
        set to background_color. Defaults to False.
    background_color : RGB1 tuple
        Background color to use if transparent_background=False.
        Specified as an RGB tuple with values between 0 and 1.

    Returns
    -------
    rgb_array : 1000x1000x3 numpy array of image RGB values
        Image RGB values.
    return_coords : dict
        Description of the image coordinate system useful for plotting
        on top of output image.

    Notes
    -----
    Call maven_iuvs.load_iuvs_spice() before calling this function to
    ensure kernels are loaded.

    """
    myet = spice.str2et(time)

    # disable mlab display (this is done by matplotlib later)
    mlab.options.offscreen = True

    # create a figure window (and scene)
    mlab_pix = 1000
    mfig = mlab.figure(size=(mlab_pix, mlab_pix),
                       bgcolor=background_color)

    # disable rendering as objects are added
    mfig.scene.disable_render = True

    #
    # Set up the planet surface
    #

    # load and map the Mars surface texture
    image_file = _os.path.join(anc_dir, 'marssurface_2.jpg')
    img = tvtk.JPEGReader()
    img.file_name = image_file
    texture = tvtk.Texture(input_connection=img.output_port, interpolate=1)

    # attach the texture to a sphere
    mars_radius = 3395.
    sphere_radius = 1  # radius of planet is 1 rM
    sphere_resolution = 180  # 180 points on the sphere
    sphere = tvtk.TexturedSphereSource(radius=sphere_radius,
                                       theta_resolution=sphere_resolution,
                                       phi_resolution=sphere_resolution)
    sphere_mapper = tvtk.PolyDataMapper(input_connection=sphere.output_port)
    mars = tvtk.Actor(mapper=sphere_mapper, texture=texture)

    # adjust the reflection properties for a pretty image
    mars.property.ambient = 0.2  # so the nightside is slightly visible
    mars.property.specular = 0.15  # make it shinier near dayside

    # now apply the rotation matrix to the planet

    # tvtk only thinks about rotations with Euler angles, so we need
    # to use a SPICE routine to get these from the rotation matrix

    # to get from the surface to MSO coordinates we'd normally do
    # this:
    rmat = spice.pxform('IAU_MARS', 'MAVEN_MSO', myet)

    # but we need to use transpose because spice.m2eul assumes the matrix
    # defines a coordinate system rotation, the inverse of the matrix
    # to rotate vectors
    trmat = spice.pxform('MAVEN_MSO', 'IAU_MARS', myet)

    # now we can get the Euler angles
    rangles = np.rad2deg(spice.m2eul(trmat, 2, 1, 3))
    #                                      ^^^^^^^^
    #                                      2,1,3 because vtk performs
    #                                      rotations in the order
    #                                      z,x,y and SPICE wants these
    #                                      in REVERSE order

    mars.orientation = rangles[[1, 0, 2]]
    #                           ^^^^^^^
    #                           orientation must be specified as x,y,z
    #                           rotations in that order even though they
    #                           are applied in the order above

    # OK, that was hard, but now we're good!

    mfig.scene.add_actor(mars)

    #
    # make a lat/lon grid
    #

    line_x = []
    line_y = []
    line_z = []
    line_o = []

    line_t = np.linspace(0, 2*np.pi, 100)
    line_r = 1.0

    longrid = np.arange(0, 360, 30)
    for lon in longrid:
        line_x.append(line_r*np.cos(np.deg2rad(lon))*np.cos(line_t))
        line_x.append([0])
        line_y.append(line_r*np.sin(np.deg2rad(lon))*np.cos(line_t))
        line_y.append([0])
        line_z.append(line_r*np.sin(line_t))
        line_z.append([0])
        line_o.append(np.ones_like(line_t))
        line_o.append([0])

    latgrid = np.arange(-90, 90, 30)[1:]
    for lat in latgrid:
        line_x.append(line_r*np.cos(np.deg2rad(lat))*np.cos(line_t))
        line_x.append([0])
        line_y.append(line_r*np.cos(np.deg2rad(lat))*np.sin(line_t))
        line_y.append([0])
        line_z.append(line_r*np.sin(np.deg2rad(lat))*np.ones_like(line_t))
        line_z.append([0])
        line_o.append(np.ones_like(line_t))
        line_o.append([0])

    line_x = np.concatenate(line_x)
    line_y = np.concatenate(line_y)
    line_z = np.concatenate(line_z)
    line_o = np.concatenate(line_o)

    linearray = [np.matmul(rmat, [x, y, z]) for x, y, z in zip(line_x,
                                                               line_y,
                                                               line_z)]
    (line_x, line_y, line_z) = np.transpose(np.array(linearray))

    grid_linewidth = 0.25*mlab_pix/1000
    mlab.plot3d(line_x, line_y, line_z, line_o,
                transparent=True,
                color=(0, 0, 0),
                tube_radius=None,
                line_width=grid_linewidth)

    #
    # compute the spacecraft orbit
    #

    # for the given time, we determine the orbit period
    maven_state = spice.spkezr('MAVEN', myet,
                               'MAVEN_MME_2000', 'NONE', 'MARS')[0]
    marsmu = spice.bodvrd('MARS', 'GM', 1)[1][0]
    maven_elements = spice.oscltx(maven_state, myet, marsmu)
    orbit_period = 1.001*maven_elements[-1]

    # make an etlist corresponding to the half-orbit ahead and behind
    orbit_subdivisions = 2000
    etlist = (myet
              - orbit_period/2
              + orbit_period*np.linspace(0, 1,
                                         num=orbit_subdivisions))

    # get the position of the orbit in MSO
    statelist = spice.spkezr('MAVEN', etlist,
                             'MAVEN_MSO', 'NONE', 'MARS')[0]
    statelist = np.append(statelist, [statelist[0]], axis=0)  # close the orbit
    poslist = np.transpose(statelist)[:3]/mars_radius  # scale to Mars radius

    # plot the orbit
    orbitcolor = np.array([222, 45, 38])/255  # a nice red
    orbitcolor = tuple(orbitcolor)
    maven_x, maven_y, maven_z = poslist
    if show_orbit:
        mlab.plot3d(maven_x, maven_y, maven_z,
                    color=orbitcolor,
                    tube_radius=None,
                    line_width=3*mlab_pix/1000)

    if not parallel_projection:
        # add a dot indicating the location of the Sun
        # this only makes sense with a perspective transform... with
        # orthographic coordinates we're always too far away
        # TODO: non parallel projection results in goofy images
        sun_distance = 10
        sun_sphere = tvtk.SphereSource(center=(sun_distance, 0, 0),
                                       radius=1*np.pi/180*sun_distance,
                                       theta_resolution=sphere_resolution,
                                       phi_resolution=sphere_resolution)
        sun_sphere_mapper = tvtk.PolyDataMapper(input_connection=sun_sphere.output_port)
        sun_sphere = tvtk.Actor(mapper=sun_sphere_mapper)
        sun_sphere.property.ambient = 1.0
        sun_sphere.property.lighting = False
        # mfig.scene.add_actor(sun_sphere)

        # put a line along the x-axis towards the sun
        # sunline_x=np.arange(0, 5000, 1)
        # mlab.plot3d(sunline_x, 0*sunline_x, 0*sunline_x,
        #             color=(1.0,1.0,1.0),
        #             tube_radius=None,line_width=6)

    #
    # Define camera coordinates
    #

    if view_from_periapsis:
        # to do this we need to get the position of apoapsis and the
        # orbit normal
        rlist = [np.linalg.norm(p) for p in np.transpose(poslist)]
        apoidx = np.argmax(rlist)
        apostate = spice.spkezr('MAVEN', etlist[apoidx],
                                'MAVEN_MSO', 'NONE', 'MARS')[0]
        camera_pos = -1.0 * apostate[:3]
        camera_pos = 5 * (camera_pos/np.linalg.norm(camera_pos))
        camera_up = np.cross(apostate[:3], apostate[-3:])
        camera_up = camera_up/np.linalg.norm(camera_up)
        parallel_projection = True

    if view_from_orbit_normal:
        # to do this we need to get the position of apoapsis and the
        # orbit normal
        rlist = [np.linalg.norm(p) for p in np.transpose(poslist)]
        apoidx = np.argmax(rlist)
        apostate = spice.spkezr('MAVEN', etlist[apoidx],
                                'MAVEN_MSO', 'NONE', 'MARS')[0]
        camera_up = apostate[:3]
        camera_up = camera_up/np.linalg.norm(camera_up)
        camera_pos = np.cross(apostate[:3], apostate[-3:])
        camera_pos = 5 * (camera_pos/np.linalg.norm(camera_pos))
        parallel_projection = True

    # construct an orthonormal coordinate system
    camera_pos = np.array(camera_pos)
    camera_pos_norm = camera_pos/np.linalg.norm(camera_pos)
    camera_up = (camera_up
                 - camera_pos_norm*np.dot(camera_pos_norm,
                                          camera_up))
    camera_up = camera_up/np.linalg.norm(camera_up)
    camera_right = np.cross(-camera_pos_norm, camera_up)

    # set location of camera and orthogonal projection
    camera = mlab.gcf().scene.camera
    if parallel_projection:
        camera_pos = 5*camera_pos_norm
        camera.parallel_projection = True
        camera.parallel_scale = extent  # half box size
    else:
        # TODO: this results in goofy images, fix this
        camera.parallel_projection = False
        camera.view_angle = 50
    camera.position = np.array(camera_pos)
    camera.focal_point = (0, 0, 0)
    camera.view_up = camera_up
    camera.clipping_range = (0.01, 5000)

    #
    # Set up lighting
    #

    # The only light is the Sun, which is fixed on the MSO +x axis.

    # VTK's default lights are uniform and don't fall off with
    # distance, which is what we want
    mfig.scene.light_manager.light_mode = "vtk"
    sun = mfig.scene.light_manager.lights[0]
    sun.activate = True
    sun_vec = (1, 0, 0)

    # The only way to set a light in mayavi/vtk is with respect to the
    # camera position. This means we have to get elevation/azimuth
    # coordinates for the Sun with respect to the camera, which could
    # be anywhere.

    # Here's how the coordinate system is defined:
    # elevation:
    #    [-90 -- +90]
    #    +90 places the light along the direction of camera_up
    # azimuth:
    #    [-180 -- +180],
    #    +90 is in the plane of camera_up and camera_right.
    #    +/-180 is behind, pointing at the camera
    #    -90 places light to the left

    # so, to get elevation we need to put the sun in scene coordinates
    sun_scene = np.matmul([camera_right, camera_up, camera_pos_norm],
                          sun_vec)

    # elevation is the angle is latitude measured wrt the y-axis of
    # the scene
    sun_elevation = np.rad2deg(np.arcsin(np.dot(sun_scene, [0, 1, 0])))
    # azimuth is the angle in the x-z plane, clockwise from the z-axis
    sun_azimuth = np.rad2deg(np.arctan2(sun_scene[0], sun_scene[2]))

    # now we can set the location of the light, computed to always lie
    # along MSO+x
    sun.azimuth = sun_azimuth
    sun.elevation = sun_elevation

    # set the brightness of the Sun based on the ambient lighting of
    # Mars so there is no washing out
    sun.intensity = 1.0 - mars.property.ambient

    #
    # Render the 3D scene
    #

    mfig.scene.disable_render = False
    # mfig.scene.anti_aliasing_frames = 0 # can uncomment to make
    #                                     # rendering faster and uglier
    mlab.show()

    mode = 'rgba' if transparent_background else 'rgb'
    img = mlab.screenshot(mode=mode, antialiased=True)
    mlab.close(all=True)  # 3D stuff ends here

    #
    # Draw text and labels in matplotlib
    #

    fig, ax = plt.subplots(1, 1,
                           dpi=400*mlab_pix/1000,
                           figsize=(2.5, 2.5))
    ax.imshow(img)

    # put an arrow along the orbit direction
    if show_orbit:
        arrow_width = 5
        arrow_length = 1.5*arrow_width
        # by default, draw the arrow at the closest point on the orbit
        # to the viewer
        arrowidx = np.argmax([np.dot(camera_pos_norm, p) for p in
                              np.transpose(poslist)])
        if view_from_periapsis:
            # draw the arrow 45 degrees after periapsis
            arrowidx = np.argmax(
                [np.dot(
                    (camera_right + camera_pos_norm)/np.sqrt(2),
                    p)
                 for p in np.transpose(poslist)])
        if view_from_orbit_normal:
            # draw the arrow 45 degrees after periapsis
            arrowidx = np.argmax(
                [np.dot(
                    (camera_right-camera_up)/np.sqrt(2.),
                    p)
                 for p in np.transpose(poslist)])

        arrowetlist = etlist[arrowidx] + 5*60*np.array([0, 1])
        arrowstatelist = spice.spkezr('MAVEN', arrowetlist,
                                      'MAVEN_MSO', 'NONE', 'MARS')[0]
        arrowdir = arrowstatelist[1][:3] - arrowstatelist[0][:3]
        arrowdirproj = [np.dot(camera_right, arrowdir),
                        np.dot(camera_up, arrowdir)]
        arrowdirproj = arrowdirproj/np.linalg.norm(arrowdirproj)

        arrowloc = np.transpose(poslist)[arrowidx]
        arrowlocproj = np.array([np.dot(camera_right, arrowloc),
                                 np.dot(camera_up, arrowloc)])
        arrowlocdisp = (arrowlocproj + extent)/extent/2
        arrow = ax.annotate('',
                            xytext=(arrowlocdisp - 0.05*arrowdirproj),
                            xy=(arrowlocdisp + 0.05*arrowdirproj),
                            xycoords='axes fraction',
                            textcoords='axes fraction',
                            arrowprops=dict(facecolor=orbitcolor,
                                            edgecolor='none',
                                            width=0,
                                            headwidth=arrow_width,
                                            headlength=arrow_length))

    # label the poles
    if view_from_periapsis:
        label_poles = True
    if view_from_orbit_normal:
        label_poles = True
    if label_poles is None:
        label_poles = False

    if label_poles:
        # label the north and south pole if they are visible
        def label_pole(loc, lbl):
            polepos = np.matmul(rmat, loc)
            poleposproj = np.array([np.dot(camera_right, polepos),
                                    np.dot(camera_up, polepos)])
            poleposdisp = (poleposproj+extent)/extent/2

            # determine if the north pole is visible
            polevis = (not (np.linalg.norm([poleposproj]) < 1
                            and np.dot(camera_pos, polepos) < 0))
            if polevis:
                polelabel = ax.text(*poleposdisp, lbl,
                                    transform=ax.transAxes,
                                    color='#888888',
                                    ha='center', va='center',
                                    size=4, zorder=1)
                # outline the letter
                polelabel.set_path_effects([
                    path_effects.withStroke(linewidth=0.75, foreground='k')])

        label_pole([0, 0,  1], 'N')
        label_pole([0, 0, -1], 'S')

    if show_orbit:
        # add a mark for periapsis and apoapsis

        rlist = [np.linalg.norm(p) for p in np.transpose(poslist)]

        # find periapsis/apoapsis
        def label_apsis(apsis_fn, label, **kwargs):
            apsisidx = apsis_fn(rlist)
            apsispos = np.transpose(poslist)[apsisidx]
            apsisposproj = np.array([np.dot(camera_right, apsispos),
                                     np.dot(camera_up, apsispos)])
            apsisposdisp = (apsisposproj + extent)/extent/2
            apsisvis = (not (np.linalg.norm([apsisposproj]) < 1
                             and np.dot(camera_pos, apsispos) < 0))

            if apsisvis:
                apsis = mpatches.CirclePolygon(apsisposdisp, 0.015,
                                               resolution=4,
                                               transform=ax.transAxes,
                                               fc=orbitcolor, lw=0, zorder=10)
                ax.add_patch(apsis)
                ax.text(*apsisposdisp, label,
                        transform=ax.transAxes,
                        color='k',
                        ha='center',
                        size=4, zorder=10,
                        **kwargs)

        label_apsis(np.argmin, 'P', va='center_baseline')
        label_apsis(np.argmax, 'A', va='center')

    if show_maven:
        # add a dot for the spacecraft location
        mavenpos = spice.spkezr('MAVEN', myet,
                                'MAVEN_MSO', 'NONE', 'MARS')[0][:3]/mars_radius
        mavenposproj = np.array([np.dot(camera_right, mavenpos),
                                 np.dot(camera_up, mavenpos)])
        mavenposdisp = (mavenposproj + extent)/extent/2
        mavenvis = (not (np.linalg.norm([mavenposproj]) < 1
                         and np.dot(camera_pos, mavenpos) < 0))
        if mavenvis:
            maven = mpatches.Circle(mavenposdisp, 0.012,
                                    transform=ax.transAxes,
                                    fc=orbitcolor,
                                    lw=0, zorder=11)
            ax.add_patch(maven)
            ax.text(*mavenposdisp, 'M',
                    transform=ax.transAxes,
                    color='k', ha='center', va='center_baseline',
                    size=4, zorder=11)

    # suppress all whitespace around the plot
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    ax.set_axis_off()
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    fig.canvas.draw()

    rgb_array = fig2rgb_array(fig)

    if not show:
        plt.close(fig)

    return_coords = {'extent': extent,
                     'scale': '3395 km',
                     'camera_pos': camera_pos,
                     'camera_pos_norm': camera_pos_norm,
                     'camera_up': camera_up,
                     'camera_right': camera_right,
                     'orbit_coords': poslist}

    return rgb_array, return_coords

def maven_orbit_summary(time, show_maven=False):
    """Creates a three image summary of Mars and the MAVEN orbit at a
    specified time.

    Parameters
    ----------
    time : str
        Time to diplay, in a string format interpretable by
        spiceypy.str2et.
    show_maven : bool
        Whether to place a dot representing MAVEN on the
        orbits. Defaults to False.

    Returns
    -------
    fig : matplotlib.pyplot figure
        Combined figure with three images of Mars and MAVEN's orbit.

    """
    fig, ax = plt.subplots(1, 1,
                           dpi=400,
                           figsize=(5, 5))
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_aspect(1)

    # make background black
    ax.imshow(np.zeros([2, 2, 3]),
              extent=(0, 1, 0, 1),
              transform=ax.transAxes)

    xview, orbinfo = maven_orbit_image(time,
                                       camera_pos=[1, 0, 0],
                                       camera_up=[0, 0, 1],
                                       show=False,
                                       show_maven=show_maven)
    ax.imshow(xview, extent=(0, 1, 0, 1))
    yview, orbinfo = maven_orbit_image(time,
                                       camera_pos=[0, 1, 0],
                                       camera_up=[0, 0, 1],
                                       show=False,
                                       show_maven=show_maven)
    ax.imshow(yview, extent=(1, 2, 0, 1))
    zview, orbinfo = maven_orbit_image(time,
                                       camera_pos=[0, 0, 1],
                                       camera_up=[-1, 0, 0],
                                       show=False,
                                       show_maven=show_maven)
    ax.imshow(zview, extent=(0, 1, 1, 2))

    # suppress all whitespace
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    ax.set_axis_off()
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    ax.text(0.02, 0.98,
            (time
             + "\nL"+r'$_\mathrm{s}$'+" = "
             + str(int(np.round(Ls(spice.str2et(time)))))
             + r'$^\circ$'),
            transform=ax.transAxes,
            color='w',
            ha='left', va='top')

    return fig
