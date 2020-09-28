import numpy as np
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection, PolyCollection
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.io import fits



def get_pixel_vec_mso(myfits):
    import spiceypy as spice
    pixel_vecs=np.transpose(myfits['PixelGeometry'].data['PIXEL_VEC'],(0,2,3,1))
    pixel_vecs_shape=pixel_vecs.shape
    rmat_iau_to_mso=np.array([spice.pxform('IAU_MARS','MAVEN_MSO',t) for t in myfits['Integration'].data['ET']])
    pixel_vecs_mso=np.matmul(np.reshape(np.repeat(rmat_iau_to_mso[:,np.newaxis,:,:],5*pixel_vecs.shape[1],axis=1),(-1,3,3)),
                             np.reshape(pixel_vecs,(-1,3))[:,:,np.newaxis])
    pixel_vecs_mso=np.reshape(pixel_vecs_mso,pixel_vecs_shape)
    return pixel_vecs_mso

def pixelcorner_avg(pixel_x, pixel_y, pixel_z=None, integration_cross_slit=None):
    #return average pixel corners for input x and y arrays
    n1,n2=pixel_x.shape[:2]
    
    #axis 0 of pixel_x, pixel_y corresponds to integration #
    #axis 1 corresponds to slit position
    #axis 3 corresponds to pixel corner
    #  pixel_corner quantities look like this:
    #  
    #      ^
    #      | along slit (towards big keyhole)
    #   -------
    #   |2   3|
    #   |  4  |---> cross slit (dispersion direction?)
    #   |0   1|      Note: whether this is in the direction of integration
    #   -------            depends on the direction of slit motion on the sky
    

    if integration_cross_slit == None:
        # attempt to determine if the 0-1 direction corresponds to the direction of integration or not
        # use pixel_y to do this because 1) pixel_x is sometimes slit-relative and meaningless
        #                                2) pixel_x is sometimes longitude with a branch cut
        # there could still be problems with pixel_y latitude crossing the pole... ignoring this for now
        pixel_corner_direction = pixel_y[0,0,1] - pixel_y[0,0,0]
        integration_direction  = pixel_y[1,0,0] - pixel_y[0,0,0]
        integration_cross_slit = (pixel_corner_direction*integration_direction > 0)
    
    #bottom/top = along slit
    #left/right = along integration
    if integration_cross_slit:
        pixel_bottom_left=0
        pixel_bottom_right=1
        pixel_top_left=2
        pixel_top_right=3
    else:
        pixel_bottom_left=1
        pixel_bottom_right=0
        pixel_top_left=3
        pixel_top_right=2
        
    #join and transpose so we can do this once for x and y (and z)
    if type(pixel_z)!=type(None):
        pixelxy=np.transpose([pixel_x,pixel_y,pixel_z],(1,2,3,0))
        n_dim=3
    else:
        pixelxy=np.transpose([pixel_x,pixel_y],(1,2,3,0))
        n_dim=2
        
    #adds up the interior grid points to get an average grid
    avgxy=(pixelxy[:-1,:-1,pixel_top_right]+pixelxy[1:,:-1,pixel_top_left]+pixelxy[:-1,1:,pixel_bottom_right]+pixelxy[1:,1:,pixel_top_left])/4
    #now let's do the first/last integration
    avgxy=np.concatenate(([(pixelxy[0,:-1,pixel_top_left]+pixelxy[0,1:,pixel_bottom_left])/2],
                          avgxy,
                          [(pixelxy[-1,:-1,pixel_top_right]+pixelxy[-1,1:,pixel_bottom_right])/2]),axis=0)
    #now the top/bottom of the slit and observation corners
    avgxy=np.concatenate((np.reshape(np.concatenate(([pixelxy[0,0,pixel_bottom_left]],(pixelxy[:-1,0,pixel_bottom_right]+pixelxy[1:,0,pixel_bottom_left])/2,[pixelxy[-1,0,pixel_bottom_right]]),axis=0),(n1+1,1,n_dim)),
                          avgxy,
                          np.reshape(np.concatenate(([pixelxy[0,-1,pixel_top_left]],(pixelxy[:-1,-1,pixel_top_right]+pixelxy[1:,-1,pixel_top_left])/2,[pixelxy[-1,-1,pixel_top_right]]),axis=0),(n1+1,1,n_dim))),
                         axis=1)
    
    #now we have an array of dimensions [n_int+1, n_slit+1, 2-3] containing the averaged corners
    #this lets us plot with pcolormesh without any gaps
    
    if type(pixel_z)!=type(None):
        return avgxy[:,:,0], avgxy[:,:,1], avgxy[:,:,2]
    else:
        return avgxy[:,:,0], avgxy[:,:,1]


def get_pixel_mrh_alt(scvec_iau,pixelvec_iau):
    #for some reason this is giving different values from Chris' pixelgeometry, not using it
    
    mars_radii = spice.gdpool('BODY499_RADII',0,3)
    
    pt, alt = spice.npedln(mars_radii[0],mars_radii[1],mars_radii[2],
                           scvec_iau,
                           pixelvec_iau)
    
    if alt==0:
        #the line intersects the ellipsoid
        
        #we need to find the intersection point 
        #on the other side of the planet to get 
        #the correct negative altitude
        dist_add = 10000 #km     
        
        dist_to_tanpt = -np.sum(scvec_iau*pixelvec_iau)
        if dist_to_tanpt > 0:
            #the tangent point is along the look direction
            pt_otherside, alt_otherside = spice.npedln(mars_radii[0],mars_radii[1],mars_radii[2],
                                                       scvec_iau+dist_add*pixelvec_iau,
                                                       -pixelvec_iau)
        else:
            #the tangent point is behind the observation
            pt_otherside, alt_otherside = spice.npedln(mars_radii[0],mars_radii[1],mars_radii[2],
                                                       scvec_iau-dist_add*pixelvec_iau,
                                                       pixelvec_iau)
        
        pt = (pt+pt_otherside)/2
        
        #find the ellipsoid point directly above the tangent point
        norm_pt = pt/np.linalg.norm(pt)
        surf_pt = spice.surfpt(pt,norm_pt,mars_radii[0],mars_radii[1],mars_radii[2])
        alt = np.linalg.norm(pt) - np.linalg.norm(surf_pt)
        pt=surf_pt
    
    return alt




def unit_vector(vector):
    """ Returns the unit vector of the vect?or.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return 180/np.pi*np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def lrl_vec(r, v):
    # get laplace-runge-lenz vector from spacecraft position and velocity
    # needs position and velocity in inertial frame, km, km/s
    vv = 1e3*v
    rr = 1e3*r
    G = 6.67408e-11
    M = 6.4171e23
    lang = np.cross(rr, vv)
    lrlvec = np.cross(vv, lang)-G*M*(rr/np.linalg.norm(rr))
    return lrlvec/np.linalg.norm(lrlvec)


def getcoordmats(fits, frame):
    # extract spacecraft coordinate matrix from a fits file, doe a
    # specified frame: MSO, Inertial, IAU_Mars = "" mostly a
    # convenience function to construct a transformation matrix from
    # the coordinate axes
    if frame == 'IAU_MARS':
        frame = ""
    if len(frame) > 0:
        frame = "_"+frame
    return np.transpose(np.array(
        [fits['SpacecraftGeometry'].data[:]['VX_SPACECRAFT'+frame],
         fits['SpacecraftGeometry'].data[:]['VY_SPACECRAFT'+frame],
         fits['SpacecraftGeometry'].data[:]['VZ_SPACECRAFT'+frame]]), axes=[1, 2, 0])


def gettmats(fits, fromframe, toframe):
    # construct transformation matrix from one frame to another
    return np.array([
        np.matmul(tomat, np.linalg.inv(frommat))
        for tomat, frommat in zip(getcoordmats(fits, toframe),
                                  getcoordmats(fits, fromframe))])


def xyoffset(pixelvec, mat, r, x, y):  # need km, km/s
    # project pixel offsets into x,y coordinates
    pv = np.matmul(mat, pixelvec)
    ps = -r-np.dot(r, pv)*pv
    return [np.dot(ps, x), np.dot(ps, y)]


def get_pixel_orbit_offsets(fits):
    # get pixel x,y coordinates in along/cross orbit coordinate frame
    tomats = gettmats(fits, "", "INERTIAL")
    scvecs = fits['SpacecraftGeometry'].data[:]['V_SPACECRAFT_INERTIAL']
    scr = [np.linalg.norm(scv) for scv in scvecs]
    scvels = fits['SpacecraftGeometry'].data[:]['V_SPACECRAFT_RATE_INERTIAL']
    normvecs = [v/np.linalg.norm(v) for v in [np.cross(r, v)
                                              for r, v in zip(scvecs, scvels)]]
    lrlvecs = [lrl_vec(r, v) for r, v in zip(scvecs, scvels)]
    alongvecs = [-np.cross(l, n) for l, n in zip(lrlvecs, normvecs)]
    pixelvecs = np.transpose(
        fits['PixelGeometry'].data[:]['PIXEL_VEC'], axes=[0, 2, 3, 1])
    pvshape = pixelvecs.shape
    print(pvshape)
    pv = np.array([[[xyoffset(pv, mat, r, x, y)
                     for pv in p] for p in s]
                   for s, mat, r, x, y in zip(pixelvecs,
                                              tomats,
                                              scvecs,
                                              alongvecs,
                                              normvecs)])
    return pv


def boundarypoly(pixeloffsets):
    polycoords = np.concatenate((pixeloffsets[:, 0, 0, :],
                                 pixeloffsets[-1, :, 2, :],
                                 pixeloffsets[-1::-1, -1, 3, :],
                                 pixeloffsets[0, -1::-1, 1, :]))
    return polycoords


def expand_poly(poly, scale=1.5):
    center = np.mean(poly, axis=0)
    return scale*(poly-center)+center


def apoorbitplot(orbno, xuv='fuv'):
    orbfilenames = getfiles('*apo*orbit'+str(orbno).zfill(5)+'*'+xuv+'*')
    if len(orbfilenames) == 0:
        print("no apoapse files for orbit "+str(orbno))
        return

    orbfiles = [fits.open(f) for f in orbfilenames]

    # get the pixel coordinates in the preferred frame
    pixeloffsets = np.array([get_pixel_orbit_offsets(f) for f in orbfiles])

    # figure out where the swath boundaries are
    boundarypolys = [boundarypoly(p) for p in pixeloffsets]
    scandir = boundarypolys[0][1, 1]-boundarypolys[0][0, 1]
    nextscanoffset = np.diff([x[0, 1] for x in boundarypolys])
    swathboundaries = [n for
                       n, d in enumerate(scandir*nextscanoffset) if d < 0]
    swathboundaries = np.concatenate(([-1],
                                      swathboundaries,
                                      [len(boundarypolys)-1]))

    # group the files into swaths
    swathboundarygroups = [range(a+1, b+1)
                           for a, b in zip(swathboundaries[:-1],
                                           swathboundaries[1:])]
    boundarypolygroups = [boundarypolys[a+1:b+1]
                          for a, b in zip(swathboundaries[:-1],
                                          swathboundaries[1:])]

    # figure out how much to pad the swaths to show all of the data
    swathymedian = np.array([np.median(np.concatenate(pg)[:, 1])
                             for pg in boundarypolygroups])
    bpolyminmax = np.array([
        [np.min(x[:, 0]), np.max(x[:, 0])]
        for x in [np.array(sorted(np.concatenate(pg),
                                  key=lambda x: np.abs(x[1]-ymed))[:400])
                  for pg, ymed in zip(boundarypolygroups,
                                      swathymedian)]])
    spaces = np.concatenate(
        [[0.], np.cumsum(bpolyminmax[:-1, 1]-bpolyminmax[1:, 0])])

    # determine the pixel count values to plot
    pixelcounts = np.array([gainscaledcounts(f) for f in orbfiles])

    # set up the quantities to plot
    allpixelcoords = np.concatenate([
        np.concatenate([
            np.reshape(
                np.take(a, [0, 2, 3, 1], axis=2)
                + [spaces[n], 0],
                [-1, 4, 2])
            for a in pixeloffsets[grp]])
        for n, grp in enumerate(swathboundarygroups)])
    allpixelcounts = np.concatenate([
        np.concatenate([
            np.reshape(a[:, ::-1], [-1])
            for a in pixelcounts[grp]])
        for n, grp in enumerate(swathboundarygroups)])

    fig, ax = plt.subplots(1, figsize=(6, 6))

    p = PolyCollection(map(expand_poly, allpixelcoords),
                       array=allpixelcounts, linewidths=0)
    p.set_cmap('gray')
    p.set_norm(mpl.colors.PowerNorm(gamma=0.5))
    ax.add_collection(p)

    ax.autoscale_view()
    ax.set_aspect(1)
    ax.set_facecolor('black')
    ax.grid(False)

    [[ax.add_patch(Polygon(b+[spaces[n], 0],
                           closed=True,
                           ec='gray',
                           fill=0,
                           linewidth=0.2))
      for b in bp]
     for n, bp in enumerate(boundarypolygroups)]

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(p, cax=cax)

    plt.show()
