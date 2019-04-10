import numpy as np
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection, PolyCollection
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.io import fits


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
