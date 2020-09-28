import matplotlib
import matplotlib.pyplot as plt
from cartopy.feature import ShapelyFeature
from .plot_defaults import *
import numpy as np

def longitude_mod(lon):
    #keeps longitudes in the range 0-360
    return (lon+180)%360-180

class draw_shadow(ShapelyFeature):
    def __init__(self, subsolar_lon, subsolar_lat, delta=0.1, refraction=0,
                     color="k", alpha=0.5, **kwargs):  
        #based on nightshade from cartopy, available here: https://github.com/SciTools/cartopy/blob/master/lib/cartopy/feature/nightshade.py

        # Returns the Greenwich hour angle,
        # need longitude (opposite direction)
        lat = subsolar_lat
        lon = subsolar_lon
        pole_lon = lon
        if lat > 0:
            pole_lat = -90 + lat
            central_lon = 180
        else:
            pole_lat = 90 + lat
            central_lon = 0

        import cartopy.crs as ccrs

        rotated_pole = ccrs.RotatedPole(pole_latitude=pole_lat,
                                        pole_longitude=pole_lon,
                                        central_rotated_longitude=central_lon)

        npts = int(180/delta)
        x = np.empty(npts*2)
        y = np.empty(npts*2)

        # Fill latitudes up and then down
        y[:npts] = np.linspace(-(90+refraction), 90+refraction, npts)
        y[npts:] = y[:npts][::-1]

        # Solve the generalized equation for omega0, which is the
        # angle of sunrise/sunset from solar noon
        omega0 = np.rad2deg(np.arccos(np.sin(np.deg2rad(refraction)) /
                                      np.cos(np.deg2rad(y))))

        # Fill the longitude values from the offset for midnight.
        # This needs to be a closed loop to fill the polygon.
        # Negative longitudes
        x[:npts] = -(180 - omega0[:npts])
        # Positive longitudes
        x[npts:] = 180 - omega0[npts:]

        kwargs.setdefault('facecolor', color)
        kwargs.setdefault('alpha', alpha)

        from shapely.geometry import Polygon
        geom = Polygon(np.column_stack((x, y)))

        return super().__init__(
            [geom], rotated_pole, **kwargs)


    
def draw_sza_contours(map_ax, subsolar_lon, subsolar_lat, szas=np.arange(0,180,10)):
    sza_contour_color='#222222'
    sza_contour_alpha=0.5
    sza_contour_linewidth=0.1
    
    import cartopy.crs as ccrs
    rotated_pole = ccrs.RotatedPole(pole_latitude=subsolar_lat,
                                    pole_longitude=subsolar_lon)
    longrid=[]
    latgrid=np.arange(-90,90,10)
    gl = map_ax.gridlines(xlocs=longrid,ylocs=latgrid,linewidth=sza_contour_linewidth,color=sza_contour_color,alpha=sza_contour_alpha,crs=rotated_pole,zorder=1)
    
    circle_radius=7.5*sza_contour_linewidth
    subsolar_dot = matplotlib.patches.Circle([subsolar_lon,subsolar_lat],radius=circle_radius,fc=sza_contour_color,ec=None,alpha=sza_contour_alpha,transform=ccrs.PlateCarree(),zorder=1)
    map_ax.add_patch(subsolar_dot)
   
    antisolar_dot = matplotlib.patches.Circle([longitude_mod(subsolar_lon+180),-subsolar_lat],radius=circle_radius,color=sza_contour_color,ec=None,alpha=sza_contour_alpha,transform=ccrs.PlateCarree(),zorder=1)
    map_ax.add_patch(antisolar_dot)


def draw_map(fig, map_ax, orbmiddleutc):
    import cartopy.crs as ccrs
    import spiceypy as spice
    
    # we're plotting the orbit in MSO (transformed to IAU_MARS at orbmiddleutc) 
    # this makes sure to match the shaded outline on the map, but not the geographic map.
    # this compromise projection allows high confidence in the location of the orbit 
    # in solar coordinates and medium confidence in geographic coordinates, 
    # plus the orbit doesn't move that much
    
    #get the location of the Sun in IAU_MARS
    myet = spice.str2et(orbmiddleutc)
    to_iau_mat=spice.pxform('MAVEN_MSO','IAU_MARS',myet)
    sunpos=np.matmul(to_iau_mat,[1,0,0])
    subsolar_latitude = 90-np.rad2deg(np.arccos(sunpos[2]))
    subsolar_longitude = np.rad2deg(np.arctan2(sunpos[1],sunpos[0]))
    
    central_longitude=subsolar_longitude
    
    position = map_ax.get_position()
    map_ax.remove()
    map_projection = ccrs.PlateCarree(central_longitude=central_longitude)
    map_ax = fig.add_axes(position, projection=map_projection)
    style_axis(map_ax)

    
    #
    #draw the surface of Mars
    #
    
    surface_extent = (0, 360, -90, 90)
    surface_img = plt.imread('/home/mike/Documents/Mars/marssurface_2.jpg')
    map_ax.imshow(surface_img, origin='upper', extent=surface_extent, transform=ccrs.PlateCarree(),zorder=0)
    longrid=np.arange(-180,180,30)
    latgrid=np.linspace(-90,90,7)
    gl = map_ax.gridlines(xlocs=longrid,ylocs=latgrid,linewidth=0.1,color='#222222',alpha=0.5,crs=ccrs.PlateCarree(),zorder=1)
    
    longridticks=np.arange(-180,180,60)
    longridticklabels=[np.format_float_positional(s,precision=0,trim='-')+"°E" for s in [a+360 if a<0 else a for a in longridticks]]
    map_ax.set_xticks(longridticks,crs=ccrs.PlateCarree())
    map_ax.set_xticklabels(longridticklabels)
    
    latgridticks=np.linspace(-90,90,7)
    latgridticklabels=[np.format_float_positional(-a,precision=0,trim='-')+"°S" if a<0 else np.format_float_positional(a,precision=0,trim='-')+"°N" for a in latgridticks]
    latgridticklabels[3]="0°"
    map_ax.set_yticks(latgridticks,crs=ccrs.PlateCarree())
    map_ax.set_yticklabels(latgridticklabels)
    lat_ticklabels=map_ax.get_yticklabels()
    lat_ticklabels[0].set_va('bottom')
    lat_ticklabels[-1].set_va('top')
    
    map_ax.tick_params(axis='both', which='both', length=0, pad=1)

    #draw the shadowed region
    map_ax.add_feature(draw_shadow(subsolar_longitude,subsolar_latitude,zorder=2))
    
    #add a note about the longitudes
    map_ax.text(0,1.01,'Longitudes shown for the midpoint of the times listed above. Orbit geometry relative to shadow is correct.',
                transform=map_ax.transAxes,color='#444444',size=0.75*fontsize,clip_on=False,va='bottom',ha='left')
    
    #draw SZA contours
    draw_sza_contours(map_ax,subsolar_longitude,subsolar_latitude)
    
    
    #
    #draw the orbit
    #
    #for the given time, we determine the orbit length
    maven_state=spice.spkezr('MAVEN',myet,'MAVEN_MME_2000','NONE','MARS')[0]
    marsmu=spice.bodvrd('MARS','GM',1)[1][0]
    maven_elements=spice.oscltx(maven_state,myet,marsmu)
    orbit_period=1.01*maven_elements[-1]

    #get the orbit state vectors
    orbit_subdivisions=2000
    etlist=myet-orbit_period/2+orbit_period*np.linspace(0,1,num=orbit_subdivisions)
    statelist=np.array(spice.spkezr('MAVEN',etlist,'MAVEN_MSO','NONE','MARS')[0])
    
    #transform to the variables we want
    maven_radius=np.array([np.linalg.norm(s) for s in statelist[:,:3]])
    normpos_mso=np.array([s/r for s,r in zip(statelist[:,:3],maven_radius)])
    normpos_iau=np.array([np.matmul(to_iau_mat,p) for p in normpos_mso]) # transform 
    maven_lon=np.rad2deg(np.arctan2(normpos_iau[:,1],normpos_iau[:,0]))
    maven_lat=90-np.rad2deg(np.arccos(normpos_iau[:,2]))
    
    #split the array based on when we cross the meridian +180/-180
    branchsplits=[i+1 for i,d in enumerate(maven_lon[1:]-maven_lon[:-1]) if abs(d)>10]
    patch_across_branch=[np.arange(b-2,b+3) for b in branchsplits]
    branchsplits=np.concatenate([[0],branchsplits,[orbit_subdivisions]])
    branchsplits=[np.arange(branchsplits[i],branchsplits[i+1]) for i in range(len(branchsplits)-1)]

    
    #make a line plot
    orbitcolor=np.array([222,45,38])/255 # a nice red
    orbitcolor=tuple(orbitcolor)
    linewidths=2*(maven_radius-0.5*3395)/(4*3395)
    from matplotlib.collections import LineCollection
    for subset in branchsplits:
        points = np.array([maven_lon[subset],maven_lat[subset]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, linewidths=linewidths[subset], alpha=1,color=orbitcolor,transform=ccrs.PlateCarree(),zorder=8)
        map_ax.add_collection(lc)   
        map_ax.add_collection(lc)
        map_ax.add_collection(lc)
    #patch across the branch cut    
    for subset in patch_across_branch:
        plot_lon=[a+360 if a<0 else a for a in maven_lon[subset]]
        points = np.array([plot_lon,maven_lat[subset]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, linewidths=linewidths[subset], alpha=1, color=orbitcolor,transform=ccrs.PlateCarree(),zorder=8)
        map_ax.add_collection(lc)
        map_ax.add_collection(lc)
        map_ax.add_collection(lc)
    
    #place periapse and apoapse on the plot
    import matplotlib.patches as mpatches
    periidx=np.argmin(maven_radius)
    peripos=[maven_lon[periidx],maven_lat[periidx]]
    
    peri = mpatches.CirclePolygon(peripos, 5, resolution=4, transform=ccrs.PlateCarree(), fc=orbitcolor, lw=0,zorder=10)
    map_ax.add_patch(peri)
    map_ax.text(*peripos, 'P',transform=ccrs.PlateCarree(),color='k',ha='center',va='center_baseline',size=3,zorder=10) 
    
    apoidx=np.argmax(maven_radius)
    apopos=[maven_lon[apoidx],maven_lat[apoidx]]

    apo = mpatches.CirclePolygon(apopos, 5, resolution=4, transform=ccrs.PlateCarree(), fc=orbitcolor,lw=0,zorder=10)
    map_ax.add_patch(apo)
    map_ax.text(*apopos, 'A',transform=ccrs.PlateCarree(),color='k',ha='center',va='center',size=3,zorder=10) 
    
    #add an arrow to the orbit
    peri_vec = normpos_mso[periidx]
    orbit_normal_vec = np.mean([v/np.linalg.norm(v) for v in np.cross(normpos_mso[:-1],normpos_mso[1:])],axis=0)
    cross_orbit_vec = np.cross(orbit_normal_vec,peri_vec)
    
    arrow_loc_vec = 0.5*(peri_vec+cross_orbit_vec)
    arrowidx=np.argmax([np.dot(arrow_loc_vec,p) for p in normpos_mso])
    
    arrowloc=np.array([maven_lon[arrowidx],maven_lat[arrowidx]])
    arrowdir_index_delta=10
    arrowdir=np.array([maven_lon[arrowidx+arrowdir_index_delta],maven_lat[arrowidx+arrowdir_index_delta]])
    
    arrow_width=3.5
    arrow_length=1.5*arrow_width
    arrow = map_ax.annotate('',
                            xytext=arrowloc,
                            xy=arrowdir,
                            xycoords=ccrs.PlateCarree()._as_mpl_transform(map_ax),
                            arrowprops=dict(facecolor=orbitcolor,edgecolor='none',width=0,headwidth=arrow_width,headlength=arrow_length))

    
    #draw the observations from this orbit
    #this should be a seperate routine that takes the axis object
    
    return map_ax, to_iau_mat
