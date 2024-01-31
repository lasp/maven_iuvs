import numpy as np

def pixel_swath_quantities(myfits, label=None,
                           pixel_x=None, pixel_y=None,
                           orbit_towards_apo_vec=None): 
    from .get_lya_orbit_h5 import get_lya_orbit_h5
    from maven_iuvs.integration import fit_line
    
    if label is not None:
        brightness=get_lya_orbit_h5(myfits, label)
    else:
        brightness, brightnessunc = fit_line(myfits, 121.56)
    
    int_num_increases_along_apo=True
    
    if pixel_y==None:
        pixel_y=myfits['PixelGeometry'].data['PIXEL_CORNER_MRH_ALT'][:,:,[0,1,2,3]]
    
    #make x-coordinates for each pixel in swath space
    if pixel_x==None:
        pixelindices=np.indices(pixel_y.shape)
        pixel_x=1-(pixelindices[1]+0.5+(pixelindices[2]//2-0.5))/pixel_y.shape[1]
    
    #average pixel corners
    from maven_iuvs.geometry import pixelcorner_avg
    pixel_x,pixel_y=pixelcorner_avg(pixel_x,pixel_y)
    
    return pixel_x, pixel_y, brightness
