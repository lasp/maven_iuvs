import numpy as np

def pixel_swath_quantities(myfits,pixel_x=None,pixel_y=None,orbit_towards_apo_vec=None):  
    from ..integration import get_lya
    brightness=get_lya(myfits)
    
    int_num_increases_along_apo=True
    
    if pixel_y==None:
        pixel_y=myfits['PixelGeometry'].data['PIXEL_CORNER_MRH_ALT'][:,:,[0,1,2,3]]
    
    #make x-coordinates for each pixel in swath space
    if pixel_x==None:
        pixelindices=np.indices(pixel_y.shape)
        pixel_x=1-(pixelindices[1]+0.5+(pixelindices[2]//2-0.5))/pixel_y.shape[1]
    
    #average pixel corners
    from ..geometry import pixelcorner_avg
    pixel_x,pixel_y=pixelcorner_avg(pixel_x,pixel_y)
    
    return pixel_x, pixel_y, brightness
