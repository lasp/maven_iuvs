import numpy as np


def remove_pixel_nans(pixel_x, pixel_y, brightness):
    #  sometimes pixel_vec is missing for the first or last integration, we need to find and remove these
    bad_indices = np.array([idx for idx,vals in enumerate(zip(pixel_x, pixel_y)) if np.any(np.isnan(vals))])

    if len(bad_indices)!=0:
        bad_indices=np.concatenate([[-1],bad_indices,[len(pixel_x)]])
        bad_indices=np.unique(bad_indices)

        #  find the largest range of continuous data 
        max_continuous_range_index = ((bad_indices[1:]-1)-(bad_indices[:-1]+1)).argmax()
        start_good_idx=bad_indices[max_continuous_range_index]+1
        end_good_idx  =bad_indices[max_continuous_range_index+1]

        pixel_x    = pixel_x[start_good_idx:end_good_idx]
        pixel_y    = pixel_y[start_good_idx:end_good_idx]
        brightness = brightness[start_good_idx:end_good_idx-1]

    return pixel_x, pixel_y, brightness
