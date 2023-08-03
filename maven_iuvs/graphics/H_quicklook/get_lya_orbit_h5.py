import os
import numpy as np

def get_lya_orbit_h5_filename(orbit_number):
    from maven_iuvs.user_paths import lya_fit_vals_dir_h5

    orbit_number = str(orbit_number).zfill(5)

    # determine where to save the lya fit values
    save_file_name = 'orbit'+orbit_number+"_lya_fit_values.h5"
    save_file_subdir = 'orbit'+str(((int(orbit_number)//100)*100)).zfill(5)
    save_file_subdir = os.path.join(lya_fit_vals_dir_h5, save_file_subdir)
    save_file_name = os.path.join(save_file_subdir, save_file_name)

    return save_file_name


def get_lya_orbit_h5(myfits, label):
    # get lyman alpha brightness values for the corresponding file and
    # label, reading from a per-orbit HDF5 file that has a dataset
    # corresponding to the passed file label

    import os
    import h5py
    from maven_iuvs.geometry import get_pixel_vec_mso
    from maven_iuvs.search import get_solar_lyman_alpha
    from maven_iuvs.integration import fit_line
    import astropy.io.fits as fits

    import astropy.io.fits as fits
    if type(myfits)!=fits.hdu.hdulist.HDUList:
        myfits=fits.open(myfits)

    # get the filename of the FITS file without extensions or paths
    fits_file_name = myfits['Primary'].header['FILENAME']
    fits_file_name = fits_file_name.replace('.fits', '').replace('.gz', '')

    # determine where to save the lya fit values
    save_file_orbit = fits_file_name.split('orbit')[1][:5]
    save_file_name = get_lya_orbit_h5_filename(save_file_orbit)

    if not os.path.exists(os.path.dirname(save_file_name)):
        os.makedirs(os.path.dirname(save_file_name))

    f = h5py.File(save_file_name, 'a')
    if label in f.keys():
        grp = f[label]
        # the data is already stored in the file, read it
        if grp['l1b_filename'][0] == np.string_(fits_file_name):
            lyavals = grp['IUVS Lyman alpha'][...]
        else:
            raise Exception('filename does not match expected ' +
                            'filename in get_lya_orbit_h5')
    else:
        # the data is not stored in this file yet, get it and write it
        grp = f.create_group(label)

        # store relevant ancillary info

        # values shared for the whole observation
        grp.create_dataset('l1b_filename', data=[np.string_(fits_file_name)])
        dset = grp.create_dataset('Solar Lyman alpha',
                                  data=get_solar_lyman_alpha(myfits))
        dset.attrs['units'] = 'ph/cm2/s/nm'
        # get Mars Ecliptic coordinates
        import spiceypy as spice
        marspos, ltime = spice.spkezr('MARS',
                                      myfits['Integration'].data['ET'][0],
                                      'ECLIPJ2000', 'NONE', 'SSB')
        marspos = marspos[:3]/1.496e8  # convert to AU from km
        marssundist = np.linalg.norm(marspos)
        dset = grp.create_dataset('Mars Ecliptic Coordinates',
                                  data=marspos)
        dset.attrs['units'] = 'AU'
        dset = grp.create_dataset('Mars-Sun Distance',
                                  data=marssundist)
        dset.attrs['units'] = 'AU'
        # now the IUVS Lyman alpha brightness

        lyavals, lyaunc = fit_line(myfits, 121.56)  # get_lya(myfits)
        dset = grp.create_dataset('IUVS Lyman alpha', data=lyavals)
        dset.attrs['units'] = 'kR'

        # ancillary data for the IUVS integrations
        dset = grp.create_dataset('Uncertainty in IUVS Lyman alpha',
                                  data=lyaunc)
        dset.attrs['units'] = 'kR'
        grp.create_dataset('ET', data=myfits['Integration'].data['ET'])
        grp.create_dataset('UTC', data=np.array([np.string_(t)
                                                 for t in myfits['Integration'].data['UTC']]))
        grp.create_dataset('MSO Pixel vector along integration center',
                           data=get_pixel_vec_mso(myfits)[:, :, 4])
        dset = grp.create_dataset('MSO Spacecraft position',
                                  data=myfits['SpacecraftGeometry'].data['V_SPACECRAFT_MSO'])
        dset.attrs['units'] = 'km'
        grp.create_dataset('Pixel RA',
                           data=myfits['PixelGeometry'].data['PIXEL_CORNER_RA'][:, :, 4])
        grp.create_dataset('Pixel Dec',
                           data=myfits['PixelGeometry'].data['PIXEL_CORNER_DEC'][:, :, 4])
        grp.create_dataset('Tangent point SZA',
                           data=myfits['PixelGeometry'].data['PIXEL_SOLAR_ZENITH_ANGLE'])
        grp.create_dataset('Tangent point local time',
                           data=myfits['PixelGeometry'].data['PIXEL_LOCAL_TIME'])
        grp.create_dataset('Tangent point Latitude',
                           data=myfits['PixelGeometry'].data['PIXEL_CORNER_LAT'][:, :, 4])
        grp.create_dataset('Tangent point Longitude',
                           data=myfits['PixelGeometry'].data['PIXEL_CORNER_LON'][:, :, 4])
        dset = grp.create_dataset('Tangent point altitude',
                                  data=myfits['PixelGeometry'].data['PIXEL_CORNER_MRH_ALT'][:, :, 4])
        dset.attrs['units'] = 'km'

    f.close()

    return lyavals
