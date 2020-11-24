import numpy as np
from scipy.io.idl import readsav
from scipy.interpolate import interp1d
import os
import matplotlib.pyplot as plt
from .paths import anc_dir
import warnings

fuv_lsf_seven_segment = readsav(os.path.join(anc_dir, 'mvn_iuv_psf_fuv_2017APR24.sav'))

fuv_lsf_seven_segment_interp = []
for k in fuv_lsf_seven_segment.keys():
    # shift the LSF x coordinates to roughly match the instrument wavelength scale in nm
    waves = 5 * \
        (np.arange(len(fuv_lsf_seven_segment[k])
        )/len(fuv_lsf_seven_segment[k])-0.5)
    dwaves = np.mean(np.diff(waves))  # new wavelength spacing

    LSF = fuv_lsf_seven_segment[k]

    interp = interp1d(x=waves,
                      y=LSF,
                      bounds_error=False,
                      fill_value=0.)

    fuv_lsf_seven_segment_interp.append(interp)

# from sonal, derived from Lyman alpha (so technicallly only valid there)
seven_segment_flatfield = [0.976, 1.018, 1.018,  1.022, 1.004, 0.99, 0.953]
    
def get_lsf_from_bins(spatial_binning):
    #get the lsf for each spatial bin. 
    #spatial_binning is a list of the start pixels of the spatial bins
    #last value in the list should be end of last spatial bin+1
    
    cruise_lsf=np.load(os.path.join(anc_dir, 'cruise_lsf_23Sep2020.npy'))
    spapix=np.arange(76,916,4)#these are the start pixels of the LSF spatial bins
                              #end pixels are spapix[1:]-1
    
    lsf=np.zeros((len(spatial_binning)-1,cruise_lsf.shape[1]))
    for idx in range(len(spatial_binning)-1):
        this_lsf=np.sum(cruise_lsf[  (spatial_binning[idx]<spapix+4) 
                                   & (spapix<spatial_binning[idx+1])],
                        axis=0)
        this_lsf = this_lsf/np.sum(this_lsf)
        lsf[idx,:]=this_lsf
    
    return lsf

def get_lsf(myfits):
    spalo=myfits['Binning'].data['SPAPIXLO'][0]
    spalo=np.append(spalo,myfits['Binning'].data['SPAPIXHI'][0][-1]+1)
    return get_lsf_from_bins(spalo)

def get_lsf_interp(myfits):
    lsf=get_lsf(myfits)

    lsf_interp = [None]*len(lsf)
    for i,l in enumerate(lsf):
        # shift the LSF x coordinates to roughly match the instrument wavelength scale in nm
        waves = 7.5 * (np.arange(len(l))/len(l)-0.5)

        interp = interp1d(x=waves,
                          y=l,
                          bounds_error=False,
                          fill_value=0.)

        lsf_interp[i]=interp

    return lsf_interp

def get_lya(myfits):
    import os
    from .paths import lya_fit_vals_dir
    import astropy.io.fits as fits

    # get the filename of the FITS file without extensions or paths
    if type(myfits)!=fits.hdu.hdulist.HDUList:
        fits_file_name=os.path.basename(myfits)
    else:
        fits_file_name=myfits['Primary'].header['FILENAME']
    fits_file_name=fits_file_name.replace('.fits','').replace('.gz','')

    # determine where to save the lya fit values
    save_file_name=fits_file_name+"_lya_fit_values.npy"
    save_file_orbit=save_file_name.split('orbit')[1][:5]
    save_file_subdir='orbit'+str(((int(save_file_orbit)//100)*100)).zfill(5)
    save_file_subdir=os.path.join(lya_fit_vals_dir,save_file_subdir)
    save_file_name=os.path.join(save_file_subdir,save_file_name)

    if not os.path.exists(save_file_subdir):
        os.makedirs(save_file_subdir)

    if os.path.exists(save_file_name):
        return np.load(save_file_name)
    else:
        lyavals, lyaunc=fit_line(myfits,121.56)
        np.save(save_file_name,lyavals)
        return lyavals


def get_solar_lyman_alpha(myfits):
    import datetime
    from scipy.io.idl import readsav
    from .paths import euvm_orbit_average_filename
    import spiceypy as spice

    import astropy.io.fits as fits
    if type(myfits)!=fits.hdu.hdulist.HDUList:
        myfits=fits.open(myfits)

    # load the EUVM data
    euvm = readsav(euvm_orbit_average_filename)
    euvm_datetime = [datetime.datetime.fromtimestamp(t)
                     for t in euvm['mvn_euv_l2_orbit'].item()[0]]

    euvm_lya = euvm['mvn_euv_l2_orbit'].item()[2][2]
    euvm_mars_sun_dist = euvm['mvn_euv_l2_orbit'].item()[5]

    # get the time of the FITS file
    iuvs_mean_et = np.mean(myfits['Integration'].data['ET'])
    iuvs_datetime = spice.et2datetime(iuvs_mean_et)
    # we need to remove the timezone info to compare with EUVM times
    iuvs_datetime = iuvs_datetime.replace(tzinfo=None)

    # interpolate the EUVM data if it's close enough in time
    euvm_idx = np.searchsorted(euvm_datetime, iuvs_datetime) - 1
    if (euvm_datetime[euvm_idx] > iuvs_datetime - datetime.timedelta(days=2)
        and
        euvm_datetime[euvm_idx+1] < iuvs_datetime + datetime.timedelta(days=2)):
        interp_frac = ((iuvs_datetime-euvm_datetime[euvm_idx])
                       /
                       (euvm_datetime[euvm_idx+1]-euvm_datetime[euvm_idx]))
        lya_interp  = interp_frac*(euvm_lya[euvm_idx+1] - euvm_lya[euvm_idx]) + euvm_lya[euvm_idx]
        dist_interp = interp_frac*(euvm_mars_sun_dist[euvm_idx+1] - euvm_mars_sun_dist[euvm_idx]) + euvm_mars_sun_dist[euvm_idx]
        dist_interp = dist_interp / 1.496e8  # convert dist_interp to AU

        # this is the band-integrated value measured at Mars, we need to
        # convert back to Earth, then get the line center flux using the
        # power law relation of Emerich...
        lya_interp *= dist_interp**2
        # this is now in W/m2 at Earth. We need to convert to ph/cm2/s
        phenergy = 1.98e-25/(121.6e-9)  # energy of a lyman alpha photon in J
        lya_interp /= phenergy
        lya_interp /= 1e4  # convert to /cm2
        # we're now in ph/cm2/s

        # Use the power law relation of Emerich:
        lya_interp = 0.64*((lya_interp/1e11)**1.21)
        lya_interp *= 1e12
        # we're now in ph/cm2/s/nm

        # convert back to Mars
        lya_interp /= dist_interp**2
    else:
        lya_interp = np.nan

    return lya_interp


def get_lya_orbit_h5_filename(orbit_number):
    from .paths import lya_fit_vals_dir_h5

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
    from .geometry import get_pixel_vec_mso
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


def get_muv_contamination_templates(myfits_fuv):
    # returns the partner MUV FITS object
    # and an n_spa x n_spe array of MUV contamination
    # templates from the corresponding MUV observation

    #find the filename of the matching muv file
    fuv_filename=myfits_fuv.filename()
    fuv_dir=os.path.dirname(fuv_filename)
    muv_filename=os.path.basename(fuv_filename).replace('fuv','muv')
    muv_filename=os.path.join(fuv_dir,muv_filename)
    import astropy.io.fits as fits
    myfits_muv=fits.open(muv_filename)

    #get the MUV contamination templates
    #To be ported from IDL by Sonal
    n_int,n_spa,n_spe = myfits_muv['Primary'].data.shape
    muv_contamination_templates = np.zeros((n_spa,n_spe))

    return myfits_muv, muv_contamination_templates
    

def fit_line(myfits, l0, calibrate=True, flatfield_correct=True, plot=False, correct_muv=False):
    import astropy.io.fits as fits
    if type(myfits)!=fits.hdu.hdulist.HDUList:
        myfits=fits.open(myfits)

    if correct_muv:
        warnings.warn('correct_muv not implemented, this flag does not change output values')
        #get the muv counterpart of this observation, if it exists
        try:
            myfits_muv, muv_contamination_templates = get_muv_contamination_templates(myfits)
        except FileNotFoundError:
            print('no matching MUV observation found, cannot correct MUV')
            correct_muv=False

    if flatfield_correct:
        periapse_spatial_binning=[ 89, 204, 319, 434, 549, 664, 779, 894] #spatial binning used in periapse
        myfits_spatial_binning = np.concatenate([myfits['Binning'].data['SPAPIXLO'][0],
                                                 [myfits['Binning'].data['SPAPIXHI'][0][-1]+1]])

        if np.array_equal(myfits_spatial_binning,periapse_spatial_binning):
            #this is a periapse-binned file, we can use an empirical flat field from periapse
            flatfield = seven_segment_flatfield
        else:
            warnings.warn('Binning is not periapsis--- using a very rough FUV flatfield.')
            slit_flatfield = np.load(os.path.join(anc_dir, 'bad_flatfield_23Sep2020.npy'))
            flatfield = np.array([np.mean(slit_flatfield[p0:p1]) for p0,p1 in zip(myfits_spatial_binning[:-1],
                                                                                  myfits_spatial_binning[1:])])

        
    filedims = myfits['Primary'].shape  
    n_int = filedims[0]
    n_spa = filedims[1]

    lsf=get_lsf_interp(myfits)

    linevalues = np.zeros((n_int, n_spa))
    lineunc    = np.zeros((n_int, n_spa))
    lineDNmax = 0

    if plot:
        from .graphics import line_fit_plot
        myplot=line_fit_plot(myfits,n_int,n_spa,correct_muv) 
    #now get the data (and maybe put it on the plot)
    for iint in range(n_int):
        if plot:
            if correct_muv:
                myplot.plot_detector(myfits,iint,myfits_muv=myfits_muv)
            else:
                myplot.plot_detector(myfits,iint)
        for ispa in range(n_spa):
            # print(str(iint).rjust(5),'  ',str(ispa).rjust(5), end='\r')

            waves = myfits['Observation'].data['WAVELENGTH'][0, ispa]
            DN = myfits['detector_dark_subtracted'].data[iint, ispa]
            DN_unc = myfits['Random_dn_unc'].data[iint, ispa]
            if correct_muv:
                muv = muv_contamination_templates[ispa]
            else:
                muv=np.zeros_like(DN)
            
            # subset the data to be fitted to the vicinity of the spectral line
            d_lambda = 2.5
            fitwaves, fitDN, fitDN_unc, fitmuv = np.transpose([[w, d, du, m]
                                                               for w, d, du, m in zip(waves, DN, DN_unc, muv)
                                                               if w > l0-d_lambda and w < l0+d_lambda])
            lineDNmax = np.max([lineDNmax, np.max(fitDN)]) # for plotting

            # guess what the fit parameters should be
            backguess = (np.median(fitDN[0:3])+np.median(fitDN[-3:-1]))/2
            slopeguess = (np.median(fitDN[-3:-1])-np.median(fitDN[0:3]))/(fitwaves[-1]-fitwaves[0])
            DNguess = np.sum(fitDN) - backguess * len(fitwaves)

            # define the line spread function for this spatial element
            def this_spatial_element_lsf(x, scale=5e6, dl=1, x0=0, s=0, b=0, muv_background_scale=0,
                                         background_only=False):
                unitlsf = lsf[ispa](dl*x-x0)
                unitlsf /= np.sum(unitlsf)

                lineshape = s*(x-x0) + b
                
                if correct_muv:
                    lineshape += muv_background_scale*fitmuv[ispa]

                if not background_only:
                    lineshape += scale*unitlsf 

                return lineshape

            # do the fit
            try:
                from scipy.optimize import curve_fit

                parms_bounds = ([     0, 0.5, l0-d_lambda, -np.inf, -np.inf],
                                [np.inf, 2.0, l0+d_lambda,  np.inf,  np.inf])
    
                if correct_muv:
                    parms_bounds[0].append(0)
                    parms_bounds[1].appens(np.inf)

                parms_guess = [DNguess, 1.0, l0, slopeguess, backguess]

                
                if correct_muv:
                    #we need to append a guess for the MUV background
                    parms_guess.append(0)

                # ensure that the guesses are in bounds
                for i, p in enumerate(parms_guess):
                    if p < parms_bounds[0][i]:
                        parms_guess[i] = parms_bounds[0][i]
                    if p > parms_bounds[1][i]:
                        parms_guess[i] = parms_bounds[1][i]
    
                fit = curve_fit(this_spatial_element_lsf, fitwaves, fitDN,
                                p0=parms_guess,
                                sigma=fitDN_unc, absolute_sigma=True,
                                bounds=parms_bounds)

                thislinevalue = fit[0][0] # keep only the total DN in the line
                thislineunc   = np.sqrt(fit[1][0,0])
            except RuntimeError:
                fit = [(DNguess, 1.0, l0, slopeguess, backguess)]
                thislinevalue = np.nan
                thislineunc = np.nan

            DN_fit = thislinevalue
            DN_unc = thislineunc
                
            # return the requested values
            if flatfield_correct:
                thislinevalue /= flatfield[ispa]
                thislineunc   /= flatfield[ispa]

            if calibrate:
                cal_factor = get_line_calibration(myfits, l0)
                thislinevalue *= cal_factor
                thislineunc   *= cal_factor

            linevalues[iint, ispa] = thislinevalue
            lineunc[iint, ispa] = thislineunc
                
            if plot:
                myplot.plot_line_fits(iint, ispa,
                                      fitwaves,
                                      fitDN, fitDN_unc,
                                      this_spatial_element_lsf(fitwaves, *fit[0],background_only=True), this_spatial_element_lsf(fitwaves, *fit[0]),
                                      DNguess,
                                      DN_fit, DN_unc,
                                      thislinevalue, thislineunc)

    if plot:
        myplot.finish_plot(lineDNmax, linevalues)
        return linevalues, lineunc, myplot.fig
    else:
        return linevalues, lineunc

    
def mcp_dn_to_volt(dn):
    c0 = -1.83
    c1 = 0.244
    return dn*c1+c0


def mcp_volt_to_gain(volt, channel="FUV"):
    v0 = 900.0
    L0 = 2.560
    L1 = -0.0025
    LV = 625.0
    if channel == "MUV":
        G0 = 392.0
        alpha = 0.0185
    elif channel == "FUV":
        G0 = 494.0
        alpha = 0.0196
        A = G0/((L0+L1*v0)*np.exp(alpha*v0))
    if volt > LV:
        gain = A*(L0+L1*volt)*np.exp(alpha*volt)
    else:
        gain = A*np.exp(alpha*volt)

    return gain


def get_line_calibration(myfits,
                         line_center):

    spatial_bins_maxpix = myfits['Binning'].data[0]['SPAPIXHI']
    spatial_bins_minpix = myfits['Binning'].data[0]['SPAPIXLO']
    spatial_bins_npix = (spatial_bins_maxpix + 1) - spatial_bins_minpix
    if not np.all(spatial_bins_npix == spatial_bins_npix[0]):
        raise Exception("spatial bins are not identical widths")
    spatial_bins_npix = spatial_bins_npix[0]

    spectral_bins_maxpix = myfits['Binning'].data[0]['SPEPIXHI']
    spectral_bins_minpix = myfits['Binning'].data[0]['SPEPIXLO']
    spectral_bins_npix = (spectral_bins_maxpix + 1) - spectral_bins_minpix
    if not np.all(spectral_bins_npix == spectral_bins_npix[0]):
        raise Exception("spectral bins are not identical widths")
    spectral_bins_npix = spectral_bins_npix[0]

    if myfits['Observation'].data['DUTY_CYCLE'][0] != 1.0:
        raise Exception(
            "Duty cycle != 1.0 , line calibration routine cannot handle this")

    # check if any portion falls off the slit:
    slit_pix_min = 77
    slit_pix_max = 916

    if (np.any(spatial_bins_maxpix > slit_pix_max) |
        np.any(spatial_bins_minpix < slit_pix_min)):
        raise Exception("some spatial bins fall outside the airglow slit!")

    mcp_volt_dn = myfits['Engineering'].data[0]['MCP_GAIN']
    inttime = myfits['Engineering'].data[0]['INT_TIME']/1000.    # ms->s
    xuv = myfits['Observation'].data['CHANNEL'][0]

    pixel_width = 0.023438  # pixel width in mm
    pixel_height = pixel_width
    focal_length = 100.  # telescope focal length in mm

    slit_width = 0.1  # mm, airglow slit width

    pixel_omega = pixel_height*slit_width/focal_length/focal_length

    # Dispersion is not needed for calibrating an individual line
    fuv_dispersion = 0.08134  # nm/pix, dispersion of the FUV channel
    muv_dispersion = 0.16535  # nm/pix, dispersion of the MUV channel

    # load IUVS sensitivity curve
    effective_area = readsav(
        '/home/mike/Documents/MAVEN/IUVS/iuvs-itf-sw/anc/cal_data/sensitivity update 6_9_14.sav')

    # get channel specific parms
    adjust_cal_factor = 1.0
    wavelength_shift = 0.0
    if xuv == 'FUV':
        dispersion = fuv_dispersion
        line_effective_area = np.interp(line_center,
                                        effective_area['waveg'] /
                                        10.-wavelength_shift,
                                        # /10 here Angstroms -> nm
                                        effective_area['sens_g_star'])  # cm2
        adjust_cal_factor = 1.27  # we decided to adjust the FUV by
        # this factor in 2014 to accommodate
        # airglow models
    elif xuv == 'MUV':
        dispersion = muv_dispersion
        wavelength_shift = 7.0  # shift MUV calibration by 7 nm redward
        # to correct for poor wavelength
        # calibration in the cruise data
        # derived calibration
        line_effective_area = np.interp(line_center,
                                        effective_area['wavef'] /
                                        10.-wavelength_shift,
                                        # /10 here Angstroms -> nm
                                        effective_area['sens_f_star'])  # cm2
    else:
        raise Exception("channel is not FUV or MUV")

    bin_omega = pixel_omega*spatial_bins_npix  # sr / spatial bin
    bin_dispersion = dispersion*spectral_bins_npix  # nm / spectral bin

    gain = mcp_volt_to_gain(mcp_dn_to_volt(mcp_volt_dn))

    kr = 1e9/(4*np.pi)  # photon/kR
    calfactor = gain * inttime * kr * line_effective_area * \
        bin_omega  # DN/photon * s * photon/kR * cm2 * sr
    # [DN / cal_factor] = 10^9 ph /cm2/s/sr = kR

    calfactor *= adjust_cal_factor

    return 1.0/calfactor  # 10^9 ph/cm2/s/sr / DN/bin/gain/s = 1 kR / DN/bin/gainc


def getspecbins(myfits):
    spapix = np.append(myfits['Binning'].data['SPAPIXLO'][0],
                       myfits['Binning'].data['SPAPIXHI'][0][-1])
    spepix = np.append(myfits['Binning'].data['SPEPIXLO'][0],
                       myfits['Binning'].data['SPEPIXHI'][0][-1])
    return (spepix, spapix)


def gainscaledcounts(fits):
    gain = np.nanmedian(fits[0].data/fits['detector_dark_subtracted'].data)
    return np.sum(gain*fits['detector_dark_subtracted'].data, axis=2)


def lya_flux_to_g_factor(euvm_flux):
    # converts EUVM fluxes in ph/cm2/s/nm to a g factor
    lambda_lya = 121.56e-7  # cm
    lya_cross_section_total = 2.647e-2 * 0.416  # cm^2 Hz
    clight = 3e10  # cm/s
    g_factor = euvm_flux * 1e7 * lambda_lya * lambda_lya / clight
    #                      ^^^ nm/cm
    # now we're in photons/s/cm2/Hz
    g_factor *= lya_cross_section_total
    return g_factor
