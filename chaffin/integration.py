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

    #get the filename of the FITS file without extensions or paths
    if type(myfits)!=fits.hdu.hdulist.HDUList:
        fits_file_name=os.path.basename(myfits)
    else:
        fits_file_name=myfits['Primary'].header['FILENAME']
    fits_file_name=fits_file_name.replace('.fits','').replace('.gz','')

    #determine where to save the lya fit values
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
        lyavals=fit_line(myfits,121.56)
        np.save(save_file_name,lyavals)
        return lyavals
    

def fit_line(myfits, l0, calibrate=True, flatfield_correct=True, plot=False, correct_muv=False):
    import astropy.io.fits as fits
    if type(myfits)!=fits.hdu.hdulist.HDUList:
        myfits=fits.open(myfits)

    if correct_muv:
        warnings.warn('correct_muv not implemented, this flag does not change output values')
        #get the muv counterpart of this observation, if it exists
        try:
            fuv_filename=myfits.filename()
            fuv_dir=os.path.dirname(fuv_filename)
            muv_filename=os.path.basename(fuv_filename).replace('fuv','muv')
            muv_filename=os.path.join(fuv_dir,muv_filename)
            myfits_muv=fits.open(muv_filename)
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
            waves = myfits['Observation'].data['WAVELENGTH'][0, ispa]
            DN = myfits['detector_dark_subtracted'].data[iint, ispa]

            # subset the data to be fitted to the vicinity of the spectral line
            d_lambda = 2.5
            fitwaves, fitDN = np.transpose([[w, d] for w, d in zip(waves, DN) if w > l0-d_lambda and w < l0+d_lambda])
            lineDNmax = np.max([lineDNmax, np.max(fitDN)])

            # guess what the fit parameters should be
            DNguess = np.sum(fitDN)
            backguess = np.median(fitDN[0:3])+np.median(fitDN[-3:-1])/2
            slopeguess = np.median(
                fitDN[-3:-1]-np.median(fitDN[0:3]))/(fitwaves[-1]-fitwaves[0])

            # define the line spread function for this spatial element
            def this_spatial_element_lsf(x, scale=5e6, dl=1, x0=0, s=0, b=0, background_only=False):
                unitlsf = lsf[ispa](dl*x-x0)
                unitlsf /= np.sum(unitlsf)

                if background_only:
                    return s*(x-x0) + b
                else:
                    return scale*unitlsf + s*(x-x0) + b

            # do the fit
            try:
                from scipy.optimize import curve_fit
                fit = curve_fit(this_spatial_element_lsf, fitwaves, fitDN, p0=( DNguess, 1.0, l0, slopeguess, backguess))
                thislinevalue = fit[0][0]
            except RuntimeError:
                fit = [(DNguess, 1.0, l0, slopeguess, backguess)]
                thislinevalue = np.nan

            DN_fit = thislinevalue
                
            # return the requested values
            if flatfield_correct:
                thislinevalue /= flatfield[ispa]

            if calibrate:
                cal_factor = get_line_calibration(myfits, l0)
                thislinevalue *= cal_factor

            linevalues[iint, ispa] = thislinevalue
                
            if plot:
                  myplot.plot_line_fits(iint, ispa,
                                        fitwaves,
                                        fitDN, this_spatial_element_lsf(fitwaves, *fit[0],background_only=True), this_spatial_element_lsf(fitwaves, *fit[0]),
                                        DNguess, DN_fit, thislinevalue)


    if plot:
        myplot.finish_plot(lineDNmax, linevalues)
        return linevalues, myplot.fig
    else:
        return linevalues

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
