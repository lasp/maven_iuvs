import os
import warnings
import numpy as np
from scipy.interpolate import interp1d

from maven_iuvs import anc_dir
from maven_iuvs.instrument import calculate_calibration_curve
from maven_iuvs.graphics import LineFitPlot


def fit_line(myfits, wavelength,
             calibrate=True, flatfield_correct=True,
             correct_muv=False,
             plot=False):
    """
    Fit a spectral line in the input IUVS FITS file and return line
    brightness for each integration and spatial element of the
    file. Developed and tested for Lyman alpha.

    Parameters
    ----------
    myfits : IUVSFITS or HDUList
        FITS file interface to an IUVS FITS file.
    wavelength : float
        Wavelength of the line to be fit. (e.g. 121.56 for Lyman
        alpha).
    calibrate : bool
        If True, return brightness in kR, otherwise DN. Defaults to
        True.
    flatfield_correct : bool
        Whether to perform a flatfield correction on the brightness
        values. Defaults to True.
    correct_muv : bool
        Whether to use MUV contamination templates estimated from the
        companion MUV observation in fitting for line brightness. This
        is a stub that will be expanded later once code is ported from
        IDL. Defaults to False.
    plot : bool
        If True, a plot of each line fit will be made and
        returned. For large files, this greatly increases runtime and
        memory use.

    Returns
    -------
    linevalues : list of float
        Fitted brightness values for the line for each integration and
        spatial element of the input file.
    line unc : list of float
        Estimated uncertainty of the brightness values, resulting from
        naive error propagation in the fit function.
    fig : matplotlib.pyplot.figure
        If plot = True, the figure object of the plot of line fits.
    """

    # TODO: replace with IUVSFITS throughout
    import astropy.io.fits as fits
    if not isinstance(myfits, fits.hdu.hdulist.HDUList):
        myfits = fits.open(myfits)

    if correct_muv:
        warnings.warn('correct_muv not implemented,'
                      'this flag does not change output values')
        # get the muv counterpart of this observation, if it exists
        try:
            myfits_muv, muv_contamination_templates = \
                get_muv_contamination_templates(myfits)
        except FileNotFoundError:
            warnings.warn('no matching MUV observation found,'
                          'cannot correct MUV')
            correct_muv = False

    if flatfield_correct:
        if (np.abs(wavelength-121.56) > 1):
            warnings.warn('using flat field derived at Lyman alpha')

        flatfield = get_lya_flatfield(myfits)

    filedims = myfits['Primary'].shape
    n_int = filedims[0]
    n_spa = filedims[1]

    lsf = get_lsf_interp(myfits)

    linevalues = np.zeros((n_int, n_spa))
    lineunc    = np.zeros((n_int, n_spa))
    lineDNmax  = 0  # to set max scale of plot

    if plot:
        myplot = LineFitPlot(myfits, n_int, n_spa, correct_muv)

    # Main fitting loop
    for iint in range(n_int):
        if plot:
            if correct_muv:
                myplot.plot_detector(myfits, iint, myfits_muv=myfits_muv)
            else:
                myplot.plot_detector(myfits, iint)

        for ispa in range(n_spa):
            waves = myfits['Observation'].data['WAVELENGTH'][0, ispa]
            DN = myfits['detector_dark_subtracted'].data[iint, ispa]
            DN_unc = myfits['Random_dn_unc'].data[iint, ispa]
            if correct_muv:
                muv = muv_contamination_templates[ispa]
            else:
                muv = np.zeros_like(DN)

            # subset the data to be fitted to the vicinity of the spectral line
            d_lambda = 2.5
            fitwaves, fitDN, fitDN_unc, fitmuv = \
                np.transpose([[w, d, du, m]
                              for w, d, du, m in zip(waves, DN, DN_unc, muv)
                              if (w > wavelength-d_lambda
                                  and w < wavelength+d_lambda)])
            lineDNmax = np.max([lineDNmax, np.max(fitDN)])

            # guess what the fit parameters should be
            backguess = (np.median(fitDN[0:3])+np.median(fitDN[-3:-1]))/2
            slopeguess = ((np.median(fitDN[-3:-1]) - np.median(fitDN[0:3]))
                          /
                          (fitwaves[-1]-fitwaves[0]))
            DNguess = np.sum(fitDN) - backguess * len(fitwaves)

            # define the line spread function for this spatial element
            def this_spatial_element_lsf(x,
                                         scale=5e6, dl=1, x0=0, s=0, b=0,
                                         muv_background_scale=0,
                                         background_only=False):
                """Helper function to fit individual spatial element LSF to
                data. Accepts input wavelengths and LSF fit parameters
                and returns the LSF evaluated at those parameters. MUV
                parameters are stubs to be expanded later.

                """
                unitlsf = lsf[ispa](dl*(x-x0))
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

                # Reasonable bounds for the fit parameters
                parms_bounds = ([     0,  0.0, wavelength-d_lambda, -np.inf, -np.inf],
                                [np.inf, 10.0, wavelength+d_lambda,  np.inf,  np.inf])

                parms_guess = [DNguess, 1.0, wavelength, slopeguess, backguess]

                if correct_muv:
                    # we need to append a guess for the MUV background
                    parms_bounds[0].append(0)
                    parms_bounds[1].append(np.inf)
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

                thislinevalue = fit[0][0]  # keep only the total DN in the line
                thislineunc   = np.sqrt(fit[1][0, 0])  # estimate uncertainties
            except RuntimeError:
                # the fit has probably failed, return np.nan
                fit = [(DNguess, 1.0, wavelength, slopeguess, backguess)]
                thislinevalue = np.nan
                thislineunc = np.nan

            DN_fit = thislinevalue
            DN_unc = thislineunc

            # return the requested values
            if flatfield_correct:
                thislinevalue /= flatfield[ispa]
                thislineunc   /= flatfield[ispa]

            if calibrate:
                cal_factor = calculate_calibration_curve(myfits, wavelength)
                thislinevalue /= cal_factor
                thislineunc   /= cal_factor

            linevalues[iint, ispa] = thislinevalue
            lineunc[iint, ispa] = thislineunc

            if plot:
                myplot.plot_line_fits(iint, ispa,
                                      fitwaves,
                                      fitDN, fitDN_unc,
                                      this_spatial_element_lsf(fitwaves,
                                                               *fit[0],
                                                               background_only=True),
                                      this_spatial_element_lsf(fitwaves,
                                                               *fit[0]),
                                      DNguess,
                                      DN_fit, DN_unc,
                                      thislinevalue, thislineunc)

    if plot:
        myplot.finish_plot(lineDNmax, linevalues)
        return linevalues, lineunc, myplot.fig

    return linevalues, lineunc


def get_lsf(myfits):
    """
    Get IUVS line spread function array appropriate to the input IUVS
    FITS file.

    Parameters
    ----------
    myfits : IUVSFITS or HDUList
        FITS file interface to an IUVS FITS file.

    Returns
    -------
    lsf : N x M numpy array of floats
        Array with the LSF corresponding to each spatial bin. N =
        number of spatial bins, M = number of spectral elements in LSF.

    """
    spalo = myfits['Binning'].data['SPAPIXLO'][0]
    spalo = np.append(spalo,
                      myfits['Binning'].data['SPAPIXHI'][0][-1]+1)
    return get_lsf_from_bins(spalo)


def get_lsf_from_bins(spatial_binning):
    """
    Get IUVS line spread function appropriate to the input spatial
    binning.

    Parameters
    ----------
    spatial_binning : list of int
        List of the start pixels of the spatial bins. The last value
        in the list should be end of last spatial bin+1.

    Returns
    -------
    lsf : N x M numpy array of floats
        Array with the LSF corresponding to each spatial bin. N =
        number of spatial bins, M = number of spectral elements in LSF.
    """

    # load the empirical LSF, determined using cruise Lyman alpha data
    cruise_lsf = np.load(os.path.join(anc_dir, 'cruise_lsf_23Sep2020.npy'))
    spapix = np.arange(76, 916, 4)  # these are the start pixels of
                                    # the loaded LSF spatial bins. End
                                    # pixels are spapix[1:]-1

    # Make the binned LSF by adding up the cruise LSF based on the
    # supplied binning.
    lsf = np.zeros((len(spatial_binning)-1, cruise_lsf.shape[1]))
    for idx in range(len(spatial_binning)-1):
        this_lsf = np.sum(cruise_lsf[(spatial_binning[idx] < spapix+4)
                                     & (spapix < spatial_binning[idx+1])],
                          axis=0)
        this_lsf = this_lsf/np.sum(this_lsf)  # Normalize
        lsf[idx, :] = this_lsf

    return lsf

    
def get_lsf_interp(myfits):
    """
    Get the array of IUVS line spread interpolating functions
    appropriate to the input IUVS FITS file. Used for line fitting.

    Parameters
    ----------
    myfits : IUVSFITS or HDUList
        FITS file interface to an IUVS FITS file.

    Returns
    -------
    lsf_interp : list of scipy.interp1d functions
        Interpolating functions for the IUVS LSF for each spatial element.

    """
    lsf = get_lsf(myfits)

    lsf_interp = [None]*len(lsf)
    for i, l in enumerate(lsf):
        # shift the LSF x coordinates to roughly match the instrument
        # wavelength scale in nm (this is not exact because it is a
        # fit parameter later on)
        waves = 7.5 * np.linspace(-1, 1, len(l))

        interp = interp1d(x=waves,
                          y=l,
                          bounds_error=False,
                          fill_value=0.)

        lsf_interp[i] = interp

    return lsf_interp


def get_muv_contamination_templates(myfits_fuv):
    """
    Returns MUV contamination templates for an input FUV IUVSFITS
    file. This is a stub to be filled out by Sonal at some point in
    the future.

    Parameters
    ----------
    myfits_fuv : IUVSFITS or HDUList
        FITS file interface to an IUVS FUV FITS file.

    Returns
    -------
    myfits_muv : IUVSFITS
        MUV companion to the input FUV file
    muv_contamination_templates : numpy ndarray of floats
        n_spa x n_spe array of MUV contamination templates for each
        spatial element of the input FUV file

    """
    # find the filename of the matching muv file
    # TODO: replace with IUVSFITS when available
    fuv_filename = myfits_fuv.filename()
    fuv_dir = os.path.dirname(fuv_filename)
    muv_filename = os.path.basename(fuv_filename).replace('fuv', 'muv')
    muv_filename = os.path.join(fuv_dir, muv_filename)
    import astropy.io.fits as fits
    myfits_muv = fits.open(muv_filename)

    # get the MUV contamination templates
    # To be ported from IDL by Sonal
    n_int, n_spa, n_spe = myfits_muv['Primary'].data.shape
    muv_contamination_templates = np.zeros((n_spa, n_spe))

    return myfits_muv, muv_contamination_templates


def get_lya_flatfield(myfits):
    """Returns an FUV flatfield for the input FITS file spatial binning,
    based on an estimated flatfield at Lyman from stellar
    observations.

    Parameters
    ----------
    myfits : IUVSFITS or HDUList
        FITS file interface to an IUVS FITS file.

    Returns
    -------
    flatfield : list of floats
       Flatfield values for the input spatial binning. Dividing
       brightnesses by this array gives flatfield corrected data.
    """
    binning = np.concatenate([myfits['Binning'].data['SPAPIXLO'][0],
                              [myfits['Binning'].data['SPAPIXHI'][0][-1]+1]])
    flatfield_fname = os.path.join(anc_dir,
                                   'kei_flatfield_polynomial_25Nov2020.npy')
    slit_flatfield = np.load(flatfield_fname)
    flatfield = np.array([np.mean(slit_flatfield[p0:p1])
                          for p0, p1 in zip(binning[:-1], binning[1:])])
    if np.any(np.isnan(flatfield)):
        raise ValueError("Observation extents past airglow slit,"
                         " cannot flatfield correct.")

    return flatfield


