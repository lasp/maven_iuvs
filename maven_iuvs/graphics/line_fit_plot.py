import idl_colorbars
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from maven_iuvs.binning import get_npix_per_bin, get_pix_range
from maven_iuvs.graphics import color_dict


def detector_image_echelle(myfits, data, spapixrange, spepixrange, num_frames=1, **kwargs):
    """Makes a density plot of detector DN/pix/sec for the input image data
    valid for l1a echelle files.

    Parameters
    ----------
    myfits : astropy HDUlist
             Input light fits file. Currently only used to get number of pixels per bin 
    data : array
           Coadded detector image data to be plotted (not yet divided by number of frames)
    spapixrange : array
                  Pixels in the spatial range at which data are recorded
    spepixrange : array
                  Pixels in the spectral range at which data are recorded
    num_frames : int
                 Number of total frames which were used to compile data, which 
                 should be co-added data (but not yet divided by number of frames)
    **kwargs : dictionary
               arguments which can be passed to plot_detector.

    Returns
    -------
    fig : matplotlib.pyplot figure instance
        If ax==None, the new figure created by this routine.
    """

    # Collect number of pixels per bin to correctly display data
    npixperbin = get_npix_per_bin(myfits)
    
    # Divide out the number of pix per bin
    # Total frames are already divided out elsewhere (in coadd_lights)   
    data = data / npixperbin

    plot_detector(data, spapixrange, spepixrange, **kwargs)


def detector_image(myfits, integration=0, **kwargs):
    """Makes a density plot of detector DN for the input FITS file and
    specified integration. Valid for l1b files, where dark subtraction
    has already been done.

    Parameters
    ----------
    myfits : astropy.io.fits instance
             IUVS FITS file to be plotted.
    integration : int
                  Integration number to be plotted.
    fig, ax : matplotlib.pyplot figure and axis instance
              figure and axis on which to draw.
    norm : matplotlib norm
           Norm to be used in density plot. Overrides scale.
    cmap : int
           Colormap number to use from idl_colorbars package.
    scale : "linear", "log", or "sqrt".
             Scale to use in drawing the plot. Overriden by norm.

    Returns
    -------
    fig : matplotlib.pyplot figure instance
        If ax==None, the new figure created by plot_detector.
    """

    # Collect the data for the given integration
    data = myfits['detector_dark_subtracted'].data[integration]

    spapixrange = get_pix_range(myfits, which="spatial")
    spepixrange = get_pix_range(myfits, which="spectral")
    npixperbin = get_npix_per_bin(myfits)
            
    data = data / npixperbin

    fig = plot_detector(data, spapixrange, spepixrange, **kwargs)


def plot_detector(data_to_plot, spapixrange, spepixrange, 
                  fig=None, ax=None, plot_full_extent=True,
                  scale="linear", print_scale_type=False, 
                  norm=None, cmap=None, cmap_fallback=109, show_colorbar=True, cbar_lbl_size=18, cbar_tick_size=16,
                  arange=None, prange=None):
    """
    Creates an image of the full detector with data_to_plot overlaid. 
    Factored out of the original detector_image.

    Parameters
    ----------
    data_to_plot : 2D Array
                   Valid data on the detector, prepared in whatever way 
                   is desired. 
    spapixrange : 1D Array
                  Pixel values in the spatial dimension of the data.
    spepixrange : 1D Array
                  Pixel values in the spectral dimension of the data.
    fig, ax : matplotlib.pyplot figure and axis instance
              figure and axis on which to draw.
    plot_full_extent : boolean
                       if True, creates a 1024x1024 square 
                       which contains the detector image.
                       If False, the plot will automatically 
                       set its own limits, which will then typically
                       be the ends of spapixrange and spepixrange.
    scale : "linear", "log", or "sqrt".
             Scale to use in drawing the plot. Overriden by norm.
    print_scale_type : boolean
                       Whether to print some text on the canvas stating the value of scale.
    norm : matplotlib norm
           Norm to be used in density plot. Overrides scale.
    cmap : int
           Colormap number to use from idl_colorbars package.
    show_colorbar : boolean
                    Whether to plot the colorbar. 
    cbar_lbl_size : int
                    Font size in points for the colorbar label
    cbar_tick_size : int
                    Font size in points for the colorbar tick labels    
    arange : None or 2-tuple
             Absolute range of DN scale. If None, entire range is
             plotted. Overrides prange.
    prange : None or 2-tuplen
             Percentile range of DN scale. If None, entire range is plotted.
    
    Returns
    ----------
    fig : matplotlib.pyplot figure instance
        If ax==None, the new figure created by this routine.
    """

    new_ax = False
    if ax is None:
        new_ax = True
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1, 1, 1)
    # if not isinstance(cmap, int):
    #     raise ValueError("cmap must be an integer"
    #                      " specifying an idl_colorbars colormap.")

    if cmap is None:
        cmap = idl_colorbars.getcmap(cmap_fallback)

    if plot_full_extent:
        ax.set_xlim([0, 1024])
        ax.set_ylim([0, 1024])
    
    # figure out what norm to use
    if norm is None:
        if prange is None:
            prange = [0, 100]
        if arange is None:
            arange = [np.percentile(data_to_plot, prange[0]),
                      np.percentile(data_to_plot, prange[1])]
        
        if scale == "linear":
            norm = mpl.colors.Normalize(vmin=arange[0], vmax=arange[1])
        elif scale == "sqrt":
            norm = mpl.colors.PowerNorm(gamma=0.5, vmin=arange[0], vmax=arange[1])
        elif scale == "log":
            norm = mpl.colors.LogNorm(vmin=arange[0], vmax=arange[1])
        else:
            raise ValueError
        
    ax.patch.set_color(color_dict['darkgrey'])
    ax.patch.set_alpha(1.0)
    pcm = ax.pcolormesh(spepixrange, spapixrange, data_to_plot, norm=norm, cmap=cmap)

    # add the colorbar axes
    if show_colorbar:
        ax_pos = ax.get_position()
        cax_width_frac = 0.07
        cax_margin = 0.02
        cax = fig.add_axes((ax_pos.x1+cax_margin*ax_pos.width,
                            ax_pos.y0,
                            cax_width_frac*ax_pos.width,
                            ax_pos.height))
        cb = fig.colorbar(pcm, cax=cax)
        cb.set_label(label="DN/sec/px", size=cbar_lbl_size) 
        cax.tick_params(which="both", labelsize=cbar_tick_size)
        if scale == "linear":
            cax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            
        if print_scale_type==True:
            ax.text(1.05, -0.05, f"{scale} scale", color=color_dict["darkgrey"], fontsize=16, transform=ax.transAxes)
        
    if new_ax:
        return fig


class LineFitPlot:
    """Class to draw plots of line fits in
    maven_iuvs.integration.fit_line()

    """

    n_int = 0
    n_spa = 0

    fig = None
    figure_size_x = 0
    figure_size_y = 0

    detector_image_axes = None
    counts_axes = None
    residual_axes = None
    thumbnail_axes = None
    correct_muv = False

    def __init__(self, myfits, n_int, n_spa, correct_muv):
        """Initialize the plot canvas, an n_int x n_spa array of plots. Called
        only by maven_iuvs.integration.fit_line().

        Parameters
        ----------
        myfits : astropy.io.fits instance
            Input FITS file to plot.
        n_int : int
            Number of integrations in myfits.
        n_spa : integer
            Number of spatial elements in myfits.
        correct_muv : bool
            Whether to reserve space for companion MUV file image
            plots.

        """

        self.n_int = n_int
        self.n_spa = n_spa
        self.correct_muv = correct_muv

        # set up plot axes
        bin_plot_size = 2    # plot size, square, in
        column_margin = 0.1  # in
        row_margin    = 0.6  # in

        detector_image_margin = 0.65  # in
        image_lineplot_margin = 0.5   # in, space between detector images
                                      # and line plots

        counts_plot_frac = 2.5  # ratio of height of counts plot to
                                # height of residual plot
        counts_residual_margin = 0.05  # in, fraction of plot height to
                                       # use as margin
        residual_plot_height = ((1-counts_residual_margin)
                                / (1+counts_plot_frac)
                                * bin_plot_size)
        counts_plot_height = counts_plot_frac*residual_plot_height

        # figure out how much space to save on top for annotations and
        # a quicklook of the integrated line values
        thumbnail_ratio = 0.05
        thumbnail_plot_height = n_int*bin_plot_size*thumbnail_ratio
        thumbnail_plot_width  = n_spa*bin_plot_size*thumbnail_ratio
        thumbnail_margin = [0.5, 0.1]  # bottom, top
        header_height = (thumbnail_margin[1]
                         + thumbnail_plot_height
                         + thumbnail_margin[0])
        header_height = np.max([2, header_height])

        # set overall image margins
        margins_x = [0.5, 0.1]  # left, right
        margins_y = [0.25, header_height]  # bottom, top

        self.figure_size_x = (margins_x[0]
                              + ((1+self.correct_muv)
                                 * (bin_plot_size+detector_image_margin))
                              + image_lineplot_margin
                              + n_spa*(bin_plot_size+column_margin)
                              - column_margin
                              + margins_x[1])
        self.figure_size_y = (margins_y[1]
                              + n_int*(bin_plot_size+row_margin)
                              - row_margin
                              + margins_y[0])

        # print(self.figure_size_x)
        # print(self.figure_size_y)

        dpi = np.min([100,
                      2**16/self.figure_size_x,
                      2**16/self.figure_size_y])
        # print(dpi)

        self.fig = plt.figure(figsize=(self.figure_size_x,
                                       self.figure_size_y),
                              dpi=dpi)

        # make axes for thumbnail
        self.thumbnail_axes = \
            self.fig.add_axes((margins_x[0]/self.figure_size_x,
                               (self.figure_size_y
                                - thumbnail_margin[1]
                                - thumbnail_plot_height)/self.figure_size_y,
                               thumbnail_plot_width/self.figure_size_x,
                               thumbnail_plot_height/self.figure_size_y))

        # print some basic info about the files
        file_text_start = 1+1/thumbnail_plot_width
        file_info_text = 'FUV integration report\n'
        file_info_text += myfits['Primary'].header['FILENAME']+'\n'
        file_info_text += ('MCP_VOLT: '
                           + str(myfits['Primary'].header['MCP_VOLT']))
        self.thumbnail_axes.text(file_text_start, 1, file_info_text,
                                 ha='left', va='top',
                                 transform=self.thumbnail_axes.transAxes,
                                 clip_on=False)

        # make axes for each integration and bin
        self.detector_image_axes = np.reshape([None] * n_int * (1+self.correct_muv),
                                              (n_int, 1+self.correct_muv))
        self.counts_axes         = np.reshape([None]*n_spa*n_int,
                                              (n_int, n_spa))
        self.residual_axes       = np.reshape([None]*n_spa*n_int,
                                              (n_int, n_spa))
        row_start_y = self.figure_size_y - margins_y[1] - bin_plot_size
        detector_image_row_start_y = (row_start_y
                                      + residual_plot_height
                                      + counts_residual_margin*bin_plot_size)

        for iint in range(n_int):
            plot_start_x = margins_x[0]
            self.detector_image_axes[iint][0] = \
                self.fig.add_axes((plot_start_x/self.figure_size_x,
                                   row_start_y/self.figure_size_y,
                                   bin_plot_size/self.figure_size_x,
                                   bin_plot_size/self.figure_size_y))
            plot_start_x += bin_plot_size + detector_image_margin
            if self.correct_muv:
                self.detector_image_axes[iint][1] = \
                    self.fig.add_axes((plot_start_x/self.figure_size_x,
                                       row_start_y/self.figure_size_y,
                                       bin_plot_size/self.figure_size_x,
                                       bin_plot_size/self.figure_size_y))
                plot_start_x += bin_plot_size + detector_image_margin
            plot_start_x += image_lineplot_margin
            for ispa in range(n_spa):
                self.counts_axes[iint][ispa] = \
                    self.fig.add_axes((plot_start_x/self.figure_size_x,
                                       (detector_image_row_start_y
                                        / self.figure_size_y),
                                       bin_plot_size/self.figure_size_x,
                                       counts_plot_height/self.figure_size_y))
                self.residual_axes[iint][ispa] = \
                    self.fig.add_axes((plot_start_x/self.figure_size_x,
                                       row_start_y/self.figure_size_y,
                                       bin_plot_size/self.figure_size_x,
                                       (residual_plot_height
                                        / self.figure_size_y)))
                plot_start_x += bin_plot_size + column_margin
            row_start_y -= (bin_plot_size+row_margin)
            detector_image_row_start_y -= (bin_plot_size+row_margin)


    def plot_detector(self, myfits, iint, myfits_muv=None):
        """Plot a detector image for the specified integration. Called only by
        maven_iuvs.integration.fit_line().

        Parameters
        ----------
        myfits : astropy.io.fits instance
            Input FITS file to plot.
        iint : int
            Integration number to plot
        myfits_muv : IUVS or HDUList
            Companion MUV FITS file to plot, if correct_muv = True.

        Returns
        -------
        None

        """
        # plot the detector image
        self.detector_image_axes[iint][0].text(0.0, 0.5, 'integration ' + str(iint),
                                               ha='right', va='center',
                                               rotation=90,
                                               transform=self.detector_image_axes[iint][0].transAxes,
                                               clip_on=False)
        self.detector_image_axes[iint][0].text(0.5, 1.0, 'FUV DN/pix',
                                               ha='center', va='bottom',
                                               transform=self.detector_image_axes[iint][0].transAxes,
                                               clip_on=False)
        self.detector_image_axes[iint][0].xaxis.set_ticks([])
        self.detector_image_axes[iint][0].yaxis.set_ticks([])
        detector_image(myfits, iint,
                       fig=self.fig, ax=self.detector_image_axes[iint][0],
                       scale='log', arange=[1, 1e5])
        if iint == 0:
            # add a note to orient the viewer
            self.detector_image_axes[iint][0].text(0.0, -0.025,
                                                   'spatial bins run from bottom to top\n'
                                                   '(small to large keyhole)',
                                                   size=6,
                                                   ha='left', va='top',
                                                   transform=self.detector_image_axes[iint][0].transAxes,
                                                   clip_on=False)
        if self.correct_muv:
            self.detector_image_axes[iint][1].text(0.5, 1.0,
                                                   'MUV DN/pix',
                                                   ha='center', va='bottom',
                                                   transform=self.detector_image_axes[iint][1].transAxes,
                                                   clip_on=False)
            self.detector_image_axes[iint][1].xaxis.set_ticks([])
            self.detector_image_axes[iint][1].yaxis.set_ticks([])
            detector_image(myfits_muv, iint,
                           fig=self.fig,
                           ax=self.detector_image_axes[iint][1],
                           scale='log', arange=[1, 1e5],
                           cmap=98)

    def plot_line_fits(self,
                       iint, ispa,
                       fitwaves,
                       fitDN, fitDN_unc,
                       background_fit, line_fit,
                       DNguess,
                       DN_fit, DN_unc,
                       thislinevalue, thislineunc):
        """Plot the data, fit, and residuals for the specified integration and
        spatial element. Called only by maven_iuvs.integration.fit_line().

        Parameters
        ----------
        iint : int
            Integration number to plot
        ispa : int
            Spatial bin number to plot
        fitwaves : list of float
            Detector wavelengths to plot (horizontal coordinates).
        fitDN, fitDN_unc : list of float
            Detector DN and uncertainties from the FITS file.
        background_fit, line_fit : list of float
            Fit values of background and total line fit.
        DNguess : float
            Initial guess for total line DN.
        DN_fit, DN_unc : float
            Fitted total line DN and estimated fit uncertainty.
        thislinevalue, thislineunc : float
            Fitted line value and uncertainty after any requested
            corrections.

        Returns
        -------
        None

        """

        data_color = '#1f78b4'
        fit_color = '#a6cee3'
        background_color = '#888888'

        # plot line shapes and fits
        self.counts_axes[iint][ispa].text(0.5, 1.0,
                                          'int '+str(iint)+' spa '+str(ispa),
                                          ha='center', va='bottom',
                                          transform=self.counts_axes[iint][ispa].transAxes,
                                          clip_on=False)

        self.counts_axes[iint][ispa].step(fitwaves, fitDN,
                                          color=data_color,
                                          where='mid')
        self.counts_axes[iint][ispa].errorbar(fitwaves, fitDN,
                                              lw=0,
                                              elinewidth=1.5,
                                              color=data_color,
                                              yerr=fitDN_unc)
        self.counts_axes[iint][ispa].step(fitwaves, background_fit,
                                          color=background_color,
                                          where='mid')
        self.counts_axes[iint][ispa].step(fitwaves, line_fit,
                                          color=fit_color,
                                          where='mid')

        self.counts_axes[iint][ispa].text(0.025, 0.975,
                                          'DN guess = '+str(int(DNguess)),
                                          size=6,
                                          ha='left', va='top',
                                          transform=self.counts_axes[iint][ispa].transAxes,
                                          clip_on=False)
        self.counts_axes[iint][ispa].text(0.025, 0.9  ,
                                          ('fit DN = '+str(int(np.round(DN_fit)))
                                           + ' ± '+str(int(np.round(DN_unc)))),
                                          size=6,
                                          ha='left', va='top',
                                          transform=self.counts_axes[iint][ispa].transAxes,
                                          clip_on=False)
        self.counts_axes[iint][ispa].text(0.025, 0.825,
                                          ('cal = '+'{:.2f}'.format(thislinevalue)
                                           + ' ± '
                                           + '{:.2f}'.format(thislineunc)+" kR"),
                                          size=6,
                                          ha='left', va='top',
                                          transform=self.counts_axes[iint][ispa].transAxes,
                                          clip_on=False)
        self.counts_axes[iint][ispa].xaxis.set_ticks([])
        self.counts_axes[iint][ispa].ticklabel_format(axis='y',
                                                      style='sci',
                                                      scilimits=(0, 0))

        # plot deviations
        self.residual_axes[iint][ispa].axhline(0, color=background_color)
        self.residual_axes[iint][ispa].step(fitwaves,
                                            (fitDN-line_fit)/np.sum(fitDN),
                                            color=data_color,
                                            where='mid')
        self.residual_axes[iint][ispa].errorbar(fitwaves,
                                                (fitDN-line_fit)/np.sum(fitDN),
                                                lw=0,
                                                elinewidth=1.5,
                                                color=data_color,
                                                yerr=fitDN_unc/np.sum(fitDN))
        self.residual_axes[iint][ispa].set_ylim(-0.06, 0.06)
        self.residual_axes[iint][ispa].yaxis.set_ticks([-0.05, 0, 0.05])
        self.residual_axes[iint][ispa].yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1,
                                                                                             decimals=0))

        if ispa == 0:
            self.counts_axes[iint][ispa].set_ylabel('Counts [DN/bin]')
            self.counts_axes[iint][ispa].text(0.025, 0.5, 'Data',
                                              color=data_color,
                                              ha='left', va='top',
                                              transform=self.counts_axes[iint][ispa].transAxes,
                                              clip_on=False)
            self.counts_axes[iint][ispa].text(0.025, 0.4, 'Fit',
                                              color=fit_color,
                                              ha='left', va='top',
                                              transform=self.counts_axes[iint][ispa].transAxes,
                                              clip_on=False)
            self.residual_axes[iint][ispa].text(0.025, 0.075, '(fit-data)/(fit DN)',
                                                size=6,
                                                ha='left', va='bottom',
                                                transform=self.residual_axes[iint][ispa].transAxes,
                                                clip_on=False)
        else:
            self.counts_axes[iint][ispa].yaxis.set_major_formatter(mpl.ticker.NullFormatter())
            self.residual_axes[iint][ispa].yaxis.set_major_formatter(mpl.ticker.NullFormatter())

        # if iint!=n_int-1:
        #     self.residual_axes[iint][ispa].xaxis.set_major_formatter(mpl.ticker.NullFormatter())

    def finish_plot(self, lineDNmax, linevalues):
        """Perform final plot scaling and plot a quicklook of fitted
        values. Called only by maven_iuvs.integration.fit_line().

        Parameters
        ----------
        lineDNmax : float
            Max value of line DN across all fits for this file. Used
            to set plot scaling.
        linevalues : n_int x n_spa array of float
            Fitted line values to put on quicklook plot at top of plot array.

        Returns
        -------
        None

        """
        # use the same scale for all the counts axes, based on the
        # largest value
        for iint in range(self.n_int):
            for ispa in range(self.n_spa):
                self.counts_axes[iint][ispa].set_ylim(0, 1.05*lineDNmax)

        # plot the values on the thumbnail axis
        norm = mpl.colors.Normalize(vmin=0, vmax=20)
        pcm = self.thumbnail_axes.pcolormesh(np.arange(self.n_spa+1)-0.5,
                                             np.arange(self.n_int+1)-0.5,
                                             linevalues,
                                             cmap=idl_colorbars.getcmap(109),
                                             norm=norm)
        self.thumbnail_axes.invert_yaxis()

        # add a colorbar
        ax_pos = self.thumbnail_axes.get_position()
        cax_width = 0.2/self.figure_size_x
        cax_margin = 0.05/self.figure_size_x
        cax = self.fig.add_axes((ax_pos.x1+cax_margin,
                                 ax_pos.y0,
                                 cax_width,
                                 ax_pos.height))
        self.fig.colorbar(pcm, cax=cax)
