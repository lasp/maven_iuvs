import warnings
import idl_colorbars
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def detector_image(myfits, integration=0,
                   fig=None, ax=None,
                   norm=None, cmap=109,
                   scale="linear",
                   arange=None, prange=None):
    """Makes a density plot of detector counts for the input FITS file and
    specified integration.

    Parameters
    ----------
    myfits : IUVSFITS or HDUList
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
    arange : None or 2-tuple
        Absolute range of counts scale. If None, entire range is
        plotted. Overrides prange.
    prange : None or 2-tuplen
        Percentile range of counts scale. If None, entire range is plotted.

    Returns
    -------
    fig : matplotlib.pyplot figure instance
        If ax==None, the new figure created by this routine.
    """

    new_ax = False
    if ax is None:
        new_ax = True
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1, 1, 1)
    if not isinstance(cmap, int):
        raise ValueError("cmap must be an integer"
                         " specifying an idl_colorbars colormap.")

    cmap = idl_colorbars.getcmap(cmap)

    ax.set_xlim([0, 1024])
    ax.set_ylim([0, 1024])

    # get the data
    data = myfits['detector_dark_subtracted'].data[integration]

    # figure out the binning
    spapixlo = myfits['Binning'].data['SPAPIXLO'][0]
    spapixhi = myfits['Binning'].data['SPAPIXHI'][0]
    spepixlo = myfits['Binning'].data['SPEPIXLO'][0]
    spepixhi = myfits['Binning'].data['SPEPIXHI'][0]
    if not (set((spapixhi[:-1]+1)-spapixlo[1:]) == {0}
            and set((spepixhi[:-1]+1)-spepixlo[1:]) == {0}):
        raise ValueError("Error in bin table")

    spepixrange = np.concatenate([[spepixlo[0]], spepixhi+1])
    spapixrange = np.concatenate([[spapixlo[0]], spapixhi+1])

    spepixwidth = spepixrange[1:]-spepixrange[:-1]
    spapixwidth = spapixrange[1:]-spapixrange[:-1]

    npixperbin = np.outer(spapixwidth, spepixwidth)

    data = data/npixperbin

    # figure out what norm to use
    if norm is None:
        if prange is None:
            prange = [0, 100]
        if arange is None:
            arange = [np.percentile(data, prange[0]),
                      np.percentile(data, prange[1])]
        if scale == "linear":
            norm = mpl.colors.Normalize(vmin=arange[0],
                                        vmax=arange[1])
        elif scale == "sqrt":
            norm = mpl.colors.PowerNorm(gamma=0.5,
                                        vmin=arange[0],
                                        vmax=arange[1])
        elif scale == "log":
            norm = mpl.colors.LogNorm(vmin=arange[0],
                                      vmax=arange[1])
        else:
            raise ValueError

    ax.patch.set_color('#666666')
    ax.patch.set_alpha(1.0)
    pcm = ax.pcolormesh(spepixrange, spapixrange,
                        data,
                        norm=norm,
                        cmap=cmap)

    # add the colorbar axes
    ax_pos = ax.get_position()
    cax_width_frac = 0.07
    cax_margin = 0.02
    cax = fig.add_axes((ax_pos.x1+cax_margin*ax_pos.width,
                        ax_pos.y0,
                        cax_width_frac*ax_pos.width,
                        ax_pos.height))
    fig.colorbar(pcm, cax=cax)
    if scale == "linear":
        cax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    if new_ax:
        # fig.show()
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
        myfits : IUVSFITS or HDUList
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
                           + str(myfits['Observation'].data['MCP_VOLT'][0]))
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
        myfits : IUVSFITS or HDUList
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
