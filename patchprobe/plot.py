"""
This module contains functions for plotting patch-probe datasets and
inference results

Part of the patchprobe package.

Copyright 2022, 2023, 2025 Vinothan N. Manoharan <vnm@seas.harvard.edu>

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program. If not, see <https://www.gnu.org/licenses/>.
"""

from .patchprobe import PatchprobeAccessor
from .util import dataset_from_summary
import numpy as np
from scipy.stats import pearsonr
import xarray as xr
import arviz as az
import arviz.labels as azl
import proplot as pplt
from matplotlib import pyplot as plt
import seaborn as sns

def plot_map(dataarray, ax, cmap, vmin=0, vmax=None, norm=None, patch_size=None,
             patch_minorlocator=1, probe_minorlocator=50, patch_majorlocator=5,
             probe_majorlocator=200, **pplt_kwargs):
    """
    Plot heat map of any quantity indexed by patch and probe -- for example,
    the mean of the raw data or the signal.  Note that this function currently
    does not handle decimated or truncated data sets very well.

    Parameters
    ----------
    dataarray : xarray DataArray
        Data to plot, indexed by patch and probe
    ax : ProPlot Axes object
        Axes to pass to plotting routine
    cmap : string
        Colormap to use
    vmin : float (default 0)
        minimum value for heatmap
    vmax : float (default None)
        maximum value for heatmap. If unspecified, set the maximum value
        according to 'quantile'
    norm : matplotlib.colors.Normalize (default None)
        Normalizer to use
    patch_size : int
        If specified, heatmap axes will be labeled by site number rather than
        patch and probe numbers
    patch_minorlocator, probe_minorlocator : int
        Minor tick increments for patches and probes
    patch_majrolocator, probe_majorlocator : int
        Major tick increments for patches and probes. These will be labeled

    Returns
    -------
    matplotlib.collections.PolyCollection :
        return value from ax.heatmap.  Can be used to set colormap
    """

    # attempt to copy array.  For some reason deep copy does not work here
    da = xr.DataArray(dataarray)

    patches = da.coords['patch']
    probes = da.coords['probe']
    num_probes = probes.shape[0]
    num_patches = patches.shape[0]

    if patch_size is not None:
        # convention is that the site of patch 1 is nucleotide 1. But in a
        # heatmap the tick marks are centered on each patch, so the grid
        # rectangle for a given patch,probe combination fills a region
        # extending below and above the nucleotide site. For clarity, the tick
        # marks should be placed at the top of the markers rather than the
        # centers, so that the markers span the nucleotides occupied by the
        # patch.  So we have to shift the coordinates by half a patch:
        da.coords['patch'] = (da.coords['patch']-0.5) * patch_size + 1

    patch_sites = da.coords['patch']

    # even if normalizer specifies vmax, still need to specify it explicitly in
    # the plotting call for some reason
    if norm is not None:
        heatmap_args = {'vmax': vmax, 'norm': norm}
    else:
        heatmap_args = {'vmin': vmin, 'vmax': vmax}
    # using heatmap instead of pcolormesh ensures that tick labels are centered
    # rather than placed on the edges of the grid cells. But we do have to
    # set aspect='auto' since default for heatmap is aspect of 1
    mesh = ax.heatmap(da, aspect='auto', cmap=cmap, **heatmap_args,
                      **pplt_kwargs)

    # we want patch 1 at the top. Instead of using yreverse, which can also
    # reverse panels or other plots that share the axis, we instead explicitly
    # reverse the limits. We add 0.5 of padding because the tick labels are
    # centered.
    ax.format(ylim=(patches[-1]+0.5, patches[0]-0.5))

    # set the tick labels
    ax.format(xlim=(probes[0]-0.5, probes[-1]+0.5),
              xminorlocator=probe_minorlocator,
              xlocator=pplt.arange(0, num_probes, probe_majorlocator))
    ax.format(yminorlocator=patch_minorlocator,
              ylocator=pplt.arange(0, num_patches, patch_majorlocator))
    if patch_size is not None:
        ax.format(ylim=(patches[-1]*patch_size+0.5, (patches[0]-1)*patch_size+0.5))
        ax.format(yminorlocator=probe_minorlocator,
                  ylocator=probe_majorlocator)#pplt.arange(0, patch_sites[-1], probe_majorlocator))
        ax.format(xlabel='probe site (nt)', ylabel='patch site (nt)')

    return mesh

def plot_background_panel(summary_stats, px, alpha=0.5, background_lim=None,
                          **pplt_kwargs):
    """
    Plots a Plot heat map of any quantity indexed by patch and probe -- for example,
    the mean of the raw data or the signal.  Note that this function currently
    does not handle decimated or truncated data sets very well.

    Parameters
    ----------
    summary_stats : xarray Dataset
        Summary statistics generated from PatchprobeModel.sample(), so that the
        object includes the original dataset
    px : ProPlot Axes object
        Axes to pass to plotting routine, generated by ax.panel()
    alpha : float
        Controls transparency of plot elements
    background_lim : tuple (float, float)
        Limits for background axis in panel
    pplt_kwargs : dictionary
        Dictionary of kwargs to pass to px.plot()

    Returns
    -------
    ProPlot Axes object
    """

    B = summary_stats.B.sel(metric='mean')

    # TODO: remove hardcoded default values and replace with smart selection of
    # limits
    if background_lim is None:
        if summary_stats.processing_method == 'matlab':
            background_lim = [5e5, 5e8]
        else:
            background_lim = [9, 9e3]
        if summary_stats.probe_size == 24:
            background_lim[1] = background_lim[1]*10

    px.plot(B, label='inferred', alpha=alpha)
    px.semilogy(summary_stats.background.mean(('spot', 'repeat')), '.', s=1,
                label='estimated', alpha=alpha)
    px.legend(loc='best', frame=False, fontcolor='gray')
    px.format(ylim=background_lim, ylabel = 'Background', yformatter='sci')

    return px


def plot_raw(dataset, ax=None, cmap='dusk', colorbar=True,
             colorbar_width='1em', vmin=0, vmax=None, quantile=0.999,
             show_background=True, **pplt_kwargs):
    """
    Plot heat map of raw patch-probe spectra, using mean of replicates at
    each patch, probe value.

    Parameters
    ----------
    dataset : xarray Dataset
        Data to plot
    ax : ProPlot Axes object
        Axes to pass to plotting routine
    cmap : string
        Colormap to use
    colorbar : Boolean (default True)
        If true, plot colorbar next to heatmap
    colorbar_width : string
        width of colorbar, in em
    vmin : float (default 0)
        minimum value for heatmap
    vmax : float (default None)
        maximum value for heatmap. If unspecified, set the maximum value
        according to 'quantile'
    quantile : float (default 0.999)
        Sets the quantile level corresponding to the maximum value for the
        heatmap. For example, if quantile=0.9, vmax for the heatmap is set to
        the value of the 90% percentile of the mean patch-probe fluorescence
        measurements. Ignored if vmax is specified.
    show_background : boolean (default True)
        If True, plot a heatmap of all the background measurements above the
        raw data.
    pplt_kwargs : dictionary
        Dictionary of kwargs to pass to ProPlot plotting routine
    """
    colorbar_kwargs = {'location': 'left',
                       'length': 0.7,
                       'title': 'mean\nfluorescence',
                       'labelloc': 'left',
                       'width': colorbar_width}

    if ax is None:
        fig, ax = pplt.subplots()

    patchprobe = dataset.patchprobe.copy()
    patches = dataset.coords['patch']
    repeats = dataset.coords['repeat']

    # Calculate means by replicate, using transpose so that patch is along
    # y axis when we plot.
    raw_means = patchprobe.mean('spot')

    # Set maximum level of heatmap according to specified percentile
    # of patch-probe fluorescence distribution
    if vmax is None:
        vmax = raw_means.quantile(quantile)

    mesh = plot_map(raw_means, ax, cmap, vmin=vmin, vmax=vmax,
                    **pplt_kwargs)
    ax.format(title=dataset.description)

    # Panel for plotting background
    if show_background:
        px = ax.panel('t', space='1em', width='5em')
        px.format(ylabel = 'background\nmeasurement', yrotation=90)
        px.pcolormesh(dataset.background.mean(('spot')), cmap=cmap,
                   vmin=vmin, vmax=vmax)
        # turn off minor ticks and show only major ticks, centered on rows
        px.format(yminorlocator='null', ylim=(repeats[0]-0.5, repeats[-1]+0.5),
                  ylocator=pplt.arange(repeats[0], repeats[-1], 1))

    # Now draw the colorbar to the right of the scalings panel
    if colorbar:
        ax.colorbar(mesh, **colorbar_kwargs)

    return ax, mesh

def plot_posterior_scatter(summary, ax=None, cmap='fire',
                           levels=[5, 10, 15, 20, 25],
                           signal_scale = 20, marker_scale=30,
                           colorbar_width='1em', show_legend=True,
                           **scatter_kw):
    """
    Plots the posterior of the signal as a type of heatmap, showing both the
    magnitude of the signal (through the size of the marker) and its
    credibility (mean/standard deviation, shown as a color)

    Parameters
    ----------

    summary_stats : xarray Dataset
        Summary statistics generated from PatchprobeModel.sample(), so that the
        object includes the original dataset
    ax : ProPlot Axes object
        Axes to pass to plotting routine
    cmap : string
        Colormap to use
    levels: list of float
        levels to use for the colormap (signal mean to standard deviation)
    signal_scale : float
        typical scale of a large signal value. Used to set marker sizes. Keep
        this constant to ensure that plots of different datasets can be
        compared.
    marker_scale : float
        size (in points**2) of marker corresponding to typical maximum signal
        value. As with signal_scale, keep constant to ensure that plots can be
        compared
    colorbar_width : string
        width of colorbar, in em.  If set to None, colorbar is removed
    show_legend : boolean
        If true, show a legend relating marker sizes to mean signal values
    scatter_kw : dictionary
        keyword arguments to pass to ax.scatter()
    """

    colorbar_kwargs = {'location': 'left',
                       'length': 0.7,
                       'title': 'credibility',
                       'labelloc': 'left',
                       'width': colorbar_width}

    signal = summary.S

    num_probes = summary.ppdata.num_probes
    num_patches = summary.ppdata.num_patches

    # point estimate of signal
    S_est = signal.sel(metric='mean')
    # remove negative values, since these will skew the marker sizes
    S_est = S_est.where(S_est>0)

    # uncertainty estimate of signal
    S_uncertainty = signal.sel(metric='sd')
    # ratio of estimate to uncertainty
    S_ratio = (S_est/S_uncertainty)
    # remove negative values, since these will skew the colorbar
    S_ratio = S_ratio.where(S_ratio>0)
    S_ratio_max = S_ratio.max()

    # Set defaults for scatter plot arguments that can be overriden by
    # user-specified values
    default_scatter_kw = {'discrete': True,
                          'levels': levels,
                          'extend': 'both',
                          'cmap': cmap}
    # in Python 3.5+ the following merges the dictionaries, keeping only the
    # last key:value pair if the key is duplicated. This way the user-specified
    # values override the defaults
    # see https://stackoverflow.com/a/51940623
    scatter_kwargs = {**default_scatter_kw, **scatter_kw}

    if ax is None:
        fig, ax = pplt.subplots()

    ax.format(grid=False)
    # set data aspect ratio to be square; otherwise
    # adding colorbar will change the aspect ratio
    ax.format(aspect=num_probes/num_patches)

    # construct arrays for plotting each data point
    x = np.repeat(S_est.coords['probe'].values, num_patches)
    y = np.tile(S_est.coords['patch'].values, num_probes)

    # map signal size to marker size
    #
    # matplotlib scales markers so that area is proportional to data value
    # We modify this scaling with a prefactor, so that a typical max
    # signal size (20) corresponds to the maximum marker size
    marker_sizes = S_est.values.flatten() * marker_scale/signal_scale

    m = ax.scatter(x, y, m='o', c=S_ratio.values.flatten(), s=marker_sizes,
                   edgewidth=0, alpha=1, absolute_size=True, **scatter_kwargs)

#    ax.format(xlim=[-10, num_probes+10], ylim=[0.2, num_patches+0.8],
#              ylabel='patch', xlabel='probe', yreverse=True)
    ax.format(xlim=[-10, num_probes+10], ylim=[num_patches+0.8, 0.2],
              ylabel='patch', xlabel='probe')

    if colorbar_width is not None:
        ax.colorbar(m, **colorbar_kwargs)
    # see here: https://stackoverflow.com/a/33740567
    # for how to put the label above the colorbar
    # instead of next to it.

    # Additional legend to show sizes of circles
    #
    # pieced together from answers here: https://stackoverflow.com/q/24164797
    # note that using legend_elements() of m
    # (as suggested in https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_with_legend.html
    # doesn't seem to work
    signal_sizes = np.arange(0, S_est.max(), 5)
    markers = []
    for size in signal_sizes:
        markers.append(plt.scatter([],[], s=size*marker_scale/signal_scale,
                                   label=f"{size:.0f}", c='gray'))

    if show_legend:
        leg = ax.legend(handles=markers, loc='right', ncols=1, frame=False,
                        fontcolor=pplt.rc['meta.edgecolor'], titlefontcolor='gray',
                        title='posterior\nmean')
        # center each line of the legend label
        # from https://stackoverflow.com/a/64591633
        leg.get_title().set_multialignment('center')

    return ax

def plot_signal_posterior(summary_stats, quantile=0.999, alpha=0.5,
                          figwidth=None, panel_space=0, panel_width='8em',
                          colorbar_width='1em', show_background=True,
                          show_scalings=True, background_lim=None,
                          scaling_lim=(0.6, 1.5), **posterior_scatter_kw):
    """
    Plot posterior for several quantities as inferred using the signal model.

    Parameters
    ----------
    summary_stats : xarray Dataset
        Summary statistics generated from PatchprobeModel.sample(), so that the
        object includes the original dataset
    quantile : float (default 0.999)
        Sets the quantile level corresponding to the maximum value for the
        heatmap. For example, if quantile=0.9, vmax for the heatmap is set to
        the value of the 90% percentile of the measurements.
    alpha : float
        Controls transparency of plot elements
    figwidth : float
        Width of figure in inches
    panel_space : float
        Space between heatmap and panels, which show background and scalings
    panel_width : string
        Size of panels showing background and scalings
    colorbar_width : string
        width of colorbar, in em
    show_background : boolean
        controls whether to show the background panel next to the raw data plot
    show_scalings : boolean
        controls whether to show the scalings panel next to the raw data plot
    background_lim : tuple (float, float)
        Limits for background axis in panel
    scaling_lim : tuple (float, float)
        Limits for scaling axis in panel
    posterior_colorbar_kw : dict
        keyword args for the colorbar in the plot of the posterior mean and credibility
    """

    S = summary_stats.S.sel(metric='mean')
    B = summary_stats.B.sel(metric='mean')
    c = summary_stats.c.sel(metric='mean')

    num_probes = summary_stats.ppdata.num_probes
    num_patches = summary_stats.ppdata.num_patches

    dataset = dataset_from_summary(summary_stats)

    patches = dataset.coords['patch']
    probes = dataset.coords['probe']

    # this arrangement will give the posterior map a prominent placement on the
    # right side
    array = [[1, 3, 3],
             [2, 3, 3]]
    fig, axs = pplt.subplots(array=array, share=False, span=False, figwidth=figwidth)

    # If some signals are negative, we'll use fire to indicate magnitude of the
    # negative signals and dusk for the positive
    if S.min() >= 0:
        cmap = 'dusk'
    else:
        cmap = pplt.Colormap('fire_r', 'dusk', name='Diverging')

    # Plot of raw data (mean over spots)
    plot_raw(dataset, ax=axs[0], quantile=quantile, show_background=False)
    axs[0].format(title='Patch-probe fluorescence')

    # Panel for plotting background
    if show_background:
        px = axs[0].panel('b', space=panel_space, width=panel_width)
        plot_background_panel(summary_stats, px, alpha=alpha,
                              background_lim=background_lim)

    # Panel for plotting scalings
    if show_scalings:
        px = axs[0].panel('r', space=panel_space, width=panel_width)
        px.plot(c, c.coords['patch'], label='inferred', alpha=alpha)
        expected_c = 1/dataset.ppdata.normalization_weights()
        px.plot(expected_c, c.coords['patch'], '.', s=1, label='estimated')
        px.format(xlabel='Scaling', xlim=scaling_lim)

    axs.format(abc='A')

    # Plot inferred signal below the raw data so that we can compare
    vmax = S.quantile(quantile)
    signal_norm = pplt.DivergingNorm(vcenter=0, vmin=-1, vmax=vmax, clip=True, fair=True)
    signal_mesh = plot_map(S.transpose(), axs[1], cmap, vmax=vmax, norm=signal_norm)

    axs[1].colorbar(signal_mesh, loc='l', title='signal', length=0.7,
                    labelloc='left', width=colorbar_width)
    axs[1].format(title='Inferred signal')

    # And plot the posterior on the right
    scatter_kw={'cmap' : 'fire'}
    plot_posterior_scatter(summary_stats, ax=axs[2],
                           colorbar_width=colorbar_width, **scatter_kw)
    axs[2].format(title='Posterior mean and credibility of signal')

    return fig, axs

def plot_prevalence(summary_stats, ax=None, cmap='plasma', levels=5, vmin=0.75,
                    vmax=1, colorbar=True, colorbar_width='1em', title=None,
                    **pplt_kwargs):
    """
    Plot heat map of inferred prevalences

    Parameters
    ----------
    summary_stats : xarray Dataset
        Summary statistics generated from PatchprobeModel.sample(), so that the
        object includes the original dataset
    ax : ProPlot Axes object
        Axes to pass to plotting routine
    cmap : string
        Colormap to use
    levels: list of float
        levels to use for the colormap (signal mean to standard deviation)
    vmin : float
        minimum value for heatmap
    vmax : float
        maximum value for heatmap.
    colorbar : Boolean (default True)
        If true, plot colorbar next to heatmap
    colorbar_width : string
        width of colorbar, in em
    title : string
        Title of plot. If set to 'description', uses description from dataset
    pplt_kwargs : dictionary
        Dictionary of kwargs to pass to ProPlot plotting routine
    """
    colorbar_kwargs = {'location': 'right',
                       'length': 0.7,
                       'title': 'prevalence',
                       'labelloc': 'right',
                       'width': colorbar_width}

    # Set defaults for keyword arguments that can be overriden by
    # user-specified values
    default_pplt_kw = {'discrete': True,
                       'levels': levels,
                       'extend': 'neither',
                       'cmap': cmap}
    # in Python 3.5+ the following merges the dictionaries, keeping only the
    # last key:value pair if the key is duplicated. This way the user-specified
    # values override the defaults
    # see https://stackoverflow.com/a/51940623
    pplt_kw = {**default_pplt_kw, **pplt_kwargs}

    prevalence = summary_stats.f.sel(metric='mean').transpose()

    num_probes = summary_stats.ppdata.num_probes
    num_patches = summary_stats.ppdata.num_patches

    if ax is None:
        fig, ax = pplt.subplots()

    mesh = plot_map(prevalence, ax, vmin=vmin, vmax=vmax,
                    patch_size=summary_stats.patch_size, **pplt_kw)
    if title=='description':
        ax.format(title=dataset.description)
    else:
        ax.format(title=title)

    # Now draw the colorbar to the right of the scalings panel
    if colorbar:
        ax.colorbar(mesh, **colorbar_kwargs)

    return ax, mesh

def plot_prevalence_posterior(summary_stats, quantile=0.999,
                              alpha=0.5, figwidth=None, panel_space=0,
                              panel_width='8em', colorbar_width='1em',
                              show_background=True, show_affinities=True,
                              background_lim=None, affinity_lim=(0, 1),
                              reference_affinities=None, reference_conc=None,
                              **kwargs):
    """
    Plot posterior for several quantities as inferred using the fraction model.

    Parameters
    ----------
    summary_stats : xarray Dataset
        Summary statistics generated from PatchprobeModel.sample(), so that the
        object includes the original dataset
    quantile : float (default 0.999)
        Sets the quantile level corresponding to the maximum value for the
        raw data heatmap. See plot_raw()
    alpha : float
        Controls transparency of plot elements
    figwidth : float
        Width of figure in inches
    panel_space : float
        Space between heatmap and panels, which show background and scalings
    panel_width : string
        Size of panels showing background and scalings
    colorbar_width : string
        width of colorbar, in em
    show_background : boolean
        controls whether to show the background panel next to the raw data plot
    show_affinities : boolean
        controls whether to show the patch affinities panel next to the raw
        data plot
    background_lim : tuple (float, float)
        Limits for background axis in panel
    affinity_lim : tuple (float, float)
        Limits for affinity axis in panel
    reference_affinities : xarray DataSet
        Reference set of affinities to compare inferred affinities to. If
        provided, add a subplot showing correlations between these
        quantities.  These should be in a DataSet with "patch" as coordinate
        and "pjmean" (posterior mean of p) and "pjstd" (posterior standard
        deviation) as variables
    reference_conc : float
        Concentration (in nM) to which bulk affinities were corrected
    kwargs : dict
        keyword args to pass to the prevalence heatmap
    """

    B = summary_stats.B.sel(metric='mean')
    prevalence = summary_stats.f.sel(metric='mean')
    p = summary_stats.p.sel(metric='mean')
    p_uncertainty = summary_stats.p.sel(metric='sd')
    S = summary_stats.S.sel(metric='mean')

    num_probes = summary_stats.ppdata.num_probes
    num_patches = summary_stats.ppdata.num_patches

    dataset = dataset_from_summary(summary_stats)

    patches = dataset.coords['patch']
    probes = dataset.coords['probe']

    if reference_affinities is None:
        # this arrangement will give the posterior map a prominent placement on the
        # right side
        array = [1, 2]
    else:
        array = [[1, 3, 3],
                 [2, 3, 3]]

    fig, axs = pplt.subplots(array=array, share=False, span=False, figwidth=figwidth)

    # Plot of raw data (mean over spots)
    plot_raw(dataset, ax=axs[0], quantile=quantile, show_background=False)
    axs[0].format(title='Patch-probe fluorescence')

    # Panel for plotting background
    if show_background:
        px = axs[0].panel('b', space=panel_space, width=panel_width)
        plot_background_panel(summary_stats, px, alpha=alpha,
                              background_lim=background_lim)

    # Panel for plotting scalings
    if show_affinities:
        px = axs[0].panel('r', space=panel_space, width=panel_width)
        px.barh(p, alpha=alpha)
        px.format(xlabel='patch\naffinity', xlim=affinity_lim, xminorlocator=0.1,
                  xlocator=pplt.arange(affinity_lim[0]+0.25, affinity_lim[-1], 0.25))

    axs.format(abc='A')

    if reference_affinities is not None:
        posterior_ax = axs[2]
        # Plot comparison of inferred and reference patch affinities

        bulk_patches = reference_affinities.coords['patch']
        p_subset = p.sel(patch=bulk_patches)
        p_ref = reference_affinities.pjmean
        xmax = p_ref.max()*1.10
        if xmax > 1:
            xmax = 1
        ymax = p_subset.max()*1.10
        if ymax > 1:
            ymax = 1

        correlation_stats = pearsonr(x=p.sel(patch=bulk_patches),
                                     y=reference_affinities.pjmean)
        # axs[1].scatterx(x=reference_affinities.pjmean,
        #                 y=p.sel(patch=bulk_patches),
        #                 bardata=reference_affinities.pjstd, s=0, capsize=0)
        #axs[1].format(xlim=[0, 1], ylim=[0, 1])
        sns.regplot(ax=axs[1], x=reference_affinities.pjmean,
                    y=p.sel(patch=bulk_patches), truncate=False)
        axs[1].format(xlim=[0, xmax], ylim=[0, ymax])
        if reference_conc is not None:
            label = f"bulk measurement corrected to {reference_conc} nM"
        else:
            label = "corrected bulk measurement"
        axs[1].format(xlabel=label, ylabel='inferred patch affinity',
                      title='Patch affinity correlation')
        # [] sets artist list to empty, removing the redundant entry for "p"
        axs[1].legend([], title=f'$r={correlation_stats[0]:.2}$', location='lr',
                      labels=None, frame=False)


    else:
        posterior_ax = axs[1]

    # Plot the posterior on the right
    plot_prevalence(summary_stats, ax=posterior_ax,
                    colorbar_width=colorbar_width)
    posterior_ax.format(title='Posterior mean of prevalence', aspect=1)

    return fig, axs

def plot_hyperparameter_posterior(idata,
                                  var_name_map={"sigma": r"$\sigma$",
                                                "nu_obs": r"$\nu_S$",
                                                "nu_obs_B": r"$\nu_B$",
                                                "mu_c": r"$\mu_c$",
                                                "sigma_c": r"$\sigma_c$"}):
    """
    Plots the marginal distributions of the hyperparameters, as inferred by
    MCMC. Note that this function requires the InferenceData object, since the
    summary statistics generated by arviz don't contain the full posterior.

    Parameters
    ----------
    idata : augmented arviz InferenceData object
        Samples generated by PatchprobeModel.sample(). This is an arviz
        InferenceData object augmented to include the original data
    var_name_map : dict
        Names of hyperparameters (keys) to plot, along with labels (values) to
        be used on plots
    """

    labeller = azl.MapLabeller(var_name_map)
    var_names = var_name_map.keys()
    fig, axs = pplt.subplots(ncols=len(var_names), sharex=False, sharey=False,
                             spanx=False, spany=False)
    for i, var in enumerate(var_names):
        az.plot_posterior(idata, var_names=[var], ax=axs[i], labeller=labeller,
                          textsize=pplt.rc['font.size'])

    return fig, axs
