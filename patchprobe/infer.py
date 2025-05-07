"""
Routines for Bayesian inference and loading/saving results from inference

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

from pathlib import Path
import datetime
import numpy as np
import pymc as pm
import aesara
import aesara.tensor as at
import arviz as az
import pandas as pd
import xarray as xr
import copy
import warnings

class PatchprobeModel(pm.Model):
    """
    Class that combines a pymc Model with a dataset and inference metadata. A
    pymc Data container can't hold all the metadata and other information in
    the dataset, so we use this class instead to make it easier to pass the
    model and the data together.

    Documentation for subclassing Model:
    https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.Model.html
    Thread on *not* subclassing Model:
    https://discourse.pymc.io/t/how-can-i-create-a-class-using-pm-models-class-like-keras-model-or-torch-nn-module-option/10899/2

    We follow the subclassing approach here, which is convenient because it
    allows us to use the new object within a context manager just like a
    pm.Model() object

    Attributes
    ----------
    dataset : xarray Dataset
        Data that model will use
    name : string (default '')
        Name to be passed to pm.Model()
    background_scale : float
        expected magnitude of background values. Will depend on feature
        extraction software used (default 100 for Agilent feature extraction)
    model_type : string ('background', 'signal', 'fraction')
        If 'background', create model to infer background values (B_i) and
        scalings (c_l) using only the background measurements. If 'signal',
        create model to infer the signal (S_ij), background (B_i), and scalings
        (c_l and c_j). If 'fraction', create model to infer the fractions
        (f_ij), patch affinities (p_j), background (B_i), and scalings (c_l and
        c_j)
    robust : boolean (default True)
        If True, uses Student-T distribution for likelihood instead of normal
        distribution. Student-T degrees of freedom is treated as a parameter
        and marginalized. The Student-T approach reduces the effects of
        outliers on the inferences.
    Kblocked : boolean (default )
        If True and type='fraction', infers the patch affinities using a model
        that includes the possibility that a blocked probe can bind.  The
        parameter describing this interaction is Kblocked.  If False and
        type='fraction', we model the blocked probe signals as equal to -p_j.
        If type is not 'fraction', this parameter is ignored.
    """
    def __init__(self, dataset, name='', background_scale=100,
                 model_type='signal', robust=True, Kblocked=False):
        """
        Creates pymc model for inference on microarray patch-probe datasets.
        """
        self.dataset = dataset
        self.background_scale = background_scale
        self.model_type = model_type
        self.robust = robust
        self.Kblocked = Kblocked

        # Before we define the model, we need to set up a few arrays. The setup
        # is a bit complicated because we need to keep track of two sets of
        # coordinates. The first includes the actual probe and patch numbers of
        # the data. The second includes the renumbered patches and probes
        # (renumbered so that the first probe in the dataset is 0, second is 1,
        # etc.). The actual probe and patch numbers can vary depending on
        # whether the dataset has been truncated, decimated, or sliced in some
        # other way. To easily interface with pymc, we convert the data to
        # numpy arrays, which do not retain the actual patch and probe numbers.
        # To index into these arrays, we must use the renumbered coordinates.
        # So we do all our inference calculations in the renumbered coordinate
        # system. We then use the "coords" argument to tell pymc to add back
        # the actual probe and patch numbers to the coordinates of the inferred
        # quantities.

        # the following code assigns the renumbered probes and patches (and
        # repeats) as new coordinates for the dataset
        ds = dataset.assign_coords({'probe_renum' :
                                    ('probe', np.arange(dataset.coords['probe'].shape[0])),
                                    'patch_renum' :
                                    ('patch', np.arange(dataset.coords['patch'].shape[0])),
                                    'repeat_renum' :
                                    ('repeat', np.arange(dataset.coords['repeat'].shape[0])),})

        num_probes = ds.ppdata.num_probes
        num_patches = ds.ppdata.num_patches
        num_repeats = ds.ppdata.num_repeats

        intensity = ds.patchprobe
        background = ds.background

        # convert to narrow (tidy) format, eliminating the NaNs, which correspond
        # to spots that do not exist for a given background measurement or
        # patch, probe combination.  pymc will try to impute these values as
        # missing data if we do not get rid of them.
        background_df = background.to_dataframe().dropna()
        data_df = intensity.to_dataframe().dropna()
        # now convert to renumbered probe and patch coordinates and drop the extra
        # columns corresponding to actual patch and probe coordinates
        data_df = data_df.reset_index().set_index(['spot', 'probe_renum',
                                                'patch_renum'])['patchprobe']
        background_df = background_df.reset_index().set_index(['spot', 'repeat_renum',
                                                            'probe_renum'])['background']

        if model_type=='fraction':
            # find the blocked probes and corresponding patches
            blocked_probes, blocked_patches = ds.ppdata.find_blocked_probes()

            # make a mask array in which elements are 0 if the patch, probe
            # combination reflects a blocked probe
            is_unblocked = np.ones([num_probes, num_patches])
            blocked_probes_renum = ds.coords['probe_renum'].sel(probe=blocked_probes)
            blocked_patches_renum = ds.coords['patch_renum'].sel(patch=blocked_patches)
            is_unblocked[blocked_probes_renum, blocked_patches_renum] = 0
            # invert
            is_blocked = 1 - is_unblocked

        # construct index arrays to associate each data point with a patch-probe
        # combination.  This approach allows us to handle the variable number of
        # replicates for each patch, probe combination.
        def make_idx(df, level):
            return df.index.get_level_values(level).to_numpy(dtype='int')

        probe, patch = tuple(make_idx(data_df, level) for level
                             in ['probe_renum', 'patch_renum'])
        bg_probe, repeat = tuple(make_idx(background_df, level) for level
                                     in ['probe_renum', 'repeat_renum'])

        # convert to numpy arrays so that pymc can process.
        data = data_df.to_numpy().squeeze()
        bg = background_df.to_numpy().squeeze()

        # now tell pymc the actual patch and probe numbers. Note that since we
        # normalize the scalings by the first background well, we start the
        # 'repeat' coordinate at 1 and not 0
        coords = {'probe': background.coords['probe'].to_numpy(),
                'patch': intensity.coords['patch'].to_numpy(),
                'repeat': background.coords['repeat'][1:].to_numpy()}

        # by calling the pm.Model() constructor, we are now in the context and
        # can define parameters for our model just as we would if we were using
        # "with model:"
        super().__init__(name, coords=coords)

        # Exponential prior on inferred background values so that they are
        # constrained to be positive. We choose the order of magnitude of the
        # background to set the scale on the exponential.
        B = pm.Exponential('B', dims='probe', lam=1/background_scale)

        # hierarchical model for patch concentration scalings. We expect patch
        # concentrations to vary by about 10%, so the scale factor should be
        # order 1, and the prior should be normal. We'll set hyperpriors on the
        # parameters of the normal prior, so that we can infer the variation in
        # scalings
        mu_c = pm.Normal('mu_c', mu=1, sigma=0.1)
        sigma_c = pm.Exponential('sigma_c', lam=1/0.1)
        # set the reference amount by setting the scale factor
        # for the first measurement to 1.
        cl = at.concatenate([np.ones(1),
                            pm.Normal('cl', dims='repeat', mu=mu_c, sigma=sigma_c)])
        if model_type != 'background':
            # scaling factors for the patch-probe wells. We use the same
            # hyperparameters (mu_c and sigma_c) as for the background wells,
            # since we expect that pipetting noise is the source of variation
            # in both cases
            c = pm.Normal('c', dims='patch', mu=mu_c, sigma=sigma_c)

        # We will infer the uncertainty on the background (and the uncertainty
        # on the patch-probe measurements, if model_type='signal' or
        # model_type='fraction') from the data. Agilent software reports 0.1
        # for the sigma value almost all of the time, so we'll use 0.1 as the
        # mean of an exponential prior on the uncertainties.
        sigma = pm.Exponential('sigma', lam=1/0.1)

        # Likelihood function for background. bg_probe is an index array
        # that accounts for the variable number of replicates in each repeat
        # measurement of the background for each probe
        mu_B = cl[repeat]*B[bg_probe]

        if robust:
            # Student-T likelihood approach to dealing with outliers:
            # see https://docs.pymc.io/en/v3/pymc-examples/examples/generalized_linear_models/GLM-robust-with-outlier-detection.html#4.-Simple-Linear-Model-with-Robust-Student-T-Likelihood
            # I follow the approach above but use Gamma(2, 0.1) for the prior on
            # the Student-T degrees of freedom.  This prior samples better than
            # InverseGamma, and is what STAN recommends:
            # see https://statmodeling.stat.columbia.edu/2015/05/17/do-we-have-any-recommendations-for-priors-for-student_ts-degrees-of-freedom-parameter/
            # and https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations#prior-for-degrees-of-freedom-in-students-t-distribution
            nu_obs_B = pm.Gamma("nu_obs_B", alpha=2, beta=0.1)
            y_obs_B = pm.StudentT("y_obs_B", mu=mu_B, sigma=bg*sigma, nu=nu_obs_B, observed=bg)
        else:
            y_obs_B = pm.Normal('y_obs_B', mu=mu_B, sigma=bg*sigma, observed=bg)

        if model_type == 'signal':
            # signal prior.  We us a Laplace prior since we know the order of
            # magnitude of the absolute value of the signal is approximately
            # unity, but the signal can be positive or negative.  Laplace prior
            # is a double-sided exponential distribution and is also the
            # maximum entropy distribution when the scale of the parameter is
            # known and the parameter is a real number (either positive or
            # negative).
            S = pm.Laplace('S', dims=('probe', 'patch'), mu=0, b=1)

            # mean to be used in the likelihood for the signal (pymc will add
            # the log likelihood for the signal to the log likelihood for the
            # background specified above). probe and patch are index arrays
            # that account for the variable number of measurements for each
            # patch, probe combination
            mu = c[patch] * B[probe] * (1 + S[probe, patch])

        if model_type == 'fraction':
            # for patch probabilities, use a nearly flat Beta distribution,
            # which samples better than Uniform
            p_j = pm.Beta('p', alpha=2, beta=2, dims='patch')

            # use a similar uninformative prior for the fractions
            f = pm.Beta('f', alpha=2, beta=2, dims=('probe', 'patch'))

            # Kblocked_j accounts for possibility that blocked probes bind to
            # an RNA molecules with patch j attached. We assume Kblocked_j is a
            # constant for each patch. We use an exponential prior with a small
            # mean (0.1) so that Kblocked_j will be close to zero unless the
            # data strongly suggests that the signal is positive for a blocked
            # probe.
            if Kblocked:
                Kblocked_j = pm.Exponential('Kblocked', lam=10, dims='patch')

            # code for making Kblocked depend on both blocked patch and blocked probe
            #Kblocked_j = pm.Exponential('Kratio', lam=10, shape=(num_probes, num_patches))

            # We then estimate the ratios and signals from the values for all
            # the other parameters.  Note that we could do this the other way
            # around, setting a prior on r and inferring f, but setting an
            # uninformative prior on f is easier because it is constrained to
            # (0,1), whereas r can take on any value above 0.
            r = pm.Deterministic('r', f/(1-f), dims=('probe', 'patch'))
            if Kblocked:
                S = pm.Deterministic('S', (p_j * r * is_unblocked
                                           -p_j * (1 - Kblocked_j) * is_blocked),
                                    dims=('probe', 'patch'))
            else:
                S = pm.Deterministic('S', (p_j * r * is_unblocked
                                        -p_j * is_blocked),
                                    dims=('probe', 'patch'))

            # mean to be used in the likelihood function
            if Kblocked:
                mu = c[patch] * B[probe] * \
                    (1 + (-p_j[patch] * (1 - Kblocked_j[patch]) * is_blocked[probe, patch] +
                        p_j[patch] * r[probe, patch] * is_unblocked[probe, patch]))
            else:
                mu = c[patch] * B[probe] * \
                    (1 + (-p_j[patch] * is_blocked[probe, patch] +
                        p_j[patch] * r[probe, patch] * is_unblocked[probe, patch]))
            # mean for when Kblocked depends on both blocked patch and blocked probe
            #mu = c[patch] * B[probe] * \
            #    (1 + (-p_j[patch] * (1 - Kblocked_j[probe, patch]) * is_blocked[probe, patch] +
            #           p_j[patch] * r[probe, patch] * is_unblocked[probe, patch]))

        if model_type != 'background':
            if robust:
                # Use Student-T for patch-probe measurements as well (see above)
                nu_obs = pm.Gamma("nu_obs", alpha=2, beta=0.1)
                y_obs = pm.StudentT("y_obs", mu=mu, sigma=data*sigma, nu=nu_obs, observed=data)
            else:
                y_obs = pm.Normal('y_obs', mu=mu, sigma=data*sigma, observed=data)

    def sample(self, summary=True, idata_path=None, summary_path=None,
               **pmsample_kwargs):
        """
        Wrapper around pm.sample() to ensure that the resulting InferenceData
        object includes the patch-probe dataset and associated metadata. This
        function can optionally save the InferenceData and summary statistics
        to disk. If they already exist, the function can load them instead of
        sampling.

        Parameters
        ----------
        summary : Boolean (default True)
            If True, return arviz summary statistics of inference data
        idata_path : Path (default None)
            If specified, save the InferenceData object (augmented with
            metadata) in the specified path. If the file already exists, the
            function will load it instead of sampling.  Function will not
            overwrite an existing file.
        summary_path : Path (default None)
            If specified, generate an arviz summary of the posterior, augment
            with metadata, and save it in the the specified path (or load it
            if it already exists).  Function will not overwrite an existing
            summary file.
        pmsample_kwargs :
            keyword arguments (like "draws" and "init") to pass to pm.sample()
        """

        # PatchprobeModel.sample() is expected to return an InferenceData
        # object. By default it launches the sampler unless idata_path is
        # specified, in which case it tries to find the saved file.
        do_sampling = True
        if idata_path is not None:
            idata_file = idata_path / self._idata_filename()
            if idata_file.is_file():
                # the dataset doesn't conform to the InferenceData scheme but it can
                # still be loaded and stored, so we ignore the arviz warning
                print("Loading existing inference data file: " +
                      str(idata_file.relative_to(Path.cwd())))
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="patchprobe_dataset group is not defined in the InferenceData scheme")
                    idata = az.from_netcdf(idata_file)
                print("Original sampling time was " +
                      str(datetime.timedelta(seconds=
                                             idata.posterior.attrs['sampling_time'])))
                do_sampling = False
        if do_sampling:
            with self:
                idata = pm.sample(**pmsample_kwargs)

        # add original dataset and metadata to the InferenceData object if it
        # isn't already there. The dataset will then be saved along with the
        # MCMC results when the object is serialized to netCDF format. We do
        # this because the "observed_data" holds the observed data in narrow
        # format only, and doesn't keep the metadata
        if 'patchprobe_dataset' not in idata.groups():
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="The group patchprobe_dataset is not defined in the InferenceData scheme")
                idata.add_groups({'patchprobe_dataset': self.dataset})

        # save the InferenceData object if requested
        if idata_path is not None:
            if not idata_file.is_file():
                print(f"Saving inference data as {idata_file.relative_to(Path.cwd())}")
                idata.to_netcdf(idata_file)

        if summary:
            if summary_path is not None:
                summary_file = summary_path / self._summary_filename()

            # always regenerate the summary if we have sampled again.  Also
            # generate summary if no summary file exists
            generate_summary = True

            if not do_sampling:
                # idata was loaded and sampling was skipped, so check for the
                # summary file and load it if it exists. Only in this case do
                # we skip generating the summary
                if summary_path is not None:
                    if summary_file.is_file():
                        print(f"Loading existing summary file: " +
                              str(summary_file.relative_to(Path.cwd())))
                        summary_stats = xr.open_dataset(summary_file)
                        generate_summary = False
            if generate_summary:
                print("Generating summary statistics")
                summary_stats = az.summary(idata, fmt='xarray')

            # check if microarray data and metadata are already in the summary
            # dataset.  If not, add them.
            if 'background' not in summary_stats.var().keys():
                # This will merge in metadata and patchprobe, background, and
                # error arrays. It will also add an extra value to the 'repeat'
                # dim. Since cl is not sampled for the first repeat, the values
                # of summary_stats['cl'].sel(repeat=1) will all be nan. For
                # calculations and plotting, just need to be careful to handle
                # the nans when looking at cl. No other variables should be
                # affected.
                summary_stats = xr.merge((summary_stats, self.dataset),
                                         combine_attrs='no_conflicts')
                # now add metadata from the inference data array
                summary_stats.attrs.update(idata.posterior.attrs)
                # finally, add metadata from the model object.  Note that
                # netCDF does not store Boolean types, so we must convert to
                # string.  Could convert to int, but string 'True' or 'False'
                # is more readable for humans.
                model_metadata = {"model_type" : self.model_type,
                                  "background_scale" : self.background_scale,
                                  "robust" : str(self.robust),
                                  "Kblocked" : str(self.Kblocked)}
                summary_stats.attrs.update(model_metadata)

            # save the summary statistics if requested
            if summary_path is not None:
                if summary_file.is_file():
                    if do_sampling:
                        print("No summary statistics saved to disk "
                              "because there is an existing summary file.\n"
                              "To regenerate the summary statistics, " +
                              f"delete {summary_file} and run again")
                else:
                    print("Saving summary statistics as " +
                          str(summary_file.relative_to(Path.cwd())))
                    summary_stats.to_netcdf(summary_file)

        if summary:
            return idata, summary_stats
        else:
            return idata

    def _filename(self, filetype, extension=".nc"):
        """
        Utility function to generate filename for saving inference data from
        pymc/arviz using a filename that includes metadata about the dataset
        analyzed as well as the model.

        Parameters
        ----------
        filetype : string
            Type of file to save: "idata" for inference data and "summary" for summary
        extension : string (default ".nc")
            Filename extension
        """

        dataset = self.dataset
        filename_root = dataset.ppdata._filename(extension='')
        filename = filename_root + "-" + self.model_type
        if not self.robust:
            filename = filename + "-gaussian"
        if (self.model_type == 'fraction') and self.Kblocked:
            filename = filename + "-Kblocked"
        filename = filename + "-" + filetype + extension

        return filename

    def _idata_filename(self, extension=".nc"):
        return self._filename("idata", extension)

    def _summary_filename(self, extension=".nc"):
        return self._filename("summary", extension)

def _parse_summary_filename(summary_file):
    """
    Parse the filename of a summary statistics file to determine the model that
    was used

    Parameters
    ----------
    summary_file : Path
        Path object pointing to summary statistics file

    Returns
    -------
    dict :
        Dictionary with information on model used to generate file

    """
    if summary_file.is_file()==False:
        raise FileNotFoundError(f"Summary file \"{summary_file}\" not found")

    name = summary_file.stem
    parts = str.split(name, "-")

    result_dict = {}

    non_conforming_message = f"filename {summary_file} does not correspond to naming convention; parsing may not be correct"
    if parts[-1] != 'summary':
        warnings.warn(non_conforming_message)

    # filename root (parts[0]) consists of metadata about the experiment:
    # shortID, RNA, probe_size, patch_size, patch_protocol, scan_file, processing_method
    # we don't need to parse this info since it is contained in the metadata for the
    # dataset
    result_dict['filename_root'] = parts[0]

    if parts[1] not in ['background', 'fraction', 'signal']:
        warnings.warn(non_conforming_message)
    result_dict['model_type'] = parts[1]

    if 'gaussian' in parts[1:]:
        result_dict['robust'] = False
    else:
        result_dict['robust'] = True

    if result_dict['model_type'] == 'fraction':
        if 'Kblocked' in parts[1:]:
            result_dict['Kblocked'] = True
        else:
            result_dict['Kblocked'] = False

    return result_dict

def find_summaries(dataset, path, model_type=None, robust=True, Kblocked=False):
    """
    Look for files containing summary statistics for a given dataset

    Parameters
    ----------
    dataset : xarray Dataset
        Patch-probe dataset
    path : Path
        Directory to search
    model_type : string
        Restrict search to a particular type of model, typically "signal" or
        "fraction"
    robust : boolean
        If True, look only for models that have robust inference; if False,
        look only for models that do not have robust inference
    Kblocked : boolean
        As with robust, if True, look only for models with Kblocked parameter.
        If False, look only for models without Kblocked. Only relevant to
        fraction models.

    Returns
    -------
    list of dicts:
        list of dictionaries containing model information for each summary file
        that matches the search criteria
    """

    file_list = list(path.glob(dataset.shortID + "_*-summary.nc"))
    model_info_list = [_parse_summary_filename(name) for name in file_list]
    if model_type is None:
        return file_list
    else:
        pruned_list = []
        for file, model_info in zip(file_list, model_info_list):
            flag = False
            if ((model_info['model_type'] == model_type) and 
                (model_info['robust'] == robust)):
                flag = True
            if model_info['model_type'] == "fraction":
                if model_info['Kblocked'] != Kblocked:
                    flag = False
            if flag:
                pruned_list.append(file)
        return pruned_list
