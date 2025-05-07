"""
This module contains a class for working with processed patch-probe data from
Agilent microarrays.

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

import numpy as np
import pandas as pd
import xarray as xr
import copy
import proplot as pplt
import warnings
from pathlib import Path

@xr.register_dataset_accessor('ppdata')
class PatchprobeAccessor:
    """
    Class to extend xarray Dataset object used to hold patch-probe data/metadata
    with some convenience methods.  See
    https://docs.xarray.dev/en/stable/internals/extending-xarray.html

    Attributes
    ----------
    xarray_obj : xarray Dataset
        Dataset with patch-probe, background measurements and metadata
    """
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._num_probes = None
        self._num_patches = None
        self._num_repeats = None

    @property
    def num_probes(self):
        """
        Return number of probes in dataset
        """
        if self._num_probes is None:
            self._num_probes = int(self._obj.patchprobe.coords['probe'].count())
        return self._num_probes

    @property
    def num_patches(self):
        """
        Return number of patches in dataset
        """
        if self._num_patches is None:
            self._num_patches = int(self._obj.patchprobe.coords['patch'].count())
        return self._num_patches

    @property
    def num_repeats(self):
        """
        Return number of repeat measurements (wells) of background
        """
        if self._num_repeats is None:
            self._num_repeats = int(self._obj.background.coords['repeat'].count())
        return self._num_repeats

    def normalization_weights(self):
        """
        Calculate weights for normalizing patch-probe spectra. Function
        calculates the weights by first taking the mean over replicates and
        then the median over all probes for each patch. These averages are then
        inverted and scaled by the mean of the weights over all patches. The
        reason for taking the median over all probes is that we expect that
        there will be some large but sparse signals in the patch-probe spectra.
        We don't want to weight these signals because they don't tell us much
        about the concentration variations.

        Parameters
        ----------
        None

        Returns
        -------
        c_j : pandas Series
            Contains weights for each patch, indexed by patch.  To normalize
            patch-probe spectra, multiply it by the weights.
        """

        # average values over replicates for each patch, probe
        replicate_averages = self._obj.patchprobe.mean('spot')
        # then average over all probes for each patch
        probewise_averages = replicate_averages.median('probe')
        # finally, average value across all patches
        overall_average = probewise_averages.mean()

        return overall_average/probewise_averages

    def normalize(self, weights = None):
        """Normalize patch-probe spectra to remove patch-to-patch variations arising
        from pipetting noise.

        Parameters
        ----------
        weights : array
            Array of normalization weights, one for each patch. The values for
            each patch will be multiplied by the weights. If not specified, the
            normalization weights will be calculated from the
            normalization_weights() function.

        Returns
        -------
        normalized_dataset : xr.Dataset
            Dataset in which the patchprobe attribute has been normalized. All
            other attributes are copied from the original dataset. Note that in
            the normalized data set, we use the overall average to set the
            scale of the values, rather than normalize to some arbitrary value,
            because we need to compare all values to the background, which has
            not been normalized.
        """

        normalized_dataset = self._obj.copy()

        if weights is None:
            weights = self.normalization_weights()

        # TODO: fix so that this works with xarray
        unnormalized = self._obj['patchprobe']
        normalized = unnormalized*weights

        normalized_dataset['patchprobe'] = normalized

        return normalized_dataset

    def find_blocked_probes(self):
        """
        Find the indices of probes that cannot (in principle) bind when a patch
        binds. The base sequences of each such probe is a complete subsequence
        of the patch.

        Returns
        -------
        array, array :
            arrays of blocked probe numbers and associated patches.  Can be
            used as an index arrays to select signals for only those probes.
        """

        patch_size = self._obj.patch_size
        probe_size = self._obj.probe_size
        patches = self._obj.coords['patch']
        probes = self._obj.coords['probe']
        blocked_probes_per_patch = patch_size - probe_size + 1

        # the following assumes probes are numbered starting from 1 (otherwise
        # "probes-1" would become "probes")
        blocked_probes = probes.where((probes-1) % patch_size < blocked_probes_per_patch).dropna('probe').astype('int').to_numpy()
        # the following assumes patches are numbered starting from 1 (otherwise
        # no +1 at the end)
        blocked_patches = (blocked_probes/patch_size).astype(int) + 1


        return blocked_probes, blocked_patches

    def _filename(self, extension='.nc'):
        """
        Utility function to generate a filename that is unique for a given RNA,
        patch size, probe size, patch attachment protocol, and microarray scan.

        Parameters
        ----------
        extension : string (default ".nc")
            Extension for file
        """

        # attributes are passed by reference, so we make sure to copy them to
        # avoid inadvertant modification
        metadata = self._obj.attrs.copy()

        # strip the ".tif" extension from the scan file name
        metadata['scan_file'] = str(Path(metadata['scan_file']).stem)

        root = ("{shortID}_{RNA}_{probe_size}_{patch_size}_" +
                "{patch_protocol}_{scan_file}_" +
                "{processing_method}").format(**metadata)

        return root + extension

    def save(self, save_path, filename=None):
        """
        Save dataset and metadata in netCDF format

        Parameters
        ----------
        save_path : Path
            Directory to save file in
        filename : string (default None)
            Name of file. If not specified, the function generates a filename
            from the metadata
        """

        if filename is None:
            filename = self._filename()

        self._obj.to_netcdf(save_path / Path(filename))
