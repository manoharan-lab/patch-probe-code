"""
This module contains functions for reading and writing data as well as
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

import os
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from types import SimpleNamespace

def _read_csv_datafile(filename):
    """
    Utility function to read in CSV formatted data file with a variable number
    of columns. For such a file, `pandas.read_csv()` will given an error.
    Following the advice at https://stackoverflow.com/a/55129746, we read in
    the file without using the comma separator, then tell pandas to split the
    lines in the resulting DataFrame.

    Parameters
    ----------
    filename : string
        Name of variable-column-number CSV file to open

    Returns
    -------
    DataFrame :
        DataFrame padded with NaN values in columns where there
        is no data.
    """
    df = pd.read_csv(filename, header=None, sep='\r')
    return df[0].str.split(',', expand=True).astype(float)

def read_csv_metadata(metadata_path, comment='#', skipinitialspace=True, index_col='key'):
    """
    Read in metadata about the available CSV-formatted datasets from a CSV file

    Parameters
    ----------
    metadata_path : Path
        Path to CSV file with columns corresponding to metadata
    comment : str (default '#')
        comment separator in CSV file
    skipinitialspace : boolean (default True)
        skip whitespace after delimiter
    index_col : string (default 'key')
        column in the data file that contains key names (short-form
        identifiers)

    Returns
    -------
    df : pandas DataFrame
        Columns contain metadata for each dataset, indexed by short-form identifier

    """
    # determine path to data directories from metadata_path
    data_path = metadata_path.parent

    df = pd.read_csv(metadata_path, comment='#', skipinitialspace=True,
                     index_col='key')
    # need to replace nans with None so that dictionary processing works
    df = df.replace({np.nan: None})

    return df

def read_csv_dataset(metadata):
    """
    Read in patch-probe and background spectra for a given probe size, along
    with associated metadata.  This function relies on a schema for the
    metadata which is specified below.

    Parameters
    ----------
    metadata : dictionary
        Dictionary containing metadata, as described below:
        key : string
            Short-form identifier to be used as a key for the data set
            (e.g. "sdsu12")
        directory : Path
            Directory containing the data for the given probe size and feature
            extraction method.  Data files in the directory are in CSV format.
        scan_file : string
            name of image file (.tif) from microarray scanner that was used in the
            feature extraction. This file should be in the data directory
            ("directory"). First part of the filename is the ID of the scanner.
            Second part is the barcode on the array
        subdirectory : string
            Name of subdirectory with feature extraction files
            (e.g. "AgilentFeatureExtracted")
        processing_method : string
            Method of feature extraction ("agilent" or "matlab")
        probe_size : int
            Size of probe used
        patch_size : int
            Size of patches
        description : string
            Human-readable description of data set.
        output_var : string
            Name of output variable.  For Agilent-processed data, use
            "gProcessedSignal". For Matlab-processed data, use "Signal"
        identifier : string
            Identifier for dataset embedded in filename (e.g. "SDSU_NoHeat").
        patch_string : string
            String used in filename to indicate spectrum with patch ("Patch_")
        mer_string : string
            String following probe size in filename ("mer")
        background_string : string
            String used in filename to indicate background spectrum ("NakedRNA")
        extension: string
            Extension for CSV formatted data files (".txt")
        output_var_directory : string (default None)
            Directory containing data files for the output variable.  For
            Agilent-processed data, this is "gProcessedSignal".  If None,
            the data are not in a separate subdirectory
        error_string : string (default None)
            String used in filename to indicate error files.  If None,
            don't return errors.  If specified (should be "gProcessedSignalError"
            for Agilent-processed data), return errors
        error_var_directory : string (default None)
            Directory containing error files. For Agilent-processed data, this is
            "gProcessedSignalError". If None, the errors are not in a separate
            subdirectory

    Returns
    -------
    xarray Dataset object
        with DataArray objects for patch-probe measurements, background, and
        uncertainties, and the rest of the metadata stored as attributes
    """

    # for convenience, load all the dictionary key/value pairs into variables
    # in a new namespace
    m = SimpleNamespace(**metadata)

    probe_string = str(m.probe_size) + m.mer_string
    data_dir = m.directory / Path(m.subdirectory) / Path(probe_string + "s")

    # check for existence of data_dir
    if data_dir.is_dir()==False:
        raise FileNotFoundError(f'Data directory {data_dir} not found')

    spectra_dir = data_dir
    error_dir = data_dir
    if m.output_var_directory is not None:
        spectra_dir = spectra_dir / Path(m.output_var_directory)
    if m.error_string is not None and m.error_var_directory is not None:
        error_dir = error_dir / Path(m.error_var_directory)

    file_prefix = m.output_var + "_" + probe_string + "Probes_" + m.identifier + "_"
    if m.error_string is not None:
        error_file_prefix = m.error_string + "_" + probe_string + "Probes_" + m.identifier + "_"
    spectra_files = list(spectra_dir.glob(file_prefix + m.patch_string + "*" + m.extension))
    background_files = list(spectra_dir.glob(file_prefix + m.background_string + "*" + m.extension))

    background_list = []
    num_repeats = len(background_files)
    for i in range(num_repeats):
        # files are numbered starting from 1, not 0
        repeat_num = i + 1
        filename = spectra_dir / (file_prefix + m.background_string + str(repeat_num) + m.extension)
        df = _read_csv_datafile(filename)
        background_list.append(df)

    spectra_list = []
    # use the number of files to determine number of patches
    num_patches = len(spectra_files)
    if m.error_string is not None:
        spectra_error_list = []

    for i in range(num_patches):
        # files are numbered starting from 1, not 0
        patch_num = i + 1
        filename = spectra_dir / (file_prefix + m.patch_string + str(patch_num) + m.extension)
        df = _read_csv_datafile(filename)
        spectra_list.append(df)
        # note that this doesn't load errors for background data; just the patch-probe errors
        if m.error_string is not None:
            filename = error_dir / (error_file_prefix + m.patch_string + str(patch_num) + m.extension)
            # if error files don't exist, then pass and return only
            # the patch-probe and background spectra
            try:
                df = _read_csv_datafile(filename)
                spectra_error_list.append(df)
            except:
                pass

    # use the number of rows in the dataframe to determine number of probes
    num_probes = len(spectra_list[0])

    # Convert to MultiIndex DataFrame. We could use xarray for this kind of
    # data, but it's easier to deal with ragged data in pandas, since it pads
    # with NaNs automatically. The raggedness is due to the different number of
    # replicates for each patch, probe measurement. By convention, we number
    # patches and probes starting from 1
    spectra = pd.concat(spectra_list)
    spectra.index = pd.MultiIndex.from_product((range(1, num_patches+1),
                                                range(1, num_probes+1)),
                                               names=['patch', 'probe'])
    if m.error_string is not None:
        errors = pd.concat(spectra_error_list)
        errors.index = pd.MultiIndex.from_product((range(1, num_patches+1),
                                                   range(1, num_probes+1)),
                                                  names=['patch', 'probe'])
    else:
        errors = None

    # MultiIndex the background spectra into one DataFrame, indexed by (repeat, probe)
    # By convention, we number repeats and probes starting from 1
    background_spectra = pd.concat(background_list)
    background_spectra.index = pd.MultiIndex.from_product((range(1, num_repeats+1),
                                                           range(1, num_probes+1)),
                                                          names=['repeat', 'probe'])

    # Now convert to Dataset
    patchprobe = xr.DataArray(spectra).unstack().rename({'dim_1': 'spot'})
    background = xr.DataArray(background_spectra).unstack().rename({'dim_1': 'spot'})
    if errors is not None:
        errors = xr.DataArray(errors).unstack().rename({'dim_1': 'spot'})
        dataset = xr.Dataset({'background': background,
                              'patchprobe': patchprobe,
                              'errors': errors})
    else:
        dataset = xr.Dataset({'background': background,
                              'patchprobe': patchprobe})
    # add metadata
    attrs = {'RNA' : m.RNA,
             'probe_size' : m.probe_size,
             'patch_size' : m.patch_size,
             'patch_protocol': m.patch_protocol,
             'scan_file' : m.scan_file,
             'processing_method' : m.processing_method,
             'csv_directory' : str(data_dir.relative_to(m.directory.parent)),
             'output_var' : m.output_var,
             'description' : m.description,
             'shortID' : m.shortID
             }
    dataset.attrs.update(attrs)

    return dataset

def read_csv_datasets(data_dir, metadata_df):
    """
    Read in several CSV-formatted datasets at once

    Parameters
    ----------
    data_dir : Path
        path to root directory of data files
    metadata_df : pandas DataFrame
        Metadata for the datasets, as output from read_metadata()

    Returns
    -------
    dict :
        {short-form identifier: xr.Dataset}

    """

    dataset_dict = {}
    for shortname in metadata_df.index:
        kw_dict = metadata_df.loc[shortname].to_dict()

        # add the short-form identifier itself to the dictionary
        kw_dict['shortID'] = shortname

        # prefix data directory with path
        if 'directory' in kw_dict.keys():
            kw_dict['directory'] = data_dir / Path(kw_dict['directory'])
        dataset_dict[shortname] = read_csv_dataset(kw_dict)

    return dataset_dict

def read_dataset(data_file):
    """
    Read in a dataset in netCDF format

    Parameters
    ----------
    data_file : Path
        path to netCDF data file

    Returns
    -------
    dataset :
        xarray Dataset with patch-probe, background, and (if available) errors
        stored as DataArrays
    """

    return xr.open_dataset(data_file)

def read_datasets(data_dir, extension='.nc'):
    """
    Read in several netCDF datasets at once

    Parameters
    ----------
    data_dir : Path
        path to directory where netCDF data files are stored
    extension : string (default ".nc")
        extension of data files. All files with this extension will be loaded.

    Returns
    -------
    dict :
        {short-form identifier: xr.Dataset}

    """

    data_files = list(data_dir.glob("*" + extension))

    # make sure the order of the files matches the modification time. This will
    # (hopefully) ensure that the dictionary is sorted in the same order as the
    # the CSV metadata file used to convert the CSV data files to netCDF.
    data_files.sort(key=os.path.getmtime)

    dataset_dict = {}
    for file in data_files:
        dataset = read_dataset(file)
        dataset_dict[dataset.attrs['shortID']] = dataset.copy()

    return dataset_dict
