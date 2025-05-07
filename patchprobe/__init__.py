"""
The patchprobe package contains functions and classes for working with processed
data from Agilent microarrays.

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
from .io import read_csv_metadata, read_csv_dataset, read_csv_datasets, \
    read_dataset, read_datasets
from .plot import plot_raw, plot_signal_posterior, plot_posterior_scatter, \
    plot_prevalence, plot_prevalence_posterior, plot_hyperparameter_posterior
from .infer import PatchprobeModel, find_summaries
from .util import dataset_from_summary


