'FeatureExtraction.m' is a Matlab script that implements our custom algorithm for extracting integrated fluorescence intensities from a raw microarray image. The script and related files listed below are specifically written for the raw microarray scan image file called 'SG15294487_258644410004_S001.tif'.

'SubgridLayout.txt' contains an array that represents the positions of every probe sequence in each subgrid. The values of the array elements are the probe numbers. Dark corners (upper left and lower right) are designated values -1, Bright corners (upper right and lower left) are designated values -2, and Free spots are designated value 0.

'SubgridBounds.txt' is a file that contains the lower row boundary, upper row boundary, lower column boundary, and upper column boundary for all 48 subgrids.

'FirstFeatureCoordinates.txt' contains the (x,y) coordinate of the first feature (spot) of the subgrid, within each subgrid image. This information is needed to align the probe numbers in the subgrid layout array to the spots in the image.

'key_12mers.txt' and 'key_24mers.txt' convert the probe numbers to the 5' nucleotide of their binding site on the RNA sequence.
