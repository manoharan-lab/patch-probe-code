'STMV_RNA_sequence.txt' contains the nucleotide sequence of the RNA transcript, as determined by Sanger sequencing of the cDNA.

'designProbesFromSequence.m' is a Matlab function that reads the STMV RNA sequence. It generates a list of unique probe names and probe sequences in an output file called 'ProbeNames_and_Sequences.txt'.

'generateMicroarray.m' is a script that reads the list of probes in output file from the previous step and generates a randomly distributed square subgrid, including a custom pattern bright and dark spots in the corners that are made of probes with polyA and polyT sequences. The subgrid is then tiled across the Agilent 1x1M microarray in a rectangular grid. Finally, it outputs the following text files:
 - 'SubgridLayout.txt' contains an array that represents the positions of every probe sequence in the random subgrid. The values of the array elements are the probe numbers. Dark corners (upper left and lower right) are designated values -1, Bright corners (upper right and lower left) are designated values -2, and Free spots are designated value 0.
 - 'STMV_1MGrid.txt' lists the Column, Row, ProbeID, and Sequence of every probe to be printed on the Agilent microarray.
