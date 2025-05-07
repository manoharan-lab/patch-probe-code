function designProbesFromSequence
% This function reads the sequence for STMV RNA from a text file and
% generates every 12-nt and 24-nt long DNA probe sequences. Poly-T tethers
% are added to each probe sequence for a final length of 60-nts, which will
% be printed on the Agilent DNA microarray.
%
% Copyright 2022-2025 Timothy K. Chiang
%
% This program is free software: you can redistribute it and/or modify it under
% the terms of the GNU General Public License as published by the Free Software
% Foundation, either version 3 of the License, or (at your option) any later
% version.
%
% This program is distributed in the hope that it will be useful, but WITHOUT
% ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
% FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License along with
% this program. If not, see <https://www.gnu.org/licenses/>.

STMV_RNA = readSequenceFile('STMV_RNA_sequence.txt');
L = length(STMV_RNA);

tether_12 = 'TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT';
tether_24 = 'TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT';

probeSequences_Unsorted = cell((L-11)+(L-23),1);
index = 1;

% Generate all 12-nt probes
for i=1:L-11
    probeSequence = RNA_2_rcDNA(STMV_RNA(i:i+11));
    probeSequences_Unsorted{index} = [probeSequence,tether_12];
    index = index + 1;
end

% Generate all 24-nt probes
for i=1:L-23
    probeSequence = RNA_2_rcDNA(STMV_RNA(i:i+23));
    probeSequences_Unsorted{index} = [probeSequence,tether_24];
    index = index + 1;
end

% Write to text files: ProbeNames_and_Sequences.txt
sort_and_write_ProbeNames_Sequences_ToFile(probeSequences_Unsorted,'ProbeNames_and_Sequences.txt');

end

%%
function sequence = readSequenceFile(sequenceFile)
% Read a text file containing RNA sequence information.

fileID = fopen(sequenceFile,'r');
sequence = fscanf(fileID,'%s');
fclose(fileID);

end

%%
function rcDNA_sequence = RNA_2_rcDNA(RNA_sequence)
% This function takes an RNA sequence, specified in the 5'-3' direction, and
% outputs the reverse complement DNA sequence, also in the 5'-3' direction.

L = length(RNA_sequence);
rcDNA_sequence = blanks(L);

fwd = 'AUCG';
rev = 'TAGC';

for nt=1:L
    rcDNA_sequence(nt) = rev(RNA_sequence(nt)==fwd);
end

rcDNA_sequence = flip(rcDNA_sequence);

end

%%
function sort_and_write_ProbeNames_Sequences_ToFile(probeSequences_Unsorted,file)
% This function sorts and writes probenames and sequences to a file

probeSequences_Sorted = sort(probeSequences_Unsorted);

% Write ProbeNames and Sequences
fileID = fopen(file,'w');
fprintf(fileID,'%s\t%s\n','ProbeName','Sequence');
for i=1:length(probeSequences_Sorted)
    probename = sprintf('probe%04d',i);
    fprintf(fileID,'%s\t%s\n',probename,probeSequences_Sorted{i});
end
fclose(fileID);

end
