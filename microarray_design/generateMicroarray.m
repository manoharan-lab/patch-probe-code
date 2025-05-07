% This script generates a subgrid with a random distribution of all probes
% and tiles it across an Agilent 1x1M microarray.
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

nProbes = getNumProbes('ProbeNames_and_Sequences.txt');
nSubgridRows = 148; % Rows in subarray
nSubgridCols = 43;  % Columns in subarray

subgrid = defineSubgrid(nSubgridRows,nSubgridCols,nProbes);
grid = subgridToFullgrid(subgrid);

writeGridsToFile(subgrid,grid);

%%
function subgrid = defineSubgrid(nSubgridRows,nSubgridCols,nProbes)

% We create a pattern of dark and bright spots in the subarray corners,
% which makes the boundaries of the subarrays visible in the final
% fluorescence scan of the microarray. We also create a uniform pattern of
% dark and bright spots if analysis of background uniformity is required.

% Dark corners and spots are polyTs
darkCorners = floor([1,1; 1,2; 1,3;
    2,1; 2,2;
    3,1; 3,2;
    4,1;
    5,1;
    nSubgridRows-4,nSubgridCols;
    nSubgridRows-3,nSubgridCols;
    nSubgridRows-2,nSubgridCols-1; nSubgridRows-2,nSubgridCols;
    nSubgridRows-1,nSubgridCols-1; nSubgridRows-1,nSubgridCols;
    nSubgridRows,nSubgridCols-2; nSubgridRows,nSubgridCols-1; nSubgridRows,nSubgridCols]);

darkSpots = floor((ones(4,2)*[nSubgridRows,0;0,nSubgridCols]).*[1,1; 1,3;3,1; 3,3]/4);

% Bright corners and spots are polyAs
brightCorners = floor([1,nSubgridCols-1; 1,nSubgridCols;
    2,nSubgridCols-1; 2,nSubgridCols;
    3,nSubgridCols;
    4,nSubgridCols;
    nSubgridRows-3,1;
    nSubgridRows-2,1;
    nSubgridRows-1,1; nSubgridRows-1,2;
    nSubgridRows,1; nSubgridRows,2]);

brightSpots = floor((ones(9,2)*[nSubgridRows,0;0,nSubgridCols]).*[1,1; 1,3; 1,5; 3,1; 3,3; 3,5; 5,1; 5,3; 5,5]/6);


% Generate lists of indices representing the rows and columns of the dark
% and bright spots in the subarray.

inds_DarkCorners = zeros(length(darkCorners),1);
for i=1:length(darkCorners)
    inds_DarkCorners(i) = sub2ind([nSubgridCols,nSubgridRows],darkCorners(i,2),darkCorners(i,1));
end

inds_BrightCorners = zeros(length(brightCorners),1);
for i=1:length(brightCorners)
    inds_BrightCorners(i) = sub2ind([nSubgridCols,nSubgridRows],brightCorners(i,2),brightCorners(i,1));
end

inds_BrightSpots = zeros(length(brightSpots),1);
for i=1:length(brightSpots)
    inds_BrightSpots(i) = sub2ind([nSubgridCols,nSubgridRows],brightSpots(i,2),brightSpots(i,1));
end

inds_DarkSpots = zeros(length(darkSpots),1);
for i=1:length(darkSpots)
    inds_DarkSpots(i) = sub2ind([nSubgridCols,nSubgridRows],darkSpots(i,2),darkSpots(i,1));
end


% Generate list of indices representing free spots available for random
% distribution of probe sequences.
inds_FreeSpots = [];
for i=1:nSubgridRows*nSubgridCols
    [r,c]=ind2sub([nSubgridRows,nSubgridCols],i);
    k=0;
    for j=1:length(darkCorners)
        if darkCorners(j,:)==[r,c]
            k=k+1;
        end
    end
    for j=1:length(brightCorners)
        if brightCorners(j,:)==[r,c]
            k=k+1;
        end
    end
    for j=1:length(brightSpots)
        if brightSpots(j,:)==[r,c]
            k=k+1;
        end
    end
    for j=1:length(darkSpots)
        if darkSpots(j,:)==[r,c]
            k=k+1;
        end
    end
    if ~logical(k)
        inds_FreeSpots = [inds_FreeSpots;i];
    end
end


% Randomly distribute the order of free spots
randomOrder = randperm(length(inds_FreeSpots));
inds_FreeSpots = inds_FreeSpots(randomOrder);

positions = cell(1,nProbes);
for i=1:nProbes
    for j=1:3
        [r,c] = ind2sub([nSubgridRows,nSubgridCols],inds_FreeSpots(1));
        positions{i} = [positions{i};[r,c]];
        inds_FreeSpots = circshift(inds_FreeSpots,1);
    end
end

% Define subgrid
subgrid = zeros(nSubgridRows,nSubgridCols);
for i=1:length(positions)
    for j=1:3
        r = positions{i}(j,1);
        c = positions{i}(j,2);
        subgrid(r,c) = i;
    end
end
for i=1:length(darkCorners)
    subgrid(darkCorners(i,1),darkCorners(i,2)) = -1;
end
for i=1:length(darkSpots)
    subgrid(darkSpots(i,1),darkSpots(i,2)) = -2;
end
for i=1:length(brightSpots)
    subgrid(brightSpots(i,1),brightSpots(i,2)) = -3;
end
for i=1:length(brightCorners)
    subgrid(brightCorners(i,1),brightCorners(i,2)) = -4;
end

end

%%
function grid = subgridToFullgrid(subgrid)
% Define full 1x1M grid from subgrid

[nSubgridRows,nSubgridCols] = size(subgrid);

grid_noedge = [];
grid = zeros(1068,912);
for r=1:7
    grid_noedge = [grid_noedge;subgrid];
end
grid_column = grid_noedge;
grid_noedge = [];
for c=1:21
    grid_noedge = [grid_noedge,grid_column];
end
start = ceil((size(grid)-size(grid_noedge))/2);

upper_flank = grid_noedge(nSubgridRows-start(1)+1:nSubgridRows,:);
lower_flank = grid_noedge(1:start(1),:);
grid = [upper_flank;grid_noedge;lower_flank];

left_flank = grid(:,nSubgridCols-start(2)+1:nSubgridCols);
right_flank = grid(:,1:start(2)-1);
grid = [left_flank,grid,right_flank];
end


%%
function nProbes = getNumProbes(file)

data = importdata(file);
nProbes = length(data)-1;

end


%%
function writeGridsToFile(subgrid,grid)

% Save subgrid array for later usage in feature extraction
writematrix(subgrid,'SubgridLayout.txt','Delimiter','tab');

% Write grid array to text file using Agilent template format

fID = fopen("ProbeNames_and_Sequences.txt",'r');
probedata = textscan(fID,'%s\t%s\n');
fclose(fID);
probeSeq = probedata{2}(2:end);

ind = 0;
RC = zeros(1068*912,2);
probename = cell(1068*912,1);
probeSeqGrid = cell(1068*912,1);
dim = 'TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT';
lit = 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA';
for r=1:1068
    for c=1:912
        ind = ind + 1;
        RC(ind,:) = [r,c];
        probename{ind} = strcat('probe',sprintf('%04d',grid(r,c)));
        if grid(r,c)>0
            probeSeqGrid{ind} = probeSeq{grid(r,c)};
        end
        if grid(r,c)==-1
            probename{ind} = '_DimEdge_';
            probeSeqGrid{ind} = dim;
        end
        if grid(r,c)==-2
            probename{ind} = '_DimSpot_';
            probeSeqGrid{ind} = dim;
        end
        if grid(r,c)==-3
            probename{ind} = '_LitSpot_';
            probeSeqGrid{ind} = lit;
        end
        if grid(r,c)==-4
            probename{ind} = '_LitEdge_';
            probeSeqGrid{ind} = lit;
        end
        if grid(r,c)==0
            probename{ind} = 'Free_Spot';
            probeSeqGrid{ind} = '';
        end
    end
end

fileID = fopen('STMV_1MGrid.txt','w');
fprintf(fileID,'Column\tRow\tProbeID\tSequence\tFeature Number\tControlType\n');
for i=1:1068*912
    fprintf(fileID,'%d\t%d\t%s\t%s\t%d\n',RC(i,2),RC(i,1),probename{i},probeSeqGrid{i},i);
end
fclose(fileID);

end
