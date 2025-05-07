% This script implements our custom algorithm for extracting integrated
% fluorescence intensities from a raw microarray image.
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

%% Read raw image file
filename = 'SG15294487_258644410004_S001.tif';
microarrayImage = imread(filename,2);
clear filename

%% Import subgrid layout
SubgridLayout = importdata('SubgridLayout.txt');
SubgridLayout(1,1) = -10;
Subgrid4x4 = [[SubgridLayout,SubgridLayout];[SubgridLayout,SubgridLayout]];

%% Get subgrid boundaries (rows x cols) from file
SubgridBounds = getSubgridBounds('SubgridBounds.txt');

%% Get coordinates (x,y) of subgrid first feature within the subgrid image
Feature1xy = importdata('FirstFeatureCoordinates.txt');

%% Load keys
[key12,key24] = getProbe2nt_keys;

%% Feature extraction
clc
for subgridNumber=1:48
    % Get subgrid image
    Subgrid_Rows = SubgridBounds{subgridNumber}(1,1):SubgridBounds{subgridNumber}(2,1);
    Subgrid_Cols = SubgridBounds{subgridNumber}(1,2):SubgridBounds{subgridNumber}(2,2);
    SubgridImage = microarrayImage(Subgrid_Rows,Subgrid_Cols);
    SubgridImage = SubgridImage - min(min(SubgridImage));

    % do feature extraction
    [r12,r24] = featureExtract(SubgridImage,subgridNumber,Feature1xy(subgridNumber,:),Subgrid4x4,key12,key24);

    % store replicates
    reps12{subgridNumber} = r12;
    reps24{subgridNumber} = r24;
end

%% BEGIN FUNCTION DEFINITIONS
%% Feature extraction function
% This function extracts the integrated fluorescence of the spots
% (features) in each subgrid image. The intensity values of replicate 
% probes are grouped together and sorted according to where the probe binds
% along the RNA primary sequence.

function [r12,r24] = featureExtract(SubgridImage,subgridNumber,Feature1xy,Subgrid4x4,key12,key24)

% Find the spot centers, number of rows and cols, and approximate spacing
% between them
[spotcenters,numRows,numCols,periodx,periody] = getSpotCenters(SubgridImage,subgridNumber);

% Find index of the first feature of the subgrid (upper left corner)
spot1index = findFeature1Index(spotcenters,Feature1xy);

% Create an array that matches probe number to feature locationon the image
[spot1c,spot1r] = ind2sub([numCols,numRows],spot1index);
SubgridShift = circshift(Subgrid4x4,[spot1r-1,spot1c-1]);
SubgridShift = SubgridShift(1:numRows,1:numCols);

% integrate spot intensities
intensities = cell(1,2090);
for i=1:length(intensities)

    % group replicates
    inds = find(SubgridShift'==i);
    ints = zeros(size(inds));

    for j=1:length(inds)
        imSpotCenter = spotcenters(inds(j),:);

        % spot image boundaries are asymmetric because of the hexagonal lattice
        rowbnds = ceil(imSpotCenter(2)-periody)+1:floor(imSpotCenter(2)+periody)-1;
        colbnds = ceil(imSpotCenter(1)-periodx/2):floor(imSpotCenter(1)+periodx/2);
        imT = SubgridImage(rowbnds,colbnds);

        ints(j) = mean(mean(imT));
    end
    intensities{i} = ints;
end


r12 = cell(1,length(key12));
r24 = cell(1,length(key24));
for i=1:length(key12)
    r12{i} = intensities{key12(i)};
end
for i=1:length(key24)
    r24{i} = intensities{key24(i)};
end
end

%% Function to get spot centers from subgrid image
function [spotcenters,numRows,numCols,periodx,periody] = getSpotCenters(SubgridImage, subgrid)

% ad hoc parameters needed to align grid to spot centers
dp = [[1,2,2,1];[1,1,2,2]];
dpi1 = [2,1,1,1,2,2,2,1,1,2,2,2,  1,2,2,1,1,1,2,2,2,1,1,1,  2,1,1,1,2,2,2,1,1,2,2,2,  2,2,2,1,1,1,2,2,1,1,1,2];
dpi2 = [1,2,1,2,2,2,1,2,2,1,2,1,  1,1,1,1,1,2,1,2,2,1,2,1,  1,1,1,1,1,2,2,2,2,1,2,1,  2,1,1,1,1,2,2,1,2,2,2,1];

logSubgridImage = log(double(SubgridImage)+1);
proceed = 0;
upper = 0;
while ~proceed
    % find spot x and y centers
    [xc,yc,periodx,periody] = getGridLines(logSubgridImage);
    xc = xc(dpi2(subgrid):end-upper);
    yc = yc(3:end-1);

    % generate x and y coordinates of spot centers on hexagonal lattice
    spotcenters = zeros(length(xc)*length(yc),2);
    featureNumber=0;
    for i=dp(dpi1(subgrid),1):2:length(yc)
        for j=dp(dpi1(subgrid),2):2:length(xc)
            featureNumber=featureNumber+1;
            spotcenters(featureNumber,:) = [xc(j),yc(i)];
        end
    end
    for i=dp(dpi1(subgrid),3):2:length(yc)
        for j=dp(dpi1(subgrid),4):2:length(xc)
            featureNumber=featureNumber+1;
            spotcenters(featureNumber,:) = [xc(j),yc(i)];
        end
    end
    spotcenters(spotcenters(:,1)==0,:) = [];
    [~,sortind] = sort(spotcenters(:,2));
    spotcenters = spotcenters(sortind,:);
    numRows = length(yc);
    numCols = round(length(spotcenters)/numRows);
    if numRows*numCols~=featureNumber
        upper = upper + 1;
    else
        proceed = 1;
    end
end
end

%% Function to load subgrid image boundaries
% The subgrid_boundaries_file contains the lower row boundary, upper row
% boundary, lower column boundary, and upper column boundary for all 48
% subgrids.

function SubgridBounds = getSubgridBounds(subgrid_boundaries_file)
SubgridBounds = cell(48,1);

rc = importdata(subgrid_boundaries_file);
for subgrid=1:48
    SubgridBounds{subgrid} = [rc(subgrid,1:2)',rc(subgrid,3:4)'];
end

end

%% Function to get index of first feature in the subgrid
% Finds which spot is the closest to the spot1 (x,y) coordinate.
function FirstFeatureIndex = findFeature1Index(spotcenters, firstFeatureCoordinate)

distances = sum((spotcenters - firstFeatureCoordinate).^2,2);
FirstFeatureIndex = find(distances==min(distances));
FirstFeatureIndex = FirstFeatureIndex(1);

end

%% Function to convert probe number to RNA nucleotide and probe length
function [key12,key24] = getProbe2nt_keys
k12 = importdata('key_12mers.txt');
k24 = importdata('key_24mers.txt');

k12 = k12.textdata;
k24 = k24.textdata;

pnum12 = zeros(length(k12)-1,1);
pnum24 = zeros(length(k24)-1,1);

for i=2:length(k12)
    pnum12(i-1) = str2num(k12{i,1}(6:9));
end
for i=2:length(k24)
    pnum24(i-1) = str2num(k24{i,1}(6:9));
end

key12 = pnum12;
key24 = pnum24;

end


%% Function to get grid lines
function [xCenters,yCenters,estPeriodx,estPeriody] = getGridLines(microarrayImage)
% Based on Robert Bemis (2023). DNA MicroArray Image Processing Case Study
% (https://www.mathworks.com/matlabcentral/fileexchange/2573-dna-microarray-image-processing-case-study)
% MATLAB Central File Exchange

% Estimate spot x-spacing by autocorrelation
xProfile = mean(microarrayImage,1);
ac = xcov(xProfile);
s1 = diff(ac([1 1:end]));
s2 = diff(ac([1:end end]));
maxima = find(s1>0 & s2<0);
estPeriodx = mean(diff(maxima));

% Remove background morphologically
seLine = strel('line',estPeriodx,0);
xProfile2 = imtophat(xProfile,seLine);

% Segment peaks
level = graythresh(xProfile2/255)*255;
bw = im2bw(xProfile2/255,level/255);
L = bwlabel(bw);

% Locate centers
stats = regionprops(L);
centroids = [stats.Centroid];
xCenters = centroids(1:2:end);
xCenters = xCenters(2:end-1);

% Segment offset rows into separate images to get y-centers
im1 = [];
im2 = [];
for i=1:2:length(xCenters)
    im1 = [im1,microarrayImage(:,ceil(xCenters(i)-estPeriodx/2):floor(xCenters(i)+estPeriodx/2))];
end
for i=2:2:length(xCenters)
    im2 = [im2,microarrayImage(:,ceil(xCenters(i)-estPeriodx/2):floor(xCenters(i)+estPeriodx/2))];
end

yProfile1=sum(im1,2)';
yProfile2=sum(im2,2)';
ac = xcov(yProfile1);
s1 = diff(ac([1 1:end]));
s2 = diff(ac([1:end end]));
maxima = find(s1>0 & s2<0);
estPeriodr1 = mean(diff(maxima));
seLine = strel('line',estPeriodr1,0);
yProfile3 = imtophat(yProfile1,seLine);
level = graythresh(yProfile3/255)*255;
bw = im2bw(yProfile3/255,level/255);
L = bwlabel(bw);
stats = regionprops(L);
centroids = [stats.Centroid];
yCenters1 = centroids(1:2:end);

ac = xcov(yProfile2);
s1 = diff(ac([1 1:end]));
s2 = diff(ac([1:end end]));
maxima = find(s1>0 & s2<0);
estPeriodr2 = mean(diff(maxima));
seLine = strel('line',estPeriodr2,0);
yProfile4 = imtophat(yProfile2,seLine);
level = graythresh(yProfile4/255)*255;
bw = im2bw(yProfile4/255,level/255);
L = bwlabel(bw);
stats = regionprops(L);
centroids = [stats.Centroid];
yCenters2 = centroids(1:2:end);

yCenters = sort([yCenters1,yCenters2]);

estPeriody = mean([estPeriodr1,estPeriodr2])/2;

xCenters = xCenters(2:end-2);
yCenters = yCenters(2:end-2);

end
