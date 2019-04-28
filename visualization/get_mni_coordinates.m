v = csvread('voxel.csv', ',', r, c)
v = dlmread("v.csv", ',', r, c)

function p = get_mni_coordinates(v)

% For Neurosematic project with SPM2-processed data (60WP and later),
% with 3x3x6 voxels in MNI space
%
% Get MNI coordinates for an array of N voxels
% v - Nx3 array of VOXEL COORDINATES (voxel space)
% p - Nx3 array of MNI points 
%
% For example,
% for voxel v = [45 30 12] 
% the MNI points p = [-59.3750 -21.8750 18.0000]
  
% transformation matrix from vox coord to MNI coord

transformToXYZmm = [-3.125 0     0   81.250
                     0     3.125 0 -115.625
                     0     0     6  -54.000
                     0     0     0    1.000];

% points in MNI coordinates, in mm

p = [v ones(size(v,1), 1)] * transformToXYZmm(1:3,:)';
p

