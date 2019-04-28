function L = get_aal_labels(p)

% For Neurosematic project with SPM2-processed data (60WP and later),
% with 3x3x6 voxels in MNI space
%
% Get AAL labels for an array of N MNI points (mm)
% p - Nx3 array of MNI points 
% L - 1xN cell array of AAL labels
%
% For example,
% for MNI points p = [-32 -42 -18; 16 -54 14; -40 -100 30],
% the AAL labels L = {'Fusiform_L', 'Precuneus_R', 'Not_labelled'}

% atlas location *******modify during the installation*******

atlas = 'ccbi_aal_atlas.mat';
load(atlas);

% get labels

vinv = [];
label = {};

for j = 1:size(p,1),
	pv(j,:) = round([p(j,:) 1] * vinv');
	z(j) = Y(pv(j,1), pv(j,2), pv(j,3));
	L{j} = label{find(id==z(j))};
end
