from collections import defaultdict

import my_utilities as util
import numpy as np
import pickle

common_voxel_file_name = 'common_xyz.csv'
pkl_file_tmplt = 'fMRI_data/pkl/p{}.pkl'
num_participants = 9
N_TRIALS = 360

common_voxel_coords = util.read_file(common_voxel_file_name)
common_voxel_coords = common_voxel_coords.astype(np.int32)
NUM_COMMON_COORDS = common_voxel_coords.shape[0]
print('Num common coords: {}'.format(common_voxel_coords.shape[0]))

common_voxel_coords_list = [list(common_voxel_coords[i,:].reshape(-1)) for i in range(common_voxel_coords.shape[0])]

voxel_coord_to_new_col_dict = dict()

for i in range(common_voxel_coords.shape[0]):
    these_coords = tuple(common_voxel_coords[i,:].reshape(-1))
    voxel_coord_to_new_col_dict[these_coords] = i

# loop through participants to collect data
voxel_activations = None
all_participant_data = None
for p_id in range(1, num_participants + 1):
    p_obj = pickle.load(open(pkl_file_tmplt.format(p_id), 'rb'))
    #print('CoordToCol[0,0,:]: {}'.format(p_obj.meta['coordToCol'][0,0,:]))
    #input('CoordToCol shape: {}'.format(p_obj.meta['coordToCol'].shape))

    #input(np.mean(p_obj.meta['coordToCol']))
    unused_cols = []

    #for k, v in p_obj.meta.items():
    #    print('key: {} value: {}'.format(k, v))

    #input('peep that shit')

    good_cols = []
    voxel_col_to_com_coord = defaultdict(list)
    #voxel_to_xyz = defaultdict(list)

    part_trial_data = [p_obj.data[:, p_obj.meta['coordToCol'][x, y, z] - 1].reshape(-1, 1) for x, y, z in common_voxel_coords]
    part_trial_data = np.hstack(part_trial_data)
    part_trial_data = part_trial_data.reshape(1, N_TRIALS, NUM_COMMON_COORDS)

    if all_participant_data is None:
        all_participant_data = part_trial_data[:, :, :]
    else:
        all_participant_data = np.append(all_participant_data, part_trial_data, axis=0)

    print('all_participant_data shape: {}'.format(all_participant_data.shape))

    """
    for x, y, z in common_voxel_coords_list:
        new_col = p_obj.meta['coordToCol'][x, y, z] - 1

        voxel_data = p_obj.data[:, new_col].reshape(-1,1)





        if new_col > -1:
            voxel_col_to_com_coord[new_col].append((x, y, z))
            if new_col not in good_cols:
                #print('Adding col {} to good col list (x={}, y={}, z={})...'.format(new_col, x, y, z))
                good_cols.append(new_col)

    print('Part. {} has {} voxels, {} common voxels...'.format(p_id, p_obj.meta['nvoxels'], len(voxel_col_to_com_coord)))
    for k, v in voxel_col_to_com_coord.items():
        #print('voxel: {} (x, y, z): {}'.format(k, v))
        if len(v) > 1:
            print('\t***voxel {} has {} (x, y, z) values'.format(k, len(v)))

    #input('processing of participant {} complete...'.format(p_id))
    """

print('Saving participant x trial x voxel data to file...')
with open('participant_trial_voxel.pkl', 'wb') as f:
    pickle.dump(all_participant_data, f, protocol=2)


print('Transposing data...')
all_participant_data = np.transpose(all_participant_data)
print('all_participant_data shape: {}'.format(all_participant_data.shape))
print('Saving voxel x trial x participant data to file...')
with open('voxel_trial_participant.pkl', 'wb') as f:
    pickle.dump(all_participant_data, f, protocol=2)
