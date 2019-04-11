"""
    Connor Heaton
    DS 340W Project
    Sp 19


    DESC: Utilizes both the common_xyz.csv file of common voxel coords across all participants, as well as the p{}.pkl
    files, to gather activation records of shared voxels across all participants and trials. Saves it to the two files
    participant_trial_voxel.pkl and voxel_trial_participant.pkl.

"""

from collections import defaultdict

import my_utilities as util
import numpy as np
import pickle

common_voxel_file_name = 'common_xyz.csv'
pkl_file_tmplt = 'fMRI_data/pkl/p{}_v2.pkl'
unwanted_words_file = 'corpora/brown_gutenberg_words_without_embedding.txt'
num_participants = 9
N_TRIALS = 360

common_voxel_coords = util.read_file(common_voxel_file_name)
common_voxel_coords = common_voxel_coords.astype(np.int32)
NUM_COMMON_COORDS = common_voxel_coords.shape[0]

print('Num common coords: {}'.format(common_voxel_coords.shape[0]))

common_voxel_coords_list = [list(common_voxel_coords[i,:].reshape(-1)) for i in range(common_voxel_coords.shape[0])]

voxel_coord_to_new_col_dict = dict()

# Read in unwanted words
unwanted_words = None

with open(unwanted_words_file, 'r') as f:
    for line in f:
        if unwanted_words is None:
            unwanted_words = [line.strip()]
        else:
            unwanted_words.append(line.strip())

for i in range(common_voxel_coords.shape[0]):
    these_coords = tuple(common_voxel_coords[i,:].reshape(-1))
    voxel_coord_to_new_col_dict[these_coords] = i

# loop through participants to collect data
voxel_activations = None
all_participant_data = None
for p_id in range(1, num_participants + 1):
    p_obj = pickle.load(open(pkl_file_tmplt.format(p_id), 'rb'))

    # participant object will remove data regarding words which are not in the corpora
    if unwanted_words is not None:
        print('Removing data relating to the {} unwanted words from participant data...'.format(len(unwanted_words)))
        p_obj.remove_unwanted_words(unwanted_words)

    unused_cols = []

    good_cols = []
    voxel_col_to_com_coord = defaultdict(list)
    #voxel_to_xyz = defaultdict(list)

    part_trial_data = [p_obj.data[:, p_obj.meta['coordToCol'][x, y, z] - 1].reshape(-1, 1) for x, y, z in common_voxel_coords]
    part_trial_data = np.hstack(part_trial_data)
    part_trial_data = part_trial_data.reshape(1, -1, NUM_COMMON_COORDS)

    if all_participant_data is None:
        all_participant_data = part_trial_data[:, :, :]
    else:
        all_participant_data = np.append(all_participant_data, part_trial_data, axis=0)

    print('all_participant_data shape: {}'.format(all_participant_data.shape))

print('Saving participant x trial x voxel data to file...')


with open('participant_trial_voxel.pkl', 'wb') as f:
    pickle.dump(all_participant_data, f, protocol=2)


print('Transposing data...')
all_participant_data = np.transpose(all_participant_data)
print('all_participant_data shape: {}'.format(all_participant_data.shape))
print('Saving voxel x trial x participant data to file...')
with open('voxel_trial_participant.pkl', 'wb') as f:
    pickle.dump(all_participant_data, f, protocol=2)
