"""
    Connor Heaton
    DS 340W Project
    Sp 19


    DESC: Reads in voxel activations records across all participants and trials, and then finds the n voxels
    with the highest correlation.

"""

from scipy.stats import pearsonr
import numpy as np
import pickle
import heapq
import time
import os

import Pearson_Correlation as andrea_pearson_corr

def calc_pearson_corr_mat(ptv):
    tv_all_p = np.vstack(ptv)
    print('New shape: {}'.format(tv_all_p.shape))

    corr = np.corrcoef(tv_all_p, rowvar=False)
    #corr = corr[:n_voxels, :n_voxels] # take quadrant 2 (upper left)
    print('Corr shape: {}'.format(corr.shape))

    return corr

def get_n_highest_corr_voxels(corr_mat, n):

    n_voxels = corr_mat.shape[0]
    corr_data = 1 - np.absolute(corr_mat)


    if not os.path.exists('highest_corr_voxels_coords.csv'):
        print('Doing weird np shit...')
        highest_corr_voxels_coords = np.dstack(np.unravel_index(np.argsort(corr_data.ravel()), (n_voxels, n_voxels)))
        highest_corr_voxels_coords = np.squeeze(highest_corr_voxels_coords).astype(np.int32)
        np.savetxt('highest_corr_voxels_coords.csv', highest_corr_voxels_coords, delimiter=',', fmt='%d')
    else:
        print('Reading from file instead of doing weird np shit...')
        highest_corr_voxels_coords = np.loadtxt('highest_corr_voxels_coords.csv', dtype=np.int32, delimiter=',')

    print('highest_corr_voxels_coords shape: {}'.format(highest_corr_voxels_coords.shape))

    highest_corr_voxels = []

    r = 0

    while len(highest_corr_voxels) < n:
        v1 = highest_corr_voxels_coords[r, 0]

        v2 = highest_corr_voxels_coords[r, 1]

        if not v1 == v2:
            print('corr of v{} and v{}: {}'.format(v1, v2, corr_mat[v1, v2]))
            if v1 not in highest_corr_voxels:
                highest_corr_voxels.append(v1)
            if v2 not in highest_corr_voxels:
                highest_corr_voxels.append(v2)

        r += 1

    if len(highest_corr_voxels) == n + 1:
        highest_corr_voxels = highest_corr_voxels[:-1]

    print('len of highest corr voxels: {}'.format(len(highest_corr_voxels)))

    return highest_corr_voxels


ptv_file = 'participant_trial_voxel.pkl'

ptv = pickle.load(open('participant_trial_voxel.pkl', 'rb'))

ptv = ptv.astype(np.float16)

#input(ptv.dtype)

corr_mat = calc_pearson_corr_mat(ptv)

highest_corr_voxels = get_n_highest_corr_voxels(corr_mat, n=2000)
curr_time = time.strftime('%Y%m%d-%H%M%S')
with open('highest_corr_voxels_{}.txt'.format(curr_time), 'w+') as f:
    for voxel in highest_corr_voxels:
        f.write('{}\n'.format(voxel))