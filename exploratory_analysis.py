"""
    Connor Heaton
    DS 340W Project
    Sp 19


    DESC: Performs operations on participant data depending on which file format it is told to work with, as defined by
    the variable_LOAD TYPE. If LOAD_TYPE is set to 'mat', the script will read participant data from the Matlab (lol)
    files provided by CMU, creating an object for each participant and saving it to a pickle file. Otherwise, the script
    expects to see .pkl files for each participant object, and will find voxels shared among all participants, as
    determined by their (x, y, z) coordinates. Shared voxels are then saved to a csv file.

"""

from study_participant import *
from scipy.io import loadmat
import numpy as np
import pickle


def np_to_lst(np_arr):
	return [list(np_arr[i,:]) for i in range(np_arr.shape[0])]

N_PARTICIPANTS = 9
LOAD_TYPE = 'mat'

mat_file_tmplt = 'fMRI_data/mat/data-science-P{}.mat'
pkl_file_tmplt = 'fMRI_data/pkl/p{}_v2.pkl'

if LOAD_TYPE == 'mat':
	for i in range(1, N_PARTICIPANTS + 1):
		print('Cleaning participant {}...'.format(i))
		p = study_participant(mat_file_tmplt.format(i))
		p.clean_meta()
		p.clean_info()
		p.clean_data()
		print('Saving pickle for participant {}...'.format(i))
		with open(pkl_file_tmplt.format(i), 'wb') as f:
			pickle.dump(p, f, protocol=2)
else:
	common_xyz = None
	for i in range(1, N_PARTICIPANTS + 1):
		print('Reading pkl for p{}...'.format(i))
		p = pickle.load(open(pkl_file_tmplt.format(i), 'rb'))
		#print('p{}:\n\tinfo: {}'.format(i, p.info))
		#print('meta: {}'.format(p.meta))

		if common_xyz is None:
			common_xyz = np_to_lst(p.meta['colToCoord'])
			#input(common_xyz[0])
			common_xyz = [v for v in common_xyz if not p.meta['coordToCol'][v[0], v[1], v[2]] == 0]
		else:
			new_coords = np_to_lst(p.meta['colToCoord'])
			new_coords = [v for v in new_coords if not p.meta['coordToCol'][v[0], v[1], v[2]] == 0]
			common_xyz = [v for v in common_xyz if v in new_coords]
		print('Len common coords after part. {}: {}'.format(i, len(common_xyz)))

		print('Data:')
		for i in range(p.data.shape[0]):
			#print('trial: {}\n\tActivation: {}'.format(i, p.data[i, :]))
			if np.any(p.data[i,:] == 0):
				print('**Some voxels in trial {} have no activation...'.format(i))
		input('peep it')


	print('common cords: {}\nn common coords: {}'.format(common_xyz, len(common_xyz)))

	common_xyz = np.array(common_xyz)
	np.savetxt('common_xyz.csv', common_xyz, delimiter=',')
