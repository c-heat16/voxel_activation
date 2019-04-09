from study_participant import *
from scipy.io import loadmat
import numpy as np
import pickle

"""
raw_data = loadmat('fMRI_data/data-science-P1.mat')

data = raw_data['data']

info = raw_data['info']
meta = raw_data['meta']

print('meta: {} Shape: {}'.format(type(meta), meta.shape))
print('meta[0]: {}'.format(meta[0]))
print('meta[0][0][0]: {}'.format(meta[0][0][0]))
print('meta[0][0][1]: {}'.format(meta[0][0][1]))
#print('meta[0][0]: {}'.format(meta[0][0][8]))


print('Info: {} Shape: {}'.format(type(info), info.shape))
print('Info[0]: {} Shape: {}'.format(type(info[0]), info[0].shape))
print('Info[0, 0]: {}'.format(info[0,0]))
print('Type Info[0, 0]: {}'.format(type(info[0, 0])))
print('len info[0, 0]: {}'.format(len(info[0, 0])))

for c in range(info.shape[1]):
	print('c: {}'.format(c))
	for i in range(len(info[0,0])):
		print('\tinfo[0, {}][{}]: {}'.format(c, i, info[0, c][i]))


print('Data type: {} Shape: {}'.format(type(data), data.shape))

print('Type Data[0]: {} Shape data [0]: {}'.format(type(data[0]), data[0].shape))
print('Data[0][0]: {}'.format(data[0][0]))
print('Shape data[0][0]: {}'.format(data[0][0].shape))
print('Shape data[0][0][0,0]: {}'.format(data[0][0][0,1]))

#input('lol')
"""


def np_to_lst(np_arr):
	out_set = set()
	"""
	for i in range(np_arr.shape[0]):
		out_set.add(np_arr[i,:].reshape(-1))
	"""
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
