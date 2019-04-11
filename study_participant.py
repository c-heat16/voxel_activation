"""
    Connor Heaton
    DS 340W Project
    Sp 19


    DESC: A class file representing one file participant. The object has class variables as well as methods to save
    data read from a .mat file to a dictionary.

"""

from scipy.io import loadmat
from collections import defaultdict
import numpy as np


class study_participant:
	def __init__(self, matfile_dir):
		self.matfile_dir = matfile_dir
		self.raw_data = loadmat(matfile_dir)
		self.data = self.raw_data['data']
		self.meta = self.raw_data['meta']
		self.info = self.raw_data['info']

		#input('meta: {}'.format(self.meta))

		#input('data: {}\ndata shape: {}'.format(self.data, self.data.shape))


		self.meta_attrs = ['study', 'subject', 'ntrials', 'nvoxels', 'dimx', 'dimy', 'dimz', 'colToCoord', 'coordToCol']
		self.info_attrs = ['cond', 'cond_number', 'word', 'word_number', 'epoch']

	def clean_meta(self):
		meta_dict = dict()
		for i, attr in enumerate(self.meta_attrs):
			if attr in ['study', 'subject']:
				attr_val = str(self.meta[0][0][i])
			elif attr not in ['colToCoord', 'coordToCol']:
				attr_val = int(self.meta[0][0][i])
			elif attr == 'colToCoord':
				attr_val = np.vstack(self.meta[0][0][i]).astype(np.int32)
			elif attr == 'coordToCol':
				attr_val = np.array(self.meta[0][0][i]).astype(np.int32)
				#input(self.meta[0][0][i])
			else:
				print('Gonna have to parse this shit... {}'.format(attr))
				# input('Peep that shit: {}'.format(self.meta[0][0][i]))

				attr_val = -1

			meta_dict[attr] = attr_val
		# print(meta_dict)
		self.meta = meta_dict

	def clean_info(self):
		info_dict = dict()
		word_to_trial = defaultdict(list)
		trial_to_word = dict()
		all_words = []

		for trial in range(self.info.shape[1]):
			trial_dict = dict()
			for i, attr in enumerate(self.info_attrs):
				if attr in ['cond', 'word']:
					attr_val = str(self.info[0, trial][i][0])
					if attr == 'word':
						word_to_trial[attr_val].append(trial)
						trial_to_word[trial] = attr_val
						if attr_val not in all_words:
							all_words.append(attr_val)
				else:
					# input('val: {}\nshape: {}'.format(self.info[0, trial][i], self.info[0, trial][i].shape))
					attr_val = int(self.info[0, trial][i][0,0])
					# input('attr val: {}'.format(attr_val))
				trial_dict[attr] = attr_val
			# print('Trial dict {}: {}'.format(trial, trial_dict))
			info_dict[trial] = trial_dict

		# print('n trials in trial dict: {}'.format(len(info_dict)))
		self.info = info_dict
		self.word_to_trial = word_to_trial
		self.trial_to_word = trial_to_word
		self.all_words = sorted(all_words)

	def clean_data(self):
		# data_dict = dict()
		data_arr = None
		for trial in range(self.data.shape[0]):
			# n_voxels = data_arr.shape[1]
			if data_arr is None:
				data_arr = self.data[trial][0][0, :]
				data_arr = data_arr.reshape(1, -1)
			else:
				this_data_arr = self.data[trial][0][0, :].reshape(1, -1)
				data_arr = np.append(data_arr, this_data_arr, axis=0)
		self.data = data_arr

		print('Rearranging data so trials are in alphabetic order of word presented')
		n_trials_per_word = 6
		curr_idx = 0
		alphabetic_trial_order = []

		for word in self.all_words:
			alphabetic_trial_order.extend(self.word_to_trial[word])

			self.word_to_trial[word] = list(range(curr_idx, curr_idx + n_trials_per_word))
			curr_idx += n_trials_per_word

		self.data = self.data[alphabetic_trial_order, :]
		print('Rearrange complete...')
		print('Shape of clean data: {}'.format(self.data.shape))
		"""
		for trial in range(self.data.shape[0]):
			voxel_dict = dict()
			n_voxels = self.data[trial][0].shape[1]
			for voxel in range(n_voxels):
				voxel_value = float(self.data[trial][0][0, voxel])
				voxel_dict[voxel] = voxel_value
			#print('Trial {} n voxels: {}'.format(trial, len(voxel_dict)))
			data_dict[trial] = voxel_dict

		#print('n trials in data dict: {}'.format(len(self.data)))
		self.data = data_dict
		"""

	def remove_unwanted_words(self, unwanted_words):
		desired_words = [w for w in self.all_words if w not in unwanted_words]
		rows_to_keep = []

		for word in desired_words:
			rows_to_keep.extend(self.word_to_trial[word])

		self.data = self.data[rows_to_keep, :]
		#print('Keeping {} rows'.format(len(rows_to_keep)))

