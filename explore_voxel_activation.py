import matplotlib.pyplot as plt
import my_utilities as util
from scipy import stats
import numpy as np
import pickle

ptv = pickle.load(open('participant_trial_voxel.pkl', 'rb'))

#for v in range(ptv.shape[-1]):
#    print('voxel: {} stats: {}'.format(v, stats.describe(ptv[:,:,v].reshape(-1))))
print('Overall: {}'.format(stats.describe(ptv.reshape(-1))))
print('calculating variance...')
#voxel_variance = np.var(ptv, axis=2)
#print('Voxel var shape: {}'.format(voxel_variance.shape))

voxel_variance = [np.var(ptv[:, :, v].reshape(-1)) for v in range(ptv.shape[-1])]
#max_var = np.amax(voxel_variance)
#voxel_variance = [v / max_var for v in voxel_variance]

input('Mean voxel variance: {}'.format(np.mean(voxel_variance)))

print('Len vox var: {}'.format(len(voxel_variance)))
hist_bins = list(np.arange(0,5,.1))
print('bins: {}'.format(hist_bins))
print('plotting...')
plt.hist(voxel_variance, bins=hist_bins)
plt.title('Variance of Voxels Shared by Participants')
plt.ylabel('# Occurrences')
plt.xlabel('Variance')


print('Saving fig...')
plt.savefig('var_across_voxels.pdf', dpi=25)
#plt.show()
