from collections import defaultdict
from heapq import heappush, heappop
from sklearn.cluster import KMeans
import my_utilities as util
from scipy import stats
import numpy as np
import pickle
import os

def filter_voxel_by_mean(x, cutoff=0):
    x_reshaped = x.reshape(x.shape[0], -1)
    voxel_means = [np.mean(x[i,:]) for i in range(x.shape[0])]
    chosen_mean_voxels = [i for i, mean in enumerate(voxel_means) if mean >= cutoff]
    x = x[chosen_mean_voxels, :, :]

    return x

def gen_model_voxel_groups(distances, max_members, n_centers_per_voxel=None):
    if n_centers_per_voxel is None:
        n_centers_per_voxel = distances.shape[1]
    heap = []
    print('Selecting {} closest centers per voxel...'.format(n_centers_per_voxel))
    for v in range(distances.shape[0]):
        voxel_heap = []
        for c in range(distances.shape[1]):
            dist = distances[v, c]
            heappush(voxel_heap, (dist, c))
        for _ in range(n_centers_per_voxel):
            d, c = heappop(voxel_heap)
            heappush(heap, (d, v, c))
        del voxel_heap

    model_voxel_dict = defaultdict(list)
    voxel_assigned = np.zeros(distances.shape[0])

    print('Greedily assigning voxel to closest center...')
    while np.any(voxel_assigned == 0) and heap:
        dist, voxel, center = heappop(heap)
        if voxel_assigned[voxel] == 0 and len(model_voxel_dict[center]) < max_members:
            voxel_assigned[voxel] = 1
            model_voxel_dict[center].append(voxel)

    #for grp, mems in model_voxel_dict.items():
    #    print('Group: {} n mems: {}'.format(grp, len(mems)))

    for i in range(distances.shape[1]):
        print('center: {} mems: {}'.format(i, len(model_voxel_dict[i])))

    print('Max mems: {}'.format(max_members))
    print('Any voxel unassigned: {}'.format(np.any(voxel_assigned == 0)))
    return model_voxel_dict


def gen_model_voxel_data(x, groups):
    model_voxel_data = dict()
    for model_id, model_voxels in groups.items():
        model_voxel_data[model_id] = x[model_voxels, :, :]
        print('Model {} voxel data shape: {}'.format(model_id, model_voxel_data[model_id].shape))
    return model_voxel_data


vtp = pickle.load(open('voxel_trial_participant.csv', 'rb'))
N_CLUSTERS = 100
K_MEANS_FILE = 'kmeans_model_{}k.pkl'.format(N_CLUSTERS)

vtp = filter_voxel_by_mean(vtp, 0)
print('filter shape: {}'.format(vtp.shape))

vtp_reshaped = vtp.reshape(vtp.shape[0], -1)

if not os.path.exists(K_MEANS_FILE):
    print('Fitting kmeans...')
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=16, verbose=1, n_init=5, n_jobs=-1).fit(vtp_reshaped)
    with open(K_MEANS_FILE, 'wb') as f:
        pickle.dump(kmeans, f, protocol=2)
else:
    print('Loading saved model...')
    kmeans = pickle.load(open(K_MEANS_FILE, 'rb'))

vtp_distances = kmeans.transform(vtp_reshaped)

print('distances shape: {}'.format(vtp_distances.shape))

max_center_mems = int(np.ceil(vtp.shape[0] / N_CLUSTERS))
model_voxel_groups = gen_model_voxel_groups(vtp_distances, max_center_mems)

model_voxel_data = gen_model_voxel_data(vtp, model_voxel_groups)
