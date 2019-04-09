from collections import defaultdict
from heapq import heappush, heappop
from sklearn.cluster import KMeans
import my_utilities as util
from scipy import stats
import numpy as np
import pickle
import os

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


N_CLUSTERS = 100
K_MEANS_FILE = 'kmeans_model_{}k.pkl'.format(N_CLUSTERS)

vtp = pickle.load(open('voxel_trial_participant.csv', 'rb'))


n_voxels = vtp.shape[0]
print('N voxels: {}'.format(n_voxels))

vtp_reshaped = vtp.reshape(n_voxels, -1)
voxel_means = [np.mean(vtp_reshaped[i,:]) for i in range(n_voxels)]
voxel_neg = [np.any(vtp_reshaped[i,:] < -0.5) for i in range(n_voxels)]
chosen_mean_voxels = [i for i, mean in enumerate(voxel_means) if mean >= 0]
chosen_neg_voxels = [i for i, neg in enumerate(voxel_neg) if neg == False]
print('len nonneg: {}'.format(len(chosen_neg_voxels)))
#chosen_voxels = [v for v in chosen_mean_voxels if v in chosen_neg_voxels]
chosen_voxels = chosen_mean_voxels
#print(voxel_means)
print('n voxels w/ mean >= 0: {}'.format(len(chosen_voxels)))

vtp = vtp[chosen_voxels, :]
vtp_reshaped = vtp_reshaped[chosen_voxels,:]


if not os.path.exists(K_MEANS_FILE):
    print('Fitting kmeans...')
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=16, verbose=1, n_init=5, n_jobs=-1).fit(vtp_reshaped)
    with open(K_MEANS_FILE, 'wb') as f:
        pickle.dump(kmeans, f, protocol=2)
else:
    print('Loading saved model...')
    kmeans = pickle.load(open(K_MEANS_FILE, 'rb'))

print('Predicting centers...')
voxel_cluster = kmeans.predict(vtp_reshaped)
centers, members = np.unique(voxel_cluster, return_counts=True)

for c, m in zip(centers, members):
    print('Center {} has {} members...'.format(c, m))

print('Avg number of members: {}'.format(np.mean(members)))

vtp_distances = kmeans.transform(vtp_reshaped)
print('distances shape: {}'.format(vtp_distances.shape))

max_center_mems = int(np.ceil(vtp.shape[0] / N_CLUSTERS))
model_voxel_groups = gen_model_voxel_groups(vtp_distances, max_center_mems)
