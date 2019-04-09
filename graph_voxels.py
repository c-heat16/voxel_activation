from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage.interpolation import zoom
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import nibabel
import pickle
import os


def pad_array(x, new_dim, pad_pref = None):
    #print('padding array... x shape: {} new shape: {}'.format(x.shape, new_dim))

    num_dims = len(new_dim)
    if pad_pref is None:
        pad_pref = [0 for _ in range(num_dims)]

    req_padding = [new_dim[i] - x.shape[i] for i in range(num_dims)]

    for axis, n_dim in enumerate(req_padding):
        axis_padding = pad_pref[axis]

        curr_shape = x.shape

        pad_shape = list(curr_shape[:])
        pad_shape[axis] = 1

        pad_val = np.zeros(pad_shape)
        pad_first = False
        axis_pad_pref = False

        if axis_padding != 0:
            axis_pad_pref = True
            if axis_padding < 0:
                pad_first = True
                axis_padding = -1 * axis_padding

        print('Adding zero array of shape {} to axis {} n={} times...'.format(pad_val.shape, axis, n_dim))

        for n in range(n_dim):
            if axis_pad_pref:
                if pad_first:
                    x = np.append(pad_val, x, axis=axis)
                else:
                    x = np.append(x, pad_val, axis=axis)
                axis_padding -= 1
                if axis_padding == 0:
                    axis_pad_pref = False
            else:
                if n % 2 == 0:
                    x = np.append(x, pad_val, axis=axis)
                else:
                    x = np.append(pad_val, x, axis=axis)
        #print('x shape after padding axis {}: {}'.format(axis, x.shape))
    print('x shape after padding: {}'.format(x.shape))
    return x

def trial_to_word_list(info):
    words = [info[i]['word'] for i in range(len(info))]
    #for i in range(len(info)):
    #    wor
    return words

data_path = os.path.join('fMRI_data', 'sM00223')
files = os.listdir(data_path)

brain_img = None
for data_file in files:
    if data_file[-3:] == 'hdr':
        brain_img = nibabel.load(os.path.join(data_path, data_file)).get_data()
        print('raw data shape: {}'.format(brain_img.shape))

# drop last dim
brain_img = np.rot90(brain_img.squeeze(), 1)
print('shape after dropping last dim: {}'.format(brain_img.shape))
#fig, ax = plt.subplots(2, 6, figsize=[18, 3])

num_z_imgs = 5
scale_x_pct = 0.7
scale_y_pct = 0.5

brain_x, brain_y, brain_z = brain_img.shape[0], brain_img.shape[1], brain_img.shape[2]

n_participants = 9
graph_save_file_tmplt = os.path.join('fMRI_data', 'activation_imgs', 'participant{}', 'p{}_t{}_w-{}.pdf')

#part_id = 1

for part_id in range(1, n_participants + 1):
    part_obj_path = os.path.join('fMRI_data', 'pkl', 'p{}.pkl'.format(part_id))
    part_obj = pickle.load(open(part_obj_path, 'rb'))
    part_words = trial_to_word_list(part_obj.info)

    voxel_data = part_obj.data
    print('voxel data shape: {}'.format(voxel_data.shape))

    dimx = part_obj.meta['dimx']
    dimy = part_obj.meta['dimy']
    dimz = part_obj.meta['dimz']
    print('(x, y, z) = ({}, {}, {})'.format(dimx, dimy, dimz))

    trial_activations = [np.zeros((dimx, dimy, dimz)) for _ in range(voxel_data.shape[0])]
    for trial, trial_img in enumerate(trial_activations):
        word = part_words[trial]
        trial_data = voxel_data[trial,:].reshape(-1)
        print('Trial: {} Word: {}'.format(trial, word))
        #min_mag = float('inf')
        #max_mag = float('-inf')

        for voxel, act_mag in enumerate(trial_data):
            #print('voxel {} mag: {}'.format(voxel, act_mag))
            voxel_xyz = part_obj.meta['colToCoord'][voxel]
            vx = voxel_xyz[0]
            vy = voxel_xyz[1]
            vz = voxel_xyz[2]
            trial_img[vx, vy, vz] = act_mag
        min_mag = np.amin(trial_img.reshape(-1))
        max_mag = np.amax(trial_img.reshape(-1))

        trial_img = np.rot90(trial_img, 1)
        scale_x = (brain_x / trial_img.shape[0]) * scale_x_pct
        scale_y = (brain_y / trial_img.shape[1]) * scale_y_pct
        scale_z = (brain_z / trial_img.shape[2])

        trial_img = zoom(trial_img, zoom = [scale_x, scale_y, scale_z], order=1)
        trial_img = pad_array(trial_img, brain_img.shape, pad_pref=[5, 10, 0])
        trial_img_masked = np.ma.masked_where(trial_img == 0, trial_img)
        print('scale shape: {}'.format(trial_img.shape))

        #active_voxels = np.where(trial_img == 0)
        #brain_img[active_voxels] = trial_img[active_voxels]
        """
        for x in range(brain_x):
            for y in range(brain_y):
                for z in range(brain_z):
                    if not trial_img[x, y, z] == 0:#(trial_img[x, y, z] <= 0.0001 and trial_img[x, y, z] >= -0.0001)
                        brain_img[x, y, z] = trial_img[x, y, z]
        """

        n_fig_rows = 3
        fig, ax = plt.subplots(n_fig_rows, num_z_imgs + 1, figsize=[8, 3])
        #fig, ax = plt.subplots(1)
        n = 0
        slice = 0

        z_step = int(np.floor(brain_z/num_z_imgs))
        #z_dim_idxs = np.arange(0, dimz, step=z_step)
        z_dim_idxs = [(i * z_step) + 1 for i in range(num_z_imgs)]

        #print()

        for n, z in enumerate(z_dim_idxs):
            print('n: {} z: {} ax shape: {}'.format(n, z, ax.shape))
            ax[0, n].imshow(brain_img[:, :, z], 'gray')
            ax[1, n].imshow(trial_img_masked[:, :, z], 'jet', alpha=1, vmin=min_mag, vmax=max_mag)

            ax[2, n].imshow(brain_img[:, :, z], 'gray')
            ax[2, n].imshow(trial_img_masked[:, :, z], 'jet', alpha=.5, vmin=min_mag, vmax=max_mag)

            for i in range(n_fig_rows):
                ax[i, n].set_xticks([])
                ax[i, n].set_yticks([])
            #ax[1, n].set_xticks([])
            #ax[2, n].set_xticks([])
            #ax[1, n].set_yticks([])
            #ax[2, n].set_yticks([])

            ax[0, n].set_title('Z coord: {}'.format(z), color='r', y=0.99)
            if n == 0:
                ax[0, n].set_ylabel('Raw Img')
                ax[1, n].set_ylabel('Activation')
                ax[2, n].set_ylabel('Overlay')
            elif n == len(z_dim_idxs) - 1:
                divider = make_axes_locatable(ax[1,n + 1])
                for i in range(n_fig_rows):
                    ax[i, n + 1].set_xticks([])
                    ax[i, n + 1].set_yticks([])
                    ax[i, n + 1].axis('off')
                #cax = ax[1:,-1:].ravel().tolist()
                cax = divider.append_axes("left", size="5%", pad=0.05)

        #n += 1
        #slice += 5

        cmap = plt.get_cmap('jet')
        norm = mpl.colors.Normalize(vmin=min_mag,vmax=max_mag)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        fig.subplots_adjust(wspace=0, hspace=0)
        plt.colorbar(sm, cax=cax) #ax[2, num_z_imgs - 1]
        plt.suptitle('Voxel Activation (Participant: {}, Trial:{}, Word: {})'.format(part_id, trial, word), y = 1.0)
        #plt.show()
        #input('...')
        if not os.path.exists(os.path.dirname(graph_save_file_tmplt).format(part_id)):
            os.makedirs(os.path.dirname(graph_save_file_tmplt).format(part_id))

        print('Saving img for part {} trial {}'.format(part_id, trial))
        plt.savefig(graph_save_file_tmplt.format(part_id, part_id, trial, word))
        plt.clf()
        
        #input('continue?')
        #if not input('continue?') in ['y', 'yes']:
        #    break
