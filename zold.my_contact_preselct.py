import os.path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
from nilearn import datasets, plotting
#from nilearn.image import load_img
import nibabel as nib


def view_volume(volume, labels):

    figures = []
    #fig, ax = plt.subplots(4, 4, figsize=(10, 10))
    intvl = int(volume.shape[2] / 32)
    slices = np.arange(start=int(intvl / 2), step=intvl, stop=volume.shape[2])
    for img_id, slice_id in enumerate(slices[:32]):
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax = np.array(ax).reshape(1, 1)
        slice = volume[:, :, slice_id].squeeze()
        slice *= slice < len(labels)
        #ax.flatten()[img_id].imshow(slice.T)
        ax.flatten()[0].imshow(slice.T)
        for label_id, label in enumerate(labels):
            mask = label_id == slice
            if mask.sum() > 30:
                xy = np.round(np.argwhere(mask).squeeze(), decimals=0).astype(int)
                x, y = xy[int(xy.shape[0] / 2)]
                #ax.flatten()[img_id].text(x, y, label, color=(0.5, 0.5, 0.5))
                ax.flatten()[0].text(x-2, y, '* ', color=(1, 1, 1), fontsize=10, fontweight='bold')
                ax.flatten()[0].text(x, y, label[:25].replace(' ', '\n'), color=(1, 1, 1), fontsize=10, fontweight='bold')
                ax.flatten()[0].axis(False)
                #ax.flatten()[0].set_xlim((20, 160))
                #ax.flatten()[0].set_ylim((20, -20))
        figures.append(fig)
    #plt.show()
    return figures


def plot_contact(volume, affine, labels, x, y, z, name):

    fig, ax = plt.subplots(1, 1, num=name, figsize=(8, 8))
    coords = np.round(np.linalg.inv(affine) @ np.array((x, y, z, 1)).reshape(4, 1), decimals=0).astype(int)[:3]
    slice_id = coords[-1]
    slice = volume[:, :, slice_id].squeeze()
    slice *= slice < len(labels)
    ax.imshow(slice.T, cmap='gray')

    for i, label in enumerate(labels):
        mask = i == slice
        if mask.sum() > 0:
            reg_xy = np.argwhere(mask)
            reg_xy = reg_xy[int(reg_xy.shape[0] / 2)]
            xc, yc = reg_xy
            if mask[coords[0], coords[1]]:
                exc, eyc, elabel = xc, yc, label
                #ax.text(xc, yc, label[:25], c=(0.8, 0.8, 0.8), fontsize=10, fontweight='bold')
            else:
                ax.text(xc, yc, label[:25], c=(0.2, 0.2, 0.2))
    ax.text(exc, eyc, elabel[:25], c=(1, 1, 0), fontsize=10, fontweight='bold')
    ax.add_patch(patches.Circle((coords[0], coords[1]), 1, color=(1, 0.4, 0.4)))
    ax.text(coords[0] + 2, coords[1] + 2, name, c=(1, 0.4, 0.4), fontsize=12, fontweight='bold')
    ax.set_xlabel(name + ' in ' + elabel)

    #plt.show()

    return fig


def load_harvard_oxford_atlas():

    atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-1mm')
    volume = atlas.maps.dataobj
    affine = atlas.maps.affine
    labels = atlas.labels
    ROI_list = ['Occipital Pole', 'Cuneal Cortex', 'Lateral Occipital Cortex, superior division',
                'Lateral Occipital Cortex, inferior division',
                #'Temporal Fusiform Cortex, anterior division', 'Temporal Fusiform Cortex, posterior division',
                'Temporal Occipital Fusiform Cortex', 'Occipital Fusiform Gyrus', 'Lingual Gyrus',
                'Superior Parietal Lobule', 'Precuneous Cortex', 'Supracalcarine Cortex', 'Precuneous Cortex']

    return volume, affine, labels, ROI_list

def load_desikanKillianyMNI():

    folder = 'C:/Users/menas/OneDrive/Desktop/openneuro/desikanKillianyMNI'
    atlas_map = nib.load(os.path.join(folder, 'desikanKillianyMNI.nii'))
    labels = np.loadtxt(os.path.join(folder, 'desikanKillianyNodeNames.txt'), dtype=np.str_)
    ROI_list = ['ctx-rh-pericalcarine', 'ctx-rh-lingual', 'ctx-rh-cuneus', 'ctx-rh-lateraloccipital', 'ctx-rh-parahippocampal', 'ctx-rh-fusiform', 'ctx-rh-inferiortemporal',
                'ctx-lh-pericalcarine', 'ctx-lh-lingual', 'ctx-lh-cuneus', 'ctx-lh-lateraloccipital', 'ctx-lh-parahippocampal', 'ctx-lh-fusiform', 'ctx-lh-inferiortemporal']

    return np.array(atlas_map.dataobj).astype(int), atlas_map.affine, list(labels), ROI_list


# Load electrode file
subject = 'R1006P'
fname = 'E:/ds004789-download/sub-{}/ses-0/ieeg/sub-{}_ses-0_task-FR1_space-MNI152NLin6ASym_electrodes.tsv'.format(subject, subject)
fname = 'E:/ds004789-download/sub-{}/ses-0/ieeg/sub-{}_ses-0_task-FR1_space-MNI152NLin6ASym_electrodes.tsv'.format(subject, subject)
df = pd.read_csv(fname, delimiter='\t')


atlas_sel = ['harvard_oxford', 'desikan_Killiany'][0]
if atlas_sel == 'harvard_oxford':
    volume, affine, labels, ROI_list = load_harvard_oxford_atlas()
if atlas_sel == 'desikan_Killiany':
    volume, affine, labels, ROI_list = load_desikanKillianyMNI()

img_folder = 'C:/Users/menas/OneDrive/Desktop/openneuro/figures'
generate_atlas_slices = False
if generate_atlas_slices:
    figs = view_volume(volume, labels)
    for i, fig in enumerate(figs):
        fig.suptitle('{} \n({})'.format(subject,atlas_sel ))
        fig.savefig(os.path.join(img_folder, subject + ' ' + str(i)))
    plt.clf()

# statistics for the atlas
box_volume = np.prod(volume.shape)
brain_volume = (volume > 0).sum()
print('\n\nbox volume = {}   brain volume = {}\n'.format(box_volume, brain_volume))
for i_label, label in enumerate(labels):
    if i_label > 0:
        region_volume = (volume == i_label).sum()
        print('{}:\t volume={}\t({:5.1f} %)'.format(label, region_volume, 100 * region_volume / brain_volume))
print('\n\n')


ROI_indexes = [np.argwhere([roi == label for label in labels]).squeeze() for roi in ROI_list]

for region_idx, region in zip(ROI_indexes, ROI_list):
    #print(region_idx, region)
    for i in range(df.shape[0]):
        line = df.iloc[i]
        x, y, z = line['x'], line['y'], line['z']
        name = line['name']
        r = np.array((x, y, z, 1)).reshape(4, 1)
        r = np.linalg.inv(affine) @ r
        coords = np.round(r[:3], decimals=0).astype(int)
        try:
            if region_idx == volume[coords[0], coords[1], coords[2]]:
                print('{}\t {:5.1f}  {:5.1f}  {:5.1f}\t{}   ({})'.format(name, x, y, z, region, line['ind.region']))
                fig = plot_contact(volume, affine, labels, x, y, z, name)
                plt.savefig(os.path.join(img_folder, subject + ' ' + name))
                plt.show()
        except:
            pass


# ####################
#
# import pandas as pd
#
# # Create DataFrame
# grouped_data = pd.DataFrame(region_assignments, columns=["Coordinate", "Region"])
#
# # Save to CSV
# grouped_data.to_csv("C:/Users/menas/OneDrive/Desktop/openneuro/grouped_contacts.csv", index=False)
#
# # Optional: Plot regions on the brain surface
# plotting.plot_markers(
#     coords, title="Mapped Contacts on Brain Surface"
# )
