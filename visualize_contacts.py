import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pickle
import mne
import tqdm

from path_utils import get_paths
from my_montage_reader import my_montage_reader

global base_folder
base_folder = 'E:/ds004789-download'
global epoched_folder
epoched_folder = 'E:/epoched'




if __name__ == '__main__':

    # prepare the figure
    fig, ax = plt.subplots(1, 1, figsize=(0.6, 0.6))
    ax = fig.add_subplot(projection='3d')
    mne.utils.set_config('SUBJECTS_DIR', 'E:/freesurfer/7.4.1/subjects')
    brain = mne.viz.Brain('fsaverage', hemi='both', surf='pial', views='lateral', background='white')
    labels = mne.read_labels_from_annot('fsaverage', parc='aparc.a2009s')
    # for label in labels:
    #     brain.add_label(label=label)

    total_contacts, distinct_contacts, responsive_contacts = 0, 0, 0
    total_bi_contacts, distinct_bi_contacts, responsive_bi_contacts = 0, 0, 0
    # loop over p-value files and add contacts
    #subject_list = os.listdir(epoched_folder)
    epoched_subject_list = os.listdir(epoched_folder)
    #print(epoched_subject_list)
    for subject in tqdm.tqdm(epoched_subject_list):
        paths = get_paths(base_folder=base_folder, subject=subject)
        if len(paths) == 0:
            continue
        montage = my_montage_reader(paths[0]['electrodes'])
        #
        num_contacts = len(montage.df)
        contact_name = montage.df['name'].values
        contact_locations = np.concatenate((np.atleast_2d(montage.df['x'].values), np.atleast_2d(montage.df['y'].values), np.atleast_2d(montage.df['z'].values)), axis=0).T
        #
        pattern = os.path.join(epoched_folder, subject, 'ses-*', 'list', 'p_values_WORD')
        paths_pvals = glob.glob(pattern)
        bipolar_response_masks = []
        for i_fname, fname in enumerate(paths_pvals):
            p_vals_data = pickle.load(open(fname, 'rb'))
            bipolar_names, p_values, increase_mask = p_vals_data['ch_names'], p_vals_data['p_values'], p_vals_data['increase_mask']
            bipolar_response_masks.append(increase_mask * (p_values < 0.05))
        bipolar_response_masks = np.array(bipolar_response_masks).T
        bipolar_response_strength = bipolar_response_masks.astype(int).sum(axis=1) / len(paths_pvals)
        # statistics
        total_bi_contacts += bipolar_response_strength.size
        responsive_bi_contacts += (bipolar_response_strength > 0).sum()
        distinct_bi_contacts += (bipolar_response_strength == 1).sum()

        # now map bi-polar to monopolar
        response_strength = np.zeros(num_contacts)
        for bipolar_name, bipolar_resp in zip(bipolar_names, bipolar_response_strength):
            for mono_name in bipolar_name.split('-'):
                idx_contact = np.argwhere([name == mono_name for name in contact_name]).squeeze()
                response_strength[idx_contact] = max(response_strength[idx_contact], bipolar_resp)

        # add electrodes to plot
        response_strength = response_strength * (response_strength >= 0.5)
        strength_groups = np.unique(response_strength)
        for i_strength, strength in enumerate(strength_groups):
            idx = np.argwhere(response_strength == strength).flatten()
            color = np.array((strength, 0.1, 1 - strength))
            brain.add_foci(contact_locations[idx], hemi='vol', color=color, scale_factor=0.2)
            color = 'r' if strength == 1 else 'm'
            if strength == 1:
                ax.scatter(contact_locations[idx, 0], contact_locations[idx, 1], contact_locations[idx, 2], s=8, c=color)
            # statistics
            total_contacts += idx.size
            responsive_contacts += idx.size if (strength > 0) else 0
            distinct_contacts += idx.size if strength == 1 else 0

    print('total_contacts', total_contacts)
    print('responsive_contacts', responsive_contacts)
    print('distinct_contacts', distinct_contacts)
    print('total_bi_contacts', total_bi_contacts)
    print('responsive_bi_contacts', responsive_bi_contacts)
    print('distinct_bi_contacts', distinct_bi_contacts)

    brain.show()
    #ax.legend()
    plt.show(block=True)

