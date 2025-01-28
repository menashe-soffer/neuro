import os

import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from path_utils import get_subject_list, get_paths
from event_reader import event_reader
from my_montage_reader import my_montage_reader
from my_mne_wrapper import my_mne_wrapper
from my_subband_processings import calc_HFB
from my_tfr_wrapper_1 import calc_tfr

def make_data_availability_list(base_folder, region_list, hemisphere_sel):

    subject_list = get_subject_list(base_folder=base_folder)
    data = dict()

    for subject in subject_list:

        paths = get_paths(base_folder=base_folder, subject=subject, sess_slct=None, mode='bipolar')
        # PATCH
        if len(paths) == 0:
            continue

        try:
            #
            subject_data = dict()

            # add contacts info
            montage = my_montage_reader(fname=paths[0]['electrodes'])
            electrode_list = montage.get_electrode_list_by_region(region_list=region_list, hemisphere_sel=hemisphere_sel)
            #subject_data['contact count'] = dict()
            total_contacts = 0
            for region in electrode_list:
                n = len(electrode_list[region])
                #subject_data['contact count'][region] = n
                subject_data[region] = n
                total_contacts += n

            # add session info
            subject_data['numsessions'] = len(paths)
            subject_data['num countdowns'] = []
            for i_sess, path in enumerate(paths):
                event_reader_obj = event_reader(path['events'])
                subject_data['num countdowns'].append(len(event_reader_obj.get_countdowns()))

            if total_contacts > 0:
                data[subject] = subject_data
                print(subject)
            else:
                print('no contacts in ROI for', subject)

        except:
            print('failed for', subject)

    # generate excel file
    df = pd.DataFrame.from_dict(data).T
    # "flatening" the contact counts - failed to do it so I made it flat in the first place
    writer = pd.ExcelWriter('E:/ds004789-download/data_availability_for_protocol_1.xlsx', engine='xlsxwriter')
    df.to_excel(writer)
    writer.close()


def calc_contdwn_responces(subject, regions=None, sides=None, band=[40, 100], timebin_sec=0.5, avg=6, in_session_gap=13, mode='monopolar', scan_files_only=True):

    # configurations for the sub=band processor
    # generate the sub-band spec for the processing
    subs_centers = np.linspace(start=band[0], stop=band[-1], num=int((band[-1] - band[0]) / 10 + 1.5))
    subs_centers = (subs_centers[:-1] + subs_centers[1:]) / 2
    subs_bw = np.diff(subs_centers)[0]
    # result holders
    p_list, sem_list, desc_list = [], [], []

    paths = get_paths(base_folder=base_folder, subject=subject, sess_slct=None, mode=mode)
    if len(paths) == 0:
        rc = subject + ':   no paths'
        return rc, None

    montage = my_montage_reader(fname=paths[0]['electrodes'])
    electrode_list = montage.get_electrode_list_by_region(region_list=regions, hemisphere_sel=sides)
    contact_list = []
    for group in electrode_list:
     contact_list = contact_list + electrode_list[group]
    num_contacts = len(contact_list)
    if num_contacts < 4:
        rc = 'not enough contacts for {} {}'.format(subject, group)
        #print(rc)
        return rc, None

    cntdwn_list = []
    for path in paths:
        event_reader_obj =  event_reader(path['events'])
        cntdwns = event_reader_obj.get_countdowns()
        # PATCH
        admit_list = np.argwhere([e['onset sample'] > 500 for e in cntdwns]).squeeze()
        cntdwns = [cntdwns[i] for i in admit_list]
        #
        if len(cntdwns) >= 26:#avg + in_session_gap:
            cntdwn_list.append(cntdwns)
    if len(cntdwn_list) < 2:
        rc = 'not enough valid sessions for {} {}'.format(subject, group)
        #print(rc)
        return rc, None

    print(subject, num_contacts, 'contacts', len(cntdwn_list), 'sessions')

    for path in paths:
        event_reader_obj = event_reader(path['events'])
        signals = my_mne_wrapper()
        signals.read_edf_file(fname=path['signals'], chanel_groups=electrode_list)
        event_reader_obj.align_to_sampling_rate(old_sfreq=signals.original_sfreq, new_sfreq=signals.get_mne().info['sfreq'])
        signals.preprocess(powerline=60)#, passband=[60, 160])
        #
        chan_names = signals.get_mne().info['ch_names']
        if len(chan_names) <= 1:
            rc = 'not enough channels for {} {}'.format(subject, group)
            #print(rc)
            return rc, None

        cntdwn_events = event_reader_obj.get_countdowns()
        # PATCH
        admit_list = np.argwhere([e['onset sample'] > 500 for e in cntdwn_events]).squeeze()
        cntdwn_events = [cntdwn_events[i] for i in admit_list]
        #
        events = np.zeros((len(cntdwn_events), 3), dtype=int)
        events[:, 0] = np.array([e['onset sample'] for e in cntdwn_events])
        signals.set_events(events=events, event_glossary={0: 'cntdwn'})

        if events[:, 0].max() > signals.get_mne().get_data().shape[-1]:
            rc = subject + ':      events[:, 0].max() > signals.get_mne().get_data().shape[-1]'
            #print(rc)
            return rc, None

        # process
        if not scan_files_only:
            p_list.append([])
            sem_list.append([])
            desc_list.append([])
            eidx = np.array((0, avg))
            for i in range(2):
                #print(path, signals.get_mne().get_data().shape)
                _, p, sem = calc_HFB(signals.get_mne().get_data(), dbg_markers=events[eidx[0]:eidx[1], 0], chan_names=chan_names,
                                     sub_centers=subs_centers, subs_bw=subs_bw, tscope=[-0.5, 10], plot_prefix=None, gen_plots=False)
                boundaries = np.arange(start=0, stop=p.shape[1]+1, step=int(signals.get_mne().info['sfreq'] * timebin_sec))
                p_list[-1].append(np.array([p[:, b1:b2].mean(axis=1) for (b1, b2) in zip(boundaries[:-1], boundaries[1:])]).T)
                sem_list[-1].append(np.array([sem[:, b1:b2].mean(axis=1) for (b1, b2) in zip(boundaries[:-1], boundaries[1:])]).T)
                desc_list[-1].append(path['signals'][path['signals'].find('ses'):][:5] + '  events {} - {}'.format(eidx[0], eidx[1]))
                eidx += in_session_gap


    rc = '{}   {} contacts   {} channels   {} sessions'.format(subject, num_contacts, len(chan_names), len(cntdwn_list))

    if scan_files_only:
        return rc, None
    else:
        fig, ax = plt.subplots(2, 4, figsize=(15, 10), num='calc_contdwn_responces')
        fig.clf()
        fig, ax = plt.subplots(2, 4, figsize=(15, 10), num='calc_contdwn_responces')
        #fig.suptitle(subject)
        for i_ses in range(4):
            for i_pos in range(2):
                if i_ses < len(desc_list):
                    sns.heatmap(p_list[i_ses][i_pos], ax=ax[i_pos, i_ses], vmin=0.0, vmax=3.0, cbar=False, square=False,
                                yticklabels=np.arange(len(chan_names)), xticklabels=(np.arange(start=-1, stop=20) + 0.5) / 2)
                    ax[i_pos, i_ses].set_xlabel(desc_list[i_ses][i_pos])
                else:
                    ax[i_pos, i_ses].axis('off')
        plt.show(block=False)
        plt.pause(0.25)

        return rc, dict({'p': p_list, 'sem': sem_list, 'desc': desc_list, 'fig': fig})







base_folder = 'E:/ds004789-download'
subjects_list = get_subject_list(base_folder=base_folder)

# aria of interest definition
region_list = ['entorihinal', 'cuneus', 'fusiform', 'lateraloccipital', 'lingual', 'precuneus', 'superiorpariental']
hemisphere_sel = ['LR', 'both', 'LR', 'LR', 'both', 'both', 'LR']
# create table of available data
if False:
    make_data_availability_list(base_folder=base_folder, region_list=region_list, hemisphere_sel=hemisphere_sel)
    assert False

# parameters for the protocol
regions = ['precuneus', 'cuneus', 'lateraloccipital', 'lingual']
sides = ['L', 'both', 'L', 'both']
subject_list = ['sub-R1425D', 'sub-R1355T', 'sub-R1346T', 'sub-R1338T', 'sub-R1334T', 'sub-R1161E', 'sub-R1156D', 'sub-R1154D', 'sub-R1153T',
                'sub-R1123C', 'sub-R1094T', 'sub-R1145J', 'sub-R1108J', 'sub-R1092J', 'sub-R1077T', 'sub-R1065J', 'sub-R1299T']
# subject_list = ['1355T', '1338T', '1337E', '1334T', '1323T', '1243T', '1331T', '1153T', '1134T', '1094T', '1108J', '1065J']
# subject_list = ['sub-R' + s for s in subject_list]
mode = 'bipolar'# 'monopolar'#
avg_depth = 6

logfile_name = os.path.join(base_folder, 'logfile.txt')
logfile_fd = open(logfile_name, 'wt')

for subject in subject_list:
    rc, data = calc_contdwn_responces(subject, regions=regions, sides=sides, mode=mode, scan_files_only=False, avg=avg_depth)
    print(subject, rc)
    logfile_fd.write(rc + '\n')
    if data is not None:
        data['fig'].suptitle(subject + '\navg={}'.format(avg_depth))
        fname = os.path.join(base_folder, 'plots', subject + '_' + mode + '_raw_avgs_' + str(avg_depth) + '.png')
        data['fig'].savefig(fname)
        #
        #
        #
        #
        # visualize
        nchans, nses = data['p'][0][0].shape[0], len(data['p'])
        num_lags = 4 # how many time taps to visualize
        vectors = np.zeros((2, nses, nchans, num_lags))
        #
        fig, ax = plt.subplots(2, num_lags, num='pierson', figsize=(15, 10))
        fig.clf()
        fig, ax = plt.subplots(2, num_lags, num='pierson', figsize=(15, 10))
        for i_lag in range(num_lags):
            ax[0, i_lag].grid(True)
            ax[0, i_lag].set_ylim([0.5, 3])
            ax[0, i_lag].set_title('{:2.1f} - {:2.1f} sec.'.format(i_lag / 2, (i_lag + 1) / 2))
        #
        for i_ses, (p_ses, sem_ses) in enumerate(zip(data['p'], data['sem'])):
            for i_rpt in range(2):
                # normalize the data
                nf = np.tile(p_ses[i_rpt][:, 0], (p_ses[i_rpt].shape[1], 1)).T
                # p_ses[i_rpt] /= nf
                # sem_ses[i_rpt] /= nf
                vectors[i_rpt, i_ses] = p_ses[i_rpt][:, 1:num_lags+1] / nf[:, 1:num_lags+1]
                for i_lag in range(num_lags):
                    ax[0, i_lag].plot(vectors[i_rpt, i_ses, :, i_lag], label='s{}r{}'.format(i_ses+1, i_rpt+1))
        # pierson correlation

        n1, n2, nchans, nlags = vectors.shape
        ur_vectors = vectors.reshape(n1*n2, nchans, nlags)
        pc = np.zeros((n1*n2, n1*n2))
        for i_lag in range(nlags):
            tap_vectors = ur_vectors[:, :, i_lag]
            mu = tap_vectors.mean(axis=0)
            ax[0, i_lag].plot(mu, ':', linewidth=2, label='avg')
            ax[0, i_lag].legend()
            # plain covariance
            c = (tap_vectors - mu) @ (tap_vectors - mu).T
            sigma = np.sqrt(np.diag(c))
            for i in range(c.shape[0]):
                c[:, i] /= sigma[i]
                c[i, :] /= sigma[i]
            # pairwise pierson
            for i1 in range(n1*n2):
                for i2 in range(n1*n2):
                    tap_vector_ij = tap_vectors[(i1, i2), :]
                    tap_vector_ij = tap_vector_ij * (tap_vector_ij < 4) # PATCH!!! FOR HIGH LEVEL NOISES
                    muij = tap_vector_ij.mean(axis=1)
                    dij = tap_vector_ij - muij.reshape(2, 1)
                    pc[i1, i2] = (dij[0].reshape(1, nchans) @ dij[1].reshape(nchans, 1)).squeeze() / (np.linalg.norm(dij[0]) * np.linalg.norm(dij[1]) + 1e-12)
                    # pc[i1, i2] = np.corrcoef(tap_vector_ij)[0, 1]

            sns.heatmap(np.round(pc, decimals=2), ax=ax[1, i_lag], vmin=-1, vmax=1, annot=True, cbar=False, square=True)

        fig.suptitle('PAIRWISE pearson correlation    ' + subject + '\navg={}'.format(avg_depth))
        fname = os.path.join(base_folder, 'plots', subject + '-' + mode + '_pearson' + str(avg_depth) + '.png')
        fig.savefig(fname)
        plt.show(block=False)
        plt.pause(0.25)




logfile_fd.close()

