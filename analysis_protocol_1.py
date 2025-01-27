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


def calc_contdwn_responces(subject, region=None, band=[40, 100], timebin_sec=0.5, avg=6, in_session_gap=13, scan_files_only=True):

    # configurations for the sub=band processor
    # generate the sub-band spec for the processing
    subs_centers = np.linspace(start=band[0], stop=band[-1], num=int((band[-1] - band[0]) / 10 + 1.5))
    subs_centers = (subs_centers[:-1] + subs_centers[1:]) / 2
    subs_bw = np.diff(subs_centers)[0]
    # result holders
    p_list, sem_list, desc_list = [], [], []

    paths = get_paths(base_folder=base_folder, subject=subject, sess_slct=None, mode='bipolar')
    if len(paths) == 0:
        rc = subject + ':   no paths'
        return rc, None

    montage = my_montage_reader(fname=paths[0]['electrodes'])
    electrode_list = montage.get_electrode_list_by_region(region_list=[region[0]], hemisphere_sel=[region[1]])
    group = list(electrode_list.keys())[0]
    contact_list = electrode_list[group]
    num_contacts = len(contact_list)
    if num_contacts < 4:
        rc = 'not enough contacts for {} {}'.format(subject, group)
        #print(rc)
        return rc, None

    cntdwn_list = []
    for path in paths:
        event_reader_obj =  event_reader(path['events'])
        cntdwns = event_reader_obj.get_countdowns()
        if len(cntdwns) > avg + in_session_gap:
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
        fig.suptitle(subject)
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
region = 'precuneus'
side = 'L'
subject_list = ['1355T', '1338T', '1337E', '1334T', '1323T', '1243T', '1331T', '1153T', '1134T', '1094T', '1108J', '1065J']
subject_list = ['sub-R' + s for s in subject_list]

logfile_name = os.path.join(base_folder, 'logfile.txt')
logfile_fd = open(logfile_name, 'wt')

for subject in subject_list:
    rc, data = calc_contdwn_responces(subject, region=[region, side], scan_files_only=False)
    print(subject, rc)
    logfile_fd.write(rc + '\n')
    if data is not None:
        fname = os.path.join(base_folder, 'plots', subject + '.png')
        data['fig'].savefig(fname)

logfile_fd.close()

