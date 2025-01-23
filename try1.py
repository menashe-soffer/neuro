import os

import mne
import numpy as np
import matplotlib.pyplot as plt

from path_utils import get_subject_list, get_paths
from event_reader import event_reader
from my_montage_reader import my_montage_reader
from my_mne_wrapper import my_mne_wrapper
from my_subband_processings import calc_HFB
from my_tfr_wrapper_1 import calc_tfr


base_folder = 'E:/ds004789-download'
subjects = get_subject_list(base_folder=base_folder)


# aria of interest definition
region_list = ['entorihinal', 'cuneus', 'fusiform', 'lateraloccipital', 'lingual', 'precuneus', 'superiorpariental']
hemisphere_sel = ['LR', 'both', 'LR', 'LR', 'both', 'both', 'LR']

# for debug - partial subject list
subjects = subjects[::10]
region_list = [region_list[2]]
hemisphere_sel = [hemisphere_sel[2]]

logfile_name = os.path.join(base_folder, 'logfile_' + region_list[0] + '.txt')
logfile_fd = open(logfile_name, 'wt')

for subject in subjects:

    paths = get_paths(base_folder=base_folder, subject=subject, sess_slct=None, mode='bipolar')
    # PATCH
    if len(paths) == 0:
        continue
    event_reader_obj = event_reader(paths[0]['events'])
    # print(subject, event_reader_obj.get_statistics())
    # continue


    montage = my_montage_reader(fname=paths[0]['electrodes'])
    electrode_list = montage.get_electrode_list_by_region(region_list=region_list, hemisphere_sel=hemisphere_sel)

    #print(electrode_list)
    if np.sum([len(electrode_list[r]) for r in electrode_list]) == 0:
        continue

    signals = my_mne_wrapper()
    signals.read_edf_file(fname=paths[0]['signals'], chanel_groups=electrode_list)
    #original_fs = signals.original_sfreq
    event_reader_obj.align_to_sampling_rate(old_sfreq=signals.original_sfreq, new_sfreq=signals.get_mne().info['sfreq'])
    # plt.plot(signals.get_mne().times)
    # plt.show()
    signals.preprocess(powerline=60)#, passband=[60, 160])
    chan_names = signals.get_mne().info['ch_names']
    if len(chan_names) == 0:
        continue

    cntdwn_events = event_reader_obj.get_countdowns()
    events = np.zeros((len(cntdwn_events), 3), dtype=int)
    events[:, 0] = np.array([e['onset sample'] for e in cntdwn_events])
    signals.set_events(events=events, event_glossary={0: 'cntdwn'})

    if events[:, 0].max() > signals.get_mne().get_data().shape[-1]:
        continue

    plot_folder = os.path.join(base_folder, 'plots')
    plot_prefix = os.path.join(plot_folder, subject + '_')
    MAX_CHANS_TO_PROCESS = 5
    sub_centers, subs_bw = [45, 55, 65, 75, 85, 95], 10#[70, 90, 110, 130, 150], 20#[77.5], 75#[45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95], 5#
    calc_HFB(signals.get_mne().get_data()[:MAX_CHANS_TO_PROCESS], dbg_markers=events[:, 0], chan_names=[subject + '\n' + c for c in chan_names],
            sub_centers=sub_centers, subs_bw=subs_bw, tscope=[-1, 10], plot_prefix=plot_prefix)
    # calc_tfr(signals.get_mne(), fs=500, events=events, span=[-1, 2], chan_names=None, plot_prefix=None)

    ###
    ## now process word practice events
    word_events = event_reader_obj.get_word_events()
    events = np.zeros((len(word_events), 3), dtype=int)
    events[:, 0] = np.array([e['onset sample'] for e in word_events])

    if events[:, 0].max() > signals.get_mne().get_data().shape[-1]:
        continue

    # calc_HFB(signals.get_mne().get_data()[:MAX_CHANS_TO_PROCESS], dbg_markers=events[:, 0], chan_names=[subject + '\n' + c for c in chan_names],
    #          sub_centers=sub_centers, subs_bw=subs_bw, tscope=[-0.6, 1.6], plot_prefix=plot_prefix + '__')
    ###
    ###


    #epochs = mne.Epochs(signals.get_mne(), event_id=['cntdwn'], tmin=-1.5, tmax=10.5)

    continue

    #epochs.compute_tfr(freqs=[60, 80], method='multitaper')
    #epochs.plot(scalings='auto')
    #plt.show()
    fig, ax = plt.subplots(2, 6, sharex=True)
    ax[0, 0].plot(epochs.average().data.T)
    for i_band, band in enumerate([[60, 80], [80, 100], [100, 120], [120, 140], [140, 160]]):
        tfr = epochs.compute_tfr(freqs=[min(band), max(band)], method='multitaper')
        ax[0, 1+i_band].plot(tfr.average().data.T[:, 0])
        ax[1, 1 + i_band].plot(tfr.average().data.T[:, 1])
    plt.show()

logfile_fd.close()
print('here')
