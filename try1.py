import os

import mne
import numpy as np
import matplotlib.pyplot as plt

from path_utils import get_subject_list, get_paths
from event_reader import event_reader
from my_montage_reader import my_montage_reader
from my_mne_wrapper import my_mne_wrapper
from my_subband_processings import calc_HFB


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
    events = event_reader(paths[0]['events'])
    cntdwn_events = events.get_countdowns()

    montage = my_montage_reader(fname=paths[0]['electrodes'])
    electrode_list = montage.get_electrode_list_by_region(region_list=region_list, hemisphere_sel=hemisphere_sel)

    #print(electrode_list)
    if np.sum([len(electrode_list[r]) for r in electrode_list]) == 0:
        continue

    signals = my_mne_wrapper()
    signals.read_edf_file(fname=paths[0]['signals'], chanel_groups=electrode_list)
    original_fs = signals.original_sfreq
    # plt.plot(signals.get_mne().times)
    # plt.show()
    signals.preprocess(powerline=60)#, passband=[60, 160])
    chan_names = signals.get_mne().info['ch_names']
    if len(chan_names) == 0:
        continue

    events = np.zeros((len(cntdwn_events), 3), dtype=int)
    events[:, 0] = np.array([e['onset sample'] for e in cntdwn_events])
    events[:, 0] = (events[:, 0] * (500 / original_fs)).astype(np.int64)
    signals.set_events(events=events, event_glossary={0: 'cntdwn'})
    # PATCH
    logfile_fd.write('{}  \t {}  \t  {}   ({})   fs={}\n'.format(subject, signals.get_mne().get_data().shape[-1], events[:, 0].max(),
                                                         signals.get_mne().get_data().shape[-1] - events[:, 0].max(), original_fs))
    logfile_fd.flush()
    if events[:, 0].max() > signals.get_mne().get_data().shape[-1]:
        continue

    # signals.get_mne().plot(scalings='auto', duration=20, title=subject, event_id={0: 'countdown'})
    # signals.get_mne().plot_psd()
    # plt.show()
    # signals.preprocess(powerline=60)#, passband=[60, 160])
    # signals.mne.plot(scalings='auto', duration=20, title=subject, event_id={0: 'countdown'})
    # signals.get_mne().plot_psd()
    # plt.show()

    plot_folder = os.path.join(base_folder, 'plots')
    plot_prefix = os.path.join(plot_folder, subject + '_')
    MAX_CHANS_TO_PROCESS = 5
    calc_HFB(signals.get_mne().get_data()[:MAX_CHANS_TO_PROCESS], show_dbg=True, dbg_markers=events[:, 0], chan_names=[subject + '\n' + c for c in chan_names], plot_prefix=plot_prefix)


    epochs = mne.Epochs(signals.get_mne(), event_id=['cntdwn'], tmin=-1.5, tmax=10.5)

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
