import mne
import numpy as np
import matplotlib.pyplot as plt

from path_utils import get_subject_list, get_paths
from event_reader import event_reader
from my_montage_reader import my_montage_reader
from my_mne_wrapper import my_mne_wrapper


base_folder = 'E:/ds004789-download'
subjects = get_subject_list(base_folder=base_folder)

# for debug - partial subject list
subjects = subjects[::20]

# aria of interest definition
region_list = ['entorihinal', 'cuneus', 'fusiform', 'lateraloccipital', 'lingual', 'precuneus', 'superiorpariental']
hemisphere_sel = ['LR', 'both', 'LR', 'LR', 'both', 'both', 'LR']

for subject in subjects:

    paths = get_paths(base_folder=base_folder, subject=subject, sess_slct=[0], mode='bipolar')
    events = event_reader(paths[0]['events'])
    cntdwn_events = events.get_countdowns()

    montage = my_montage_reader(fname=paths[0]['electrodes'])
    electrode_list = montage.get_electrode_list_by_region(region_list=region_list, hemisphere_sel=hemisphere_sel)

    signals = my_mne_wrapper()
    signals.read_edf_file(fname=paths[0]['signals'], chanel_groups=electrode_list)
    signals.preprocess(powerline=60, passband=[60, 160])

    events = np.zeros((len(cntdwn_events), 3), dtype=int)
    events[:, 0] = np.array([e['onset sample'] for e in cntdwn_events])
    signals.set_events(events=events, event_glossary={0: 'cntdwn'})

    # signals.get_mne().plot(scalings='auto', duration=20, title=subject, event_id={0: 'countdown'})
    # signals.get_mne().plot_psd()
    # plt.show()
    # signals.preprocess(powerline=60, passband=[60, 160])
    # signals.mne.plot(scalings='auto', duration=20, title=subject, event_id={0: 'countdown'})
    # signals.get_mne().plot_psd()
    # plt.show()


    epochs = mne.Epochs(signals.get_mne(), event_id=['cntdwn'], tmin=-1.5, tmax=10.5)

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

    print('here')
