import numpy as np
import scipy
import matplotlib.pyplot as plt # necessary for debug
import time
import mne




def calc_tfr(mne_obj, fs=500,  events=None, span=[-1, 2], chan_names=None, plot_prefix=None):

    n_cycles = 11
    freqs = np.linspace(start=40, stop=115, num=31)
    decim = 5

    num_chans = min(mne_obj.info['nchan'], 6)

    epochs = mne.Epochs(mne_obj, events=events, tmin=span[0], tmax=span[1])
    power, itc = epochs.compute_tfr(method='morlet', n_cycles=n_cycles, freqs=freqs, decim=decim, return_itc=True, average=True)
    fig, ax = plt.subplots(2, 3, figsize=(12, 8), num='power')
    power.plot(picks=np.arange(num_chans), baseline=(-0.5, 0), mode="logratio", axes=ax.flatten()[:num_chans], title=power.ch_names[:num_chans])
    fig, ax = plt.subplots(2, 3, figsize=(12, 8), num='itc')
    itc.plot(picks=np.arange(num_chans), baseline=(-0.5, 0), mode="logratio", axes=ax.flatten()[:num_chans], title=power.ch_names[:num_chans])
    # plt.show()

    power = epochs.compute_tfr(method='morlet', n_cycles=n_cycles, freqs=freqs, decim=decim, average=False)
    vmin = np.log10(power.data + 1e-9 * power.data.max()).min()
    vmax = np.log10(power.data + 1e-9 * power.data.max()).max()
    for i_ch, ch in enumerate(power.ch_names):
        fig, ax = plt.subplots(5, 8, figsize=(12, 8), sharex=True, sharey=True, num=i_ch+1)
        fig.suptitle(power.ch_names[i_ch])
        for i_epoch in range(power.data.shape[0]):
            ax.flatten()[i_epoch].imshow(np.log10(power.data[i_epoch, i_ch] + 1e-9 * power.data.max()), aspect='auto', vmin=vmin, vmax=vmax)
            ax.flatten()[i_epoch].grid(True)
        xticks = np.argwhere(power.times % 0.5 == 0).flatten()
        ax[-1, 0].set_xticks(xticks, labels=power.times[xticks])
        yticks = np.arange(0, len(power.freqs), step=int(len(power.freqs) / 4))
        ax[0, -1].set_yticks(yticks, power.freqs[yticks])
    plt.show()





