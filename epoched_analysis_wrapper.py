import numpy as np
import matplotlib.pyplot as plt
import os
import scipy
#import glob
import mne

from path_utils import get_paths
from my_montage_reader import my_montage_reader

global base_folder
base_folder = 'E:/ds004789-download'
global epoched_folder
epoched_folder = 'E:/epoched'

def find_epoched_subject(base_folder, epoched_folder, min_sessions=2, type='cntdwn'):

    epoched_subjects = os.listdir(epoched_folder)
    mask = np.zeros(len(epoched_subjects))
    output = []
    for i, subject in enumerate(epoched_subjects):
        fif_counter = 0
        sessions = os.listdir(os.path.join(base_folder, subject))
        subject_epoched_files = []
        for i_sess, sess in enumerate(sessions):
            fif_pattern = os.path.join(epoched_folder, subject, sess, type, subject + '_' + sess + '_task-FR1_acq-bipolar_ieeg-epo.fif')
            fif_counter += os.path.isfile(fif_pattern)
            subject_epoched_files.append(fif_pattern)
        paths = get_paths(base_folder=base_folder, subject=subject, mode='bipolar')
        montage_path = paths[0]['electrodes']
        if fif_counter >= min_sessions:
            output.append({'subject': subject, 'montage': montage_path, 'epoched': subject_epoched_files})

    return output


def apply_gamma_responces(epoched, epoch_mask=None, gamma_band=[60, 160], keep_relative_channel_amps=False):

    # #
    # fig, ax = plt.subplots(6, 1, sharex=True)
    # tmp = epoched.get_data()[:, :6]
    # for i_ch in range(6):
    #     ax[i_ch].plot(tmp[:, i_ch].reshape(tmp.shape[0]*tmp.shape[-1]), linewidth=2, c='k')
    #     for mrkr in np.arange(start=tmp.shape[-1], step=tmp.shape[-1], stop=np.prod(tmp.shape) / tmp.shape[1]):
    #         ax[i_ch].plot((mrkr, mrkr), (-1, 1), c='b')
    #         ax[i_ch].set_ylim([-1e-3, 1e-3])
    # fig = epoched.plot(scalings=2e-4)
    # fig.suptitle('original')
    # plt.show()
    #
    processed = epoched.copy()
    processed._data[:, :, :] = 0

    fs = epoched.info['sfreq']
    bw = 20
    h = scipy.signal.firls(73, [0, 0.45*bw, 0.55*bw, fs/2], [1, 1, 0, 0], fs=fs)
    no_transient_mask = np.ones(epoched._data.shape[-1])
    no_transient_mask[:150] = 0
    no_transient_mask[-150:] = 0
    # w, H = scipy.signal.freqz(h, 1, fs=fs)
    # ax.semilogy(w, np.real(H * np.conj(H)))
    # ax.grid(True)
    # ax.set_ylim(1e-8, 2)
    # plt.show()
    boundaries = np.arange(start=gamma_band[0], stop=gamma_band[-1]+bw/2, step=bw)
    centers = (boundaries[:-1] + boundaries[1:]) / 2
    band_pwr = np.zeros((epoched.get_data().shape[1], centers.size))
    for i_cent, center in enumerate(centers):
        centered = epoched.copy().apply_hilbert()
        cmplx_signal = centered.get_data()
        shift_signal = np.exp(-1j * 2 * np.pi * center * np.arange(cmplx_signal.shape[-1]) / fs)
        for i_epoch in range(cmplx_signal.shape[0]):
            if epoch_mask[i_epoch]:
                for i_chan in range(cmplx_signal.shape[1]):
                    centered._data[i_epoch, i_chan] = scipy.signal.filtfilt(h, 1, cmplx_signal[i_epoch, i_chan] * shift_signal) * no_transient_mask
                    band_pwr[i_chan, i_cent] += np.real(centered._data[i_epoch, i_chan] * np.conj(centered._data[i_epoch, i_chan])).mean() / epoch_mask.sum()
                    # if (i_chan == 10) and (i_epoch == 10):
                    #     shifted_signal = cmplx_signal[i_epoch, i_chan] * shift_signal
                    #     plt.plot(np.fft.fftshift(np.abs(np.fft.fft(shifted_signal))))
                    #     plt.plot(np.fft.fftshift(np.abs(np.fft.fft(centered._data[i_epoch, i_chan]))))
                    #     plt.grid(True)
                    #     plt.ylim((0, 0.1))
                    #     plt.title('center: {:4.0f}   pwr:  {:8.5f}'.format(center, 1e6*band_pwr[i_chan, i_cent]))
                    #     plt.show()
        #
        # normalizing bands
        assert not keep_relative_channel_amps
        for i_epoch in range(cmplx_signal.shape[0]):
            if epoch_mask[i_epoch]:
                for i_chan in range(cmplx_signal.shape[1]):
                    centered._data[i_epoch, i_chan] /= np.sqrt(band_pwr[i_chan, i_cent])
                    processed._data[i_epoch, i_chan] += np.abs(centered._data[i_epoch, i_chan]) / centers.size
        #
    #     # for DEBUG (confirm normalizing)
    #     if i_cent == 0:
    #         normalized_band_pwr = np.zeros((epoched.get_data().shape[1], centers.size))
    #     for i_epoch in range(cmplx_signal.shape[0]):
    #         for i_chan in range(cmplx_signal.shape[1]):
    #             normalized_band_pwr[i_chan, i_cent] += np.real(centered._data[i_epoch, i_chan] * np.conj(centered._data[i_epoch, i_chan])).mean() / cmplx_signal.shape[0]
    # #     fig = centered.plot(scalings=2, title='band {}'.format(str(i_cent)))
    # # plt.show()
    # fig = processed.plot(scalings=2)
    # plt.show()

    return processed



def normalize_all_epochs(epoched, baseline_boundaries=(-0.5, -0.05)):

    mask = (epoched.times > baseline_boundaries[0]) * (epoched.times < baseline_boundaries[-1])
    num_epochs, num_chans, num_epoch_samps = epoched._data.shape
    for i_epoch in range(num_epochs):
        for i_chan in range(num_chans):
            nf = np.norm(epoched._data[i_epoch, i_chan] * mask) / mask.sum()
            epoched._data[i_epoch, i_chan] /= nf

    return

if __name__ == '__main__':

    list = find_epoched_subject(base_folder=base_folder, epoched_folder=epoched_folder)
    print(list)
    for subject_item in list:
        for epoched_file in subject_item['epoched']:
            epoched = mne.read_epochs(epoched_file, verbose=False)
            #cmplx_signal = epoched.apply_hilbert().get_data()
            annotations = epoched.get_annotations_per_epoch()
            ch_names = epoched.ch_names
            bad_channels = epoched.info['bads']
            ch_mask = np.array([ch not in bad_channels for ch in ch_names])
            epoch_mask = np.zeros(len(annotations), dtype=bool)
            for i_epoch in range(len(annotations)):
                annot_type = [a[2] for a in annotations[i_epoch]]
                print(annot_type)
                is_cntdwn = np.any([a == 'CNTDWN' for a in annot_type])
                is_bad = np.any([a.find('bad_') > -1 for a in annot_type])
                epoch_mask[i_epoch] = (len(annot_type) > 0) and (is_cntdwn) and (not is_bad)

            #cmplx_signal = cmplx_signal[epoch_mask][:, ch_mask]
            epoched_gamma = apply_gamma_responces(epoched=epoched, epoch_mask=epoch_mask)

            evoked = epoched_gamma.average('data')
            norm_mask = (evoked.times >= -0.5) * (evoked.times <= -0.05)
            for i_chan in range(evoked._data.shape[0]):
                nf = np.linalg.norm(evoked._data[i_chan][norm_mask]) / np.sqrt(norm_mask.sum())
                evoked._data[i_chan] /= nf
            # fig = evoked.plot()
            # plt.show()
            #
            # for i_ch in range(len(evoked.ch_names)):
            #     i_ax = i_ch % 20
            #     if i_ax == 0:
            #         fig, ax = plt.subplots(4, 5, figsize=(12, 8), sharex=True, sharey=True)
            #     ax.flatten()[i_ax].plot(evoked.times[150:-150], evoked._data[i_ch][150:-150],
            #                             linewidth=0.5 if evoked.ch_names[i_ch] in evoked.info['bads'] else 2)
            #     ax.flatten()[i_ax].set_ylabel(evoked.ch_names[i_ch])
            #     [ax.flatten()[i_ax].plot([x, x], (0.75, 1.25), c='r') for x in np.arange(10)]
            #     ax.flatten()[i_ax].set_ylim((0.5, 1.8))
            #     ax.flatten()[i_ax].set_xlim((-0.8, 6))
            #     ax.flatten()[i_ax].grid(True)
            # plt.show()
            #
            p_values = np.zeros(len(evoked.ch_names))
            pre_mask = (evoked.times > -0.6) * (evoked.times < -0.1)
            post_mask = (evoked.times > 0.1) * (evoked.times < 0.6)
            for i_ch in range(len(evoked.ch_names)):
                evoked._data[i_ch] = np.log(np.convolve(np.ones(8)/8, evoked._data[i_ch], mode='same') + 1e-9)
                _, p_values[i_ch] = scipy.stats.wilcoxon(evoked._data[i_ch][pre_mask], evoked._data[i_ch][post_mask])
                p_values[i_ch] *= np.sign(evoked._data[i_ch][post_mask].mean() - evoked._data[i_ch][pre_mask].mean())
            for i_ch in range(len(evoked.ch_names)):
                i_ax = i_ch % 20
                if i_ax == 0:
                    fig, ax = plt.subplots(4, 5, figsize=(12, 8), sharex=True, sharey=True)
                    fig.suptitle(epoched_file)
                ax.flatten()[i_ax].plot(evoked.times[150:-150], evoked._data[i_ch][150:-150],
                                        linewidth=0.5 if evoked.ch_names[i_ch] in evoked.info['bads'] else 2)
                ax.flatten()[i_ax].set_ylabel(evoked.ch_names[i_ch])
                [ax.flatten()[i_ax].plot([x, x], (-1, 1), c='r') for x in np.arange(10)]
                ax.flatten()[i_ax].set_ylim((-0.5, 0.75))
                ax.flatten()[i_ax].set_xlim((-0.8, 6))
                ax.flatten()[i_ax].set_xlabel('p value : {:7.6f}'.format(p_values[i_ch]))
                ax.flatten()[i_ax].grid(True)
            plt.show()

