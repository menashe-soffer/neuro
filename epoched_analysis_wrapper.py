import numpy as np
import matplotlib.pyplot as plt
import os
import scipy
import pickle
import tqdm

#import glob
import mne

from path_utils import get_paths
#from my_montage_reader import my_montage_reader

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
            nf = np.linalg.norm(epoched._data[i_epoch, i_chan] * mask) / np.sqrt(mask.sum())
            #print(i_epoch, i_chan, nf)
            if nf > 0:
                epoched._data[i_epoch, i_chan] /= (nf + 1e-32)
            else:
                epoched._data[i_epoch, i_chan, :] = 0

    return

def my_wilcoxon(x, y):

    data_lvl = np.concatenate((x, y))
    #data_id = np.concatenate((np.zeros(x.shape), np.ones(y.shape)))
    rank = np.argsort(data_lvl)
    #ordered_data = data_lvl[rank]
    #ordered_id = data_id[rank]
    set_x = rank[:x.size]
    set_y = rank[-y.size:]
    # arrange the sets so that ie they have defferent sizes, the smallest is set_y
    if set_x.size < set_y.size:
        set_y, set_x = set_x, set_y
    n1, n2 = set_y.size, set_x.size
    # now calculate expected T and T
    Ty = set_y.sum()
    Ety = 0.5 * n1 * (n1 + n2 + 1)
    s = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = np.abs(Ty - Ety) / s
    return np.exp(-z/2)


def calculate_p_values(evoked, pre_intvl=[-0.5, -0.1], post_intval=[0.1, 0.6],
                       display_post_cursor=6, display_seperator_pitch=None, display_title=None, show=True):

    p_values = np.zeros(len(evoked.ch_names))
    increase_mask = np.zeros(len(evoked.ch_names), dtype=bool)
    pre_mask = (evoked.times > pre_intvl[0]) * (evoked.times < pre_intvl[-1])
    post_mask = (evoked.times > post_intval[0]) * (evoked.times < post_intval[-1])
    for i_ch in range(len(evoked.ch_names)):
        evoked._data[i_ch] = np.log(np.convolve(np.ones(8) / 8, evoked._data[i_ch], mode='same') + 1e-9)
        increase_mask[i_ch] = np.sign(evoked._data[i_ch][post_mask].mean() - evoked._data[i_ch][pre_mask].mean()) > 0
        p_values[i_ch] = my_wilcoxon(evoked._data[i_ch][pre_mask], evoked._data[i_ch][post_mask])

    figs = []
    if show:
        response_mask = increase_mask * (p_values < 0.05)  # log_p_thd
        for i_ch in range(len(evoked.ch_names)):
            i_ax = i_ch % 20
            if i_ax == 0:
                fig, ax = plt.subplots(4, 5, figsize=(12, 8), sharex=True, sharey=True)
                fig.suptitle(epoched_file)
                figs.append(fig)
            ax.flatten()[i_ax].plot(evoked.times[150:-150], evoked._data[i_ch][150:-150],
                                    linewidth=0.5 if evoked.ch_names[i_ch] in evoked.info['bads'] else 2,
                                    c='r' if response_mask[i_ch] else 'b')
            ax.flatten()[i_ax].plot(evoked.times[pre_mask], evoked._data[i_ch][pre_mask],
                                    linewidth=0.5 if evoked.ch_names[i_ch] in evoked.info['bads'] else 4,
                                    c='r' if response_mask[i_ch] else 'b')
            ax.flatten()[i_ax].plot(evoked.times[post_mask], evoked._data[i_ch][post_mask],
                                    linewidth=0.5 if evoked.ch_names[i_ch] in evoked.info['bads'] else 4,
                                    c='r' if response_mask[i_ch] else 'b')
            ax.flatten()[i_ax].set_ylabel(evoked.ch_names[i_ch])
            # [ax.flatten()[i_ax].plot([x, x], (-1, 1), c='r') for x in np.arange(10)] # for CNTDWV
            [ax.flatten()[i_ax].plot([x, x], (-1, 1), c='r') for x in np.arange(start=0, stop=display_post_cursor, step=display_seperator_pitch)]  # for LIST
            ax.flatten()[i_ax].set_ylim((-0.5, 0.75))
            # ax.flatten()[i_ax].set_xlim((-0.8, 6)) # for CNTDWV
            ax.flatten()[i_ax].set_xlim((-0.8, display_post_cursor))  # for LIST
            ax.flatten()[i_ax].set_xlabel('p value : {:7.4f}'.format(p_values[i_ch]))
            ax.flatten()[i_ax].grid(True)

    return p_values, increase_mask, figs


def add_synthetic_sub_annotations(epoched, event_type='CNTDWN', sub_event_type='DIGIT', num_subs_events=10, pitch=1):

    idxs = np.argwhere([d == event_type for d in epoched.annotations.description]).flatten()
    base_onsets = epoched.annotations.onset[idxs]
    sub_onsets = []
    for base_onset in base_onsets:
        sub_onsets.append(base_onset + pitch * np.arange(num_subs_events))
    sub_onsets = np.array(sub_onsets).flatten()
    epoched.annotations.append(sub_onsets, pitch * np.ones(sub_onsets.shape), sub_event_type)

    return


def process_epoched_file(epoched_file, event_type, sub_event_type, proc_params, proc_params_sub, SHOW, SHOW_ONLY):

    epoched = mne.read_epochs(epoched_file, verbose=False)
    # cmplx_signal = epoched.apply_hilbert().get_data()
    annotations = epoched.get_annotations_per_epoch()
    ch_names = epoched.ch_names
    bad_channels = epoched.info['bads']
    ch_mask = np.array([ch not in bad_channels for ch in ch_names])
    epoch_mask = np.zeros(len(annotations), dtype=bool)
    for i_epoch in range(len(annotations)):
        annot_type = [a[2] for a in annotations[i_epoch]]
        # print(annot_type)
        is_cntdwn = np.any([a == event_type for a in annot_type])
        is_bad = np.any([a.find('bad_') > -1 for a in annot_type])
        epoch_mask[i_epoch] = (len(annot_type) > 0) and (is_cntdwn) and (not is_bad)

    epoched_gamma = apply_gamma_responces(epoched=epoched, epoch_mask=epoch_mask)
    #
    # TEMPORARY PATCH: ADD SUB EVENTS
    if sub_event_type == 'DIGIT':
        add_synthetic_sub_annotations(epoched_gamma)
        annotations = epoched_gamma.get_annotations_per_epoch()

    evoked = epoched_gamma.average('data')
    norm_mask = (evoked.times >= -0.5) * (evoked.times <= -0.05)
    for i_chan in range(evoked._data.shape[0]):
        nf = np.linalg.norm(evoked._data[i_chan][norm_mask]) / np.sqrt(norm_mask.sum())
        evoked._data[i_chan] /= nf

    # # relation between major and sub events
    # idx_major = np.argwhere([e == event_type for e in epoched_gamma.annotations.description]).flatten()
    # idx_sub = np.argwhere([e == sub_event_type for e in epoched_gamma.annotations.description]).flatten()
    # onset_major = epoched_gamma.annotations.onset[idx_major]
    # onset_sub = epoched_gamma.annotations.onset[idx_sub]
    # relative_onsets = []
    # for i_major, (onset, next_onset) in enumerate(zip(onset_major[:-1], onset_major[1:])):
    #     relative_onsets.append(onset_sub[(onset_sub >= onset) * (onset_sub < next_onset)] - onset)

    p_values, increase_mask, _ = calculate_p_values(evoked, pre_intvl=proc_params['pre_intvl'], post_intval=proc_params['post_intvl'],
                                                    display_post_cursor=proc_params['display_post_cursor'], display_seperator_pitch=proc_params['display_seperator_pitch'],
                                                    display_title=epoched_file, show=SHOW)
    if not SHOW_ONLY:
        evoked_file = epoched_file.replace('ieeg-epo', '-' + event_type + '-ieeg-evoked-ave')
        evoked.save(evoked_file, overwrite=True)
        with open(os.path.join(os.path.dirname(epoched_file), 'p_values_' + event_type), 'wb') as fd:
            pickle.dump(dict({'ch_names': ch_names, 'p_values': p_values, 'increase_mask': increase_mask}), fd)
    #
    # sub event
    mask = np.array([d == event_type for d in epoched.annotations.description]).flatten()
    events_for_sub_epoching = np.zeros((mask.sum(), 3))
    events_for_sub_epoching[:, 0] = epoched.annotations.onset[mask]
    durations = epoched.annotations.duration[mask].max()

    fs = epoched_gamma.info['sfreq']
    epoched_data = epoched_gamma.get_data()
    major_duration = (epoched_data.shape[-1] - 1) / fs
    base_time = 0
    time_shift = -epoched_gamma.times[0]
    semi_raw = mne.io.RawArray(np.concatenate(epoched_data, axis=1), epoched.info, verbose=False)
    for major_epoch in annotations:
        sub_idxs = np.argwhere([e[-1] == sub_event_type for e in major_epoch]).flatten()
        sub_events_inner = np.array([major_epoch[i][:2] for i in sub_idxs])
        semi_raw.annotations.append(sub_events_inner[:, 0] + time_shift + base_time, sub_events_inner[:, 1], sub_event_type)
        base_time += major_duration
    events_for_epoching = np.zeros((len(semi_raw.annotations), 3), dtype=int)
    events_for_epoching[:, 0] = (semi_raw.annotations.onset * fs).astype(int)
    events_for_epoching[:, 2] = 0
    # BIG TBD: KKEP BAD EVENTS, ASSIGN EVENT TYPES TO DISTINGUISH
    sub_epoched = mne.Epochs(semi_raw, events=events_for_epoching, tmin=-1, tmax=sub_events_inner[:, 1].max() + 1,
                             baseline=None, preload=True)
    #normalize_all_epochs(sub_epoched)
    sub_evoked = sub_epoched.average('data')
    #
    norm_mask = (sub_evoked.times >= -0.5) * (sub_evoked.times <= -0.05)
    for i_chan in range(sub_evoked._data.shape[0]):
        nf = np.linalg.norm(sub_evoked._data[i_chan][norm_mask]) / np.sqrt(norm_mask.sum())
        sub_evoked._data[i_chan] /= nf
    #

    p_values, increase_mask, _ = calculate_p_values(sub_evoked, pre_intvl=proc_params_sub['pre_intvl'], post_intval=proc_params_sub['post_intvl'],
                                                    display_post_cursor=proc_params_sub['display_post_cursor'], display_seperator_pitch=proc_params_sub['display_seperator_pitch'],
                                                    display_title=epoched_file, show=SHOW)

    if not SHOW_ONLY:
        evoked_file = epoched_file.replace('ieeg-epo', '-' + sub_event_type + '-ieeg-evoked-ave')
        evoked.save(evoked_file, overwrite=True)
        with open(os.path.join(os.path.dirname(epoched_file), 'p_values_' + sub_event_type), 'wb') as fd:
            pickle.dump(dict({'ch_names': ch_names, 'p_values': p_values, 'increase_mask': increase_mask}), fd)

    if SHOW:
        plt.show()


def get_params_for_event_type(type_to_process):

    if TYPE_TO_PROCESS == 'cntdwn':
        file_type, event_type, sub_event_type = 'cntdwn', 'CNTDWN', 'DIGIT'
        proc_params = dict({'pre_intvl': [-0.5, -0.1], 'post_intvl': [0.1, 0.6], 'display_post_cursor': 8, 'display_seperator_pitch': 1})
        proc_params_sub = dict({'pre_intvl': [-0.4, -0.1], 'post_intvl': [0.1, 0.4], 'display_post_cursor': 3, 'display_seperator_pitch': 2.6})
    if TYPE_TO_PROCESS == 'list':
        file_type, event_type, sub_event_type = 'list', 'LIST', 'WORD'#
        proc_params = dict({'pre_intvl': [-0.5, -0.1], 'post_intvl': [0.1, 0.6], 'display_post_cursor': 30, 'display_seperator_pitch': 2.6})
        proc_params_sub = dict({'pre_intvl': [-0.4, -0.1], 'post_intvl': [0.1, 0.4], 'display_post_cursor': 3, 'display_seperator_pitch': 2.6})
    if TYPE_TO_PROCESS == 'orient':
        assert 'orient not implemented yet'

    return file_type, event_type, sub_event_type, proc_params, proc_params_sub




if __name__ == '__main__':

    import time

    SHOW, SHOW_ONLY = False, False
    assert SHOW or (not SHOW_ONLY)
    TYPE_TO_PROCESS = 'cntdwn'# 'list'# 'orient'#
    file_type, event_type, sub_event_type, proc_params, proc_params_sub = get_params_for_event_type(TYPE_TO_PROCESS)

    list = find_epoched_subject(base_folder=base_folder, epoched_folder=epoched_folder, type=file_type)
    #print(list)
    fail_list = []
    tstart = time.time()
    for subject_item in tqdm.tqdm(list):
        for epoched_file in subject_item['epoched']:
            try:
                process_epoched_file(epoched_file, event_type, sub_event_type, proc_params, proc_params_sub, SHOW, SHOW_ONLY)
            except:
                fail_list.append(epoched_file)

    elapsed = time.time() - tstart

    print('FAILED:', fail_list)
    print('total processing time: {:6.2f} minutes'.format(elapsed / 60))

