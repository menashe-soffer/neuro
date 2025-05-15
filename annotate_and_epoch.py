import os
import mne.preprocessing
import numpy as np
import matplotlib.pyplot as plt
import path_utils
#from my_montage_reader import my_montage_reader
from my_mne_wrapper import my_mne_wrapper
from event_reader import event_reader

from paths_and_constants import *

import logging
logging.basicConfig(filename=os.path.join(BASE_FOLDER, os.path.basename(__file__).replace('.py', '.log')), filemode='w', level=logging.DEBUG)


# area of interest definition
region_list = ['fusiform', 'inferiortemporal', 'lateraloccipital', 'lingual']
hemisphere_sel = ['LR', 'LR', 'LR', 'both']

subject_list =  ['sub-R1001P', 'sub-R1002P', 'sub-R1003P', 'sub-R1006P', 'sub-R1010J', 'sub-R1034D', 'sub-R1054J', 'sub-R1059J', 'sub-R1060M', 'sub-R1065J',
                 'sub-R1066P', 'sub-R1067P', 'sub-R1068J', 'sub-R1070T', 'sub-R1077T', 'sub-R1080E', 'sub-R1083J', 'sub-R1092J', 'sub-R1094T', 'sub-R1100D',
                 'sub-R1106M', 'sub-R1108J', 'sub-R1111M', 'sub-R1112M', 'sub-R1113T', 'sub-R1118N', 'sub-R1123C', 'sub-R1124J', 'sub-R1125T', 'sub-R1134T',
                 'sub-R1136N', 'sub-R1145J', 'sub-R1147P', 'sub-R1153T', 'sub-R1154D', 'sub-R1158T', 'sub-R1161E', 'sub-R1163T', 'sub-R1167M', 'sub-R1168T',
                 'sub-R1170J', 'sub-R1171M', 'sub-R1172E', 'sub-R1174T', 'sub-R1177M', 'sub-R1195E', 'sub-R1196N', 'sub-R1201P', 'sub-R1202M', 'sub-R1204T',
                 'sub-R1215M', 'sub-R1226D', 'sub-R1234D', 'sub-R1240T', 'sub-R1243T', 'sub-R1281E', 'sub-R1283T', 'sub-R1293P', 'sub-R1297T', 'sub-R1299T',
                 'sub-R1302M', 'sub-R1308T', 'sub-R1309M', 'sub-R1310J', 'sub-R1311T', 'sub-R1315T', 'sub-R1316T', 'sub-R1317D', 'sub-R1323T', 'sub-R1325C',
                 'sub-R1328E', 'sub-R1331T', 'sub-R1334T', 'sub-R1336T', 'sub-R1337E', 'sub-R1338T', 'sub-R1341T', 'sub-R1345D', 'sub-R1346T', 'sub-R1350D',
                 'sub-R1354E', 'sub-R1355T', 'sub-R1361C', 'sub-R1363T', 'sub-R1367D', 'sub-R1374T', 'sub-R1377M', 'sub-R1378T', 'sub-R1379E', 'sub-R1385E',
                 'sub-R1386T', 'sub-R1387E', 'sub-R1391T', 'sub-R1394E', 'sub-R1396T', 'sub-R1405E', 'sub-R1415T', 'sub-R1416T', 'sub-R1420T', 'sub-R1422T',
                 'sub-R1425D', 'sub-R1427T', 'sub-R1443D', 'sub-R1449T', 'sub-R1463E', 'sub-R1542J']


def gaussian(bins, m, sigma, alpha):

    return alpha * np.exp(-(((bins - m) ** 2) / (2 * sigma ** 2)))


def fit_gaussian(bins, h, init_m, init_sigma):

    def calc_J(bins, h, m, sigma, alpha):

        g = gaussian(bins, m, sigma, alpha)
        return np.sum((h - g) ** 2)


    m, sigma = init_m, init_sigma
    alpha = h.max()
    #return m, sigma, alpha
    for i in range(10):
        J0 = calc_J(bins, h, m, sigma, alpha)
        #print(i, J0, m, sigma, alpha)
        # optimise m
        srch_rng = 0.5 * sigma
        for i1 in range(10):
            srch_m = np.linspace(start=m - srch_rng, stop=m + srch_rng, num=11)
            J_vec = np.array([calc_J(bins, h, m0, sigma, alpha) for m0 in srch_m])
            m = srch_m[np.argmin(J_vec)]
            srch_rng *= 0.2
        # optimise sigma
        srch_rng = 0.2 * sigma
        for i1 in range(10):
            srch_sigma = np.linspace(start=sigma - srch_rng, stop=sigma + srch_rng, num=11)
            J_vec = np.array([calc_J(bins, h, m, s0, alpha) for s0 in srch_sigma])
            sigma = srch_sigma[np.argmin(J_vec)]
            srch_rng *= 0.2
        # optimise alpha
        srch_rng = 0.2 * alpha
        for i1 in range(10):
            srch_alpha = np.linspace(start=alpha - srch_rng, stop=alpha + srch_rng, num=11)
            J_vec = np.array([calc_J(bins, h, m, sigma, a0) for a0 in srch_alpha])
            alpha = srch_alpha[np.argmin(J_vec)]
            srch_rng *= 0.2


    return m, sigma, alpha


def read_edf_and_mark_bad_chans_and_segs(path):

    mne_wrapper = my_mne_wrapper()
    mne_wrapper.read_edf_file(path['signals'])  # ), chanel_groups=electrodes)
    mne_wrapper.preprocess()
    assert mne_wrapper.get_mne().info['sfreq'] == 500
    mne_copy = mne_wrapper.get_mne().copy()
    mne_copy.filter(l_freq=0.1, h_freq=None, fir_design='firwin', filter_length='auto', phase='zero',
                    verbose=False)  # REMOVE DC swing
    ch_names = mne_wrapper.get_mne().ch_names
    fs = mne_wrapper.get_mne().info['sfreq']

    MANUAL_PROCESSING = True
    if MANUAL_PROCESSING:
        # past 1: simple statistics
        total_chans, exclude_chans = 0, 0
        v = mne_copy.get_data()
        p = v.std(axis=-1)
        # a little bit more robust
        v_sorted = np.sort(v, axis=-1)
        p99 = np.percentile(np.abs(v), 99, axis=-1)
        for i_ch in range(v.shape[0]):
            v99 = v[i_ch][np.abs(v[i_ch]) < p99[i_ch]]
            p[i_ch] = v99.std()
        pavg = p.mean()
        pstd = p.std()
        rel_p = 20 * np.log10(p / pavg)
        # print(rel_p)
        # simple_bad_ch_indicator = np.abs((p - pavg) / pstd) > 4
        simple_bad_ch_indicator = (rel_p > 10) + (rel_p < -12)
        total_chans += v.shape[0]
        exclude_chans += simple_bad_ch_indicator.sum()
        SHOW = False
        if SHOW and simple_bad_ch_indicator.sum():
            fig, ax = plt.subplots(min(simple_bad_ch_indicator.sum(), 10), 1, figsize=(12, 8))
            ax = np.atleast_1d(ax)
            for i_ax, i_ch in enumerate(np.argwhere(simple_bad_ch_indicator).flatten()):
                if i_ax < 10:
                    ax[i_ax % 10].plot(v[i_ch], label='{} ({:4.1f})'.format(i_ch, 20 * np.log10(p[i_ch] / pavg)))
            for i_ax in range(len(ax)):
                ax[i_ax % 10].legend(loc='upper right')
            plt.show()
        # mne_copy.apply_hilbert()
        # v = np.abs(mne_copy.get_data())

        # past 2: robust statistics
        # continue
        bad_channel_indicator = np.zeros(simple_bad_ch_indicator.shape, dtype=bool)
        v = mne_copy.get_data()
        # v -= np.repeat(np.atleast_2d(v.mean(axis=1)).T, repeats=v.shape[1], axis=1)
        noise_mask = np.zeros((v.shape[0], int(v.shape[1] / 1000)))
        num_chans = v.shape[0]
        for i_chan in range(num_chans):
            mean = v[i_chan].mean()
            std = v[i_chan].std()
            max = v[i_chan].max()
            thd = np.percentile(np.abs(v[i_chan]), 95)
            v_rbst = v[i_chan][np.abs(v[i_chan]) <= thd]
            mean_rbst = v_rbst.mean()
            std_rbst = v_rbst.std()
            max_rbst = v_rbst.max()
            h = np.histogram(v[i_chan], bins=int(100 * (v[i_chan].max() - v[i_chan].min()) / std_rbst))
            bins = (h[1][:-1] + h[1][1:]) / 2
            # print('{}\t{:6.4f}\t{:6.4f}\t{:6.4f}\t{:6.4f}\t{:6.4f}\t{:6.4f}\t{:6.4f}'.format(i_chan, mean_rbst, std_rbst, max_rbst, mean, std, max, thd))
            m, sigma, alpha = fit_gaussian(bins=bins, h=h[0], init_m=mean_rbst, init_sigma=std_rbst)
            # print(m, sigma, alpha)
            tol = 15 * sigma
            mask = np.abs(v[i_chan] - m) > tol
            # expand the marks
            mask = np.convolve(np.ones(250), mask.astype(int), mode='same') > 0
            #
            starts = np.argwhere(np.diff(mask.astype(int)) == 1).flatten()
            stops = np.argwhere(np.diff(mask.astype(int)) == -1).flatten()
            # fill in small gaps
            # plt.plot(mask, label='init')
            if mask[0]:
                starts = np.concatenate((np.atleast_1d(0), np.atleast_1d(starts)))
            if mask[-1]:
                stops = np.concatenate((np.atleast_1d(stops), np.atleast_1d(mask.size + 1)))
            gaps = starts[1:] - stops[:-1]
            fill_gap = gaps < 500
            for gap_start, gap_end in zip(stops[:-1], starts[1:]):
                if gap_end - gap_start < 500:
                    mask[gap_start:gap_end + 1] = True
                    # plt.plot(mask, label=str(gap_start) + '-' + str(gap_end))
            # plt.legend()
            # plt.show()
            keep_gap = np.logical_not(fill_gap)
            starts = np.argwhere(np.diff(mask.astype(int)) == 1).flatten()
            stops = np.argwhere(np.diff(mask.astype(int)) == -1).flatten()
            if mask[0]:
                starts = np.concatenate((np.atleast_1d(0), np.atleast_1d(starts)))
            if mask[-1]:
                stops = np.concatenate((np.atleast_1d(stops), np.atleast_1d(mask.size + 1)))
            #
            only_at_end = (starts.size == 1) and (stops.size == 0) and (mask.sum() < 0.01 * v[i_chan].size)
            boundaris = np.linspace(start=0, stop=mask.size, num=noise_mask.shape[1] + 1).astype(int)
            mcounts = np.array([mask[i1:i2].sum() for (i1, i2) in zip(boundaris[:-1], boundaris[1:])])
            noise_mask[i_chan] = np.log(1 + np.array(mcounts)) / np.log(np.diff(boundaris).max())
            #
            # now decide if the channel is to be excluded
            frac_bad_windows = (mcounts > 0).sum() / mcounts.size
            num_events = starts.size
            logging.info('{}    channel {} {}    frac={:5.3f}  events={}'.format(subject, i_chan, ch_names[i_chan],
                                                                          frac_bad_windows, num_events))
            bad_channel_indicator[i_chan] = (frac_bad_windows > 0.05) or (num_events > 50)
            if bad_channel_indicator[i_chan]:  # simple_bad_ch_indicator[i_chan] or
                logging.info('  exclude channel {} {}'.format(i_chan, ch_names[i_chan]))
            else:
                description = 'bad_' + ch_names[i_chan]
                event_type = 1000 + i_chan
                onset = starts / fs
                duration = (stops - starts - 1) / fs
                # mne_copy.set_annotations(mne.Annotations(onset, duration, description))
                mne_copy.annotations.append(onset, duration, description)
            #
            SHOW = False
            if SHOW and np.any(mask) and (not only_at_end):
                # print(m, sigma, v[i_chan].max())
                fig, ax = plt.subplots(2, 1)
                ax[0].bar(bins, h[0], width=np.diff(bins)[0] * 0.8)
                ax[0].plot(bins, gaussian(bins, m, sigma, alpha), 'r')
                h0 = 0.05 * alpha
                ax[0].plot([m, m], [0, h0], 'k')
                ax[0].plot([m - 3 * sigma, m - 3 * sigma], [0, h0], 'k')
                ax[0].plot([m + 3 * sigma, m + 3 * sigma], [0, h0], 'k')
                ax[0].plot([m - tol, m - tol], [0, h0], 'k')
                ax[0].plot([m + tol, m + tol], [0, h0], 'k')
                ax[1].plot(np.arange(v[i_chan].size) / 500, v[i_chan], 'b')
                ax[1].scatter(np.argwhere(mask).squeeze() / 500, v[i_chan][mask], c='r')
                fig.suptitle(
                    'channel {} {}    frac={:5.3f}  events={}'.format(i_chan, ch_names[i_chan], frac_bad_windows,
                                                                      num_events))
                plt.show()

        #

        noise_mask_w_sums = np.zeros((noise_mask.shape[0] + 8, noise_mask.shape[1] + 8))
        noise_mask_w_sums[:-8, :-8] = noise_mask
        noise_mask_w_sums[-1, :-8] = noise_mask.sum(axis=0) / noise_mask.shape[1]
        noise_mask_w_sums[:-8, -1] = noise_mask.sum(axis=1) / noise_mask.shape[0]
        # print(noise_mask_w_sums[:, -1])

        drop_channels = np.logical_or(simple_bad_ch_indicator, bad_channel_indicator)
        if drop_channels.sum() > 0:
            mne_copy.info['bads'] = [ch_names[i] for i in np.argwhere(drop_channels).flatten()]
            # mne_copy.drop_channels()

        SHOW = False
        if SHOW:
            plt.imshow(noise_mask_w_sums, cmap='gray', aspect='auto')
            mne_copy.plot()
            plt.show()

    return mne_copy, mne_wrapper.original_sfreq

def generate_epoched_version(path, mne_copy, event_obj, create_new=True, event_name='cntdwn'):

    epoched_fname = path['signals'].replace(BASE_FOLDER, PROC_FOLDER).replace('ieeg', event_name, 1).replace('.edf', '-epo.fif')
    if os.path.isfile(epoched_fname) and (not create_new):
        logging.info(epoched_fname + 'ALREADY EXISTS')
        return

    fs = mne_copy.info['sfreq']

    if event_name == 'random':
        epoching_events = event_obj.get_countdowns()
        # RANDOMIZE ONSETS
        rand_samps = np.sort(np.random.uniform(low=1000, high=mne_copy.times.size-20000, size=len(epoching_events))).astype(int)
        rand_onset = mne_copy.times[rand_samps]
        for i_event, event in enumerate(epoching_events):
            event['onset'] = rand_onset[i_event]
            event['onset sample'] = rand_samps[i_event]
            event['end sample'] = rand_samps[i_event] + 5250
        #
        description = 'RANDOM'
        sub_description = None
        tmin, tmax = -2.5, 10.5 + 2.5
        sub_description, sub_events = None, None
    if event_name == 'cntdwn':
        epoching_events = event_obj.get_countdowns()
        description = 'CNTDWN'
        sub_description = 'DIGIT'
        tmin, tmax = -2.5, 10.5 + 2.5
        # generating the DIGIT annotations
        sub_events = []
        for cntdwn_event in epoching_events:
            sub_events = sub_events + [{'onset':  cntdwn_event['onset'] + float(i), 'end': cntdwn_event['onset'] + float(i + 1),
                                        'onset sample': cntdwn_event['onset sample'] + int(fs * i),
                                        'end sample': cntdwn_event['onset sample'] + int(fs * (i + 1))}  for i in range(10)]
    if event_name == 'orient':
        epoching_events = event_obj.get_orients()
        description = 'ORIENT'
        sub_description, sub_events = None, None
        tmin, tmax = -2.5-2, 9 + 2.5+2
    if event_name == 'list':
        epoching_events = event_obj.get_list_events()
        description = 'LIST'
        sub_events = event_obj.get_word_events()
        sub_description = 'WORD'
        tmin, tmax = -2.5, 30.5 + 2.5
    if event_name == 'recall':
        epoching_events = event_obj.get_recalls(random_start=True, max_duration=12)
        description = 'RECALL'
        sub_description, sub_events = None, None
        tmin, tmax = -2.5, 9 + 2.5
    if event_name == 'dstrct':
        epoching_events = event_obj.get_distracts(random_start=True, max_duration=12)
        description = 'DSTRCT'
        sub_description, sub_events = None, None
        tmin, tmax = -2.5, 9 + 2.5
    if event_name == 'rest':
        epoching_events = event_obj.get_rests(random_start=True, max_duration=12)
        description = 'REST'
        sub_description, sub_events = None, None
        tmin, tmax = -2.5, 9 + 2.5

    event_type = EVENT_TYPES[description]
    sub_event_type = None if sub_description == None else EVENT_TYPES[sub_description]

    #onset = np.array([e['onset'] for e in countdown_events])
    onset = np.array([e['onset sample'] / fs for e in epoching_events])
    duration = np.array([(e['end sample'] - e['onset sample']) / fs for e in epoching_events])
    mne_copy.annotations.append(onset, duration, description)
    if sub_events is not None:
        sub_onset = np.array([e['onset sample'] / fs for e in sub_events])
        sub_duration = np.array([(e['end sample'] - e['onset sample']) / fs for e in sub_events])
        mne_copy.annotations.append(sub_onset, sub_duration, sub_description)
    onset_sample = np.array([e['onset sample'] for e in epoching_events])
    events_for_epoching = np.zeros((onset_sample.size, 3), dtype=int)
    events_for_epoching[:, 0] = onset_sample
    events_for_epoching[:, 2] = event_type
    # crop_start = onset - 2
    # crop_end = onset + duration + 2
    # make sure the events are chronologically ordered
    order = np.argsort(events_for_epoching[:, 0])
    events_for_epoching = events_for_epoching[order]

    # save
    #mne_copy._data[2] = 1e-6 * (np.arange(mne_copy._data.shape[-1]) % 1000 - 500)
    epoched = mne.Epochs(mne_copy, events=events_for_epoching, tmin=tmin, tmax=tmax, reject_by_annotation=False, verbose=False)
    #epoched.ch_amps = mne_copy.get_data().std(axis=1)
    os.makedirs(os.path.dirname(epoched_fname), exist_ok=True)
    epoched.save(epoched_fname, overwrite=True, verbose=False)

    SHOW = False
    if SHOW:
        epoched1 = mne.read_epochs(epoched_fname)
        _ = epoched1.plot(scalings=2e-4)
        plt.show()



def check_if_files_exist(src_path, event_names, force_override, verbose=True):

    exist_mask = np.zeros(len(event_names), dtype=bool)
    if not force_override:
        for i, event_name in enumerate(event_names):
            epoched_path = src_path['signals'].replace(BASE_FOLDER, PROC_FOLDER).replace('ieeg', event_name, 1).replace('.edf', '-epo.fif')
            exist_mask[i] = os.path.isfile(epoched_path)
            if exist_mask[i] and verbose:
                logging.info(epoched_path + '  ALREADY EXISTS')

    events_to_process = [event_names[i] for i in np.argwhere(np.logical_not(exist_mask)).flatten()]
    return exist_mask, events_to_process


def psd_wrapper(raw_obj):

    sfreq = raw_obj.info['sfreq']
    n_fft = 2024
    raw_clean = raw_obj.copy()
    for onset, duration, desc in zip(raw_obj.annotations.onset,
                                     raw_obj.annotations.duration,
                                     raw_obj.annotations.description):
        if desc.upper().startswith('BAD'):
            start_sample = int(onset * sfreq)
            stop_sample = int((onset + duration) * sfreq)
            start_sample =int(np.floor(start_sample / (n_fft / 2)) * n_fft / 2)
            stop_sample = int(np.ceil(stop_sample / (n_fft / 2)) * n_fft / 2)
            ch_to_clear = np.argwhere([desc[4:] == c for c in raw_clean.ch_names]).squeeze()
            raw_clean._data[ch_to_clear, start_sample:stop_sample] = 0.0  # Zero out

    # Now compute PSD on the zeroed version
    psd = raw_clean.compute_psd(method='welch', fmin=0.5, fmax=sfreq / 2, verbose=False, reject_by_annotation=False, n_fft=n_fft)
    evoked = mne.EvokedArray(psd.data, psd.info, tmin=0.)

    return evoked


FORCE_OVERRIDE = False
from tqdm import tqdm
fail_list = []
events_to_process = ['orient', 'dstrct', 'recall', 'cntdwn', 'list', 'rest'] #orient has a bug' need to troubleshoot
for subject in tqdm(subject_list):
    paths = path_utils.get_paths(BASE_FOLDER, subject=subject, mode='bipolar')
    for path in paths:
        _, events_to_process_for_session = check_if_files_exist(src_path=path, event_names=events_to_process, force_override=FORCE_OVERRIDE, verbose=True)
        if len(events_to_process_for_session) > 0:
            try:
                # read .edf file and generate mne_object with annotation of bad chans and segs
                mne_copy, original_sfreq = read_edf_and_mark_bad_chans_and_segs(path)
                # read the annotations
                event_obj = event_reader(fname=path['events'])
                event_obj.align_to_sampling_rate(old_sfreq=original_sfreq, new_sfreq=mne_copy.info['sfreq'])
                read_success = True
            except:
                logging.warning('FAILED TO READ  ' + path['signals'])
                fail_list.append(path['signals'])
                read_success = False
            if read_success:
                for event_name in events_to_process_for_session:
                    try:
                        #
                        psd = psd_wrapper(mne_copy)
                        psd_fname = path['signals'].replace(BASE_FOLDER, PROC_FOLDER).replace('ieeg', 'PSD').replace('.edf', '_ave.fif')
                        os.makedirs(os.path.dirname(psd_fname), exist_ok=True)
                        psd.save(psd_fname, overwrite=FORCE_OVERRIDE)
                        #
                        generate_epoched_version(path, mne_copy=mne_copy, event_obj=event_obj, create_new=FORCE_OVERRIDE, event_name=event_name)
                    except:
                            fail_list.append(path['signals'] + ' : ' + event_name)
                            logging.warning('FAILED TO GENERATE .fif for {} FROM   {}'.format(event_name, path['signals']))

if len(fail_list) > 0:
    print('FAILED TO GENERATE THE FOLLOWING FILES:')
    for fname in fail_list:
        print('\t', fname)

#print('total channels:   {}   excluded channels:  {}'.format(total_chans, exclude_chans))