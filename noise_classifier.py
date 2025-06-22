import os
#import mne.preprocessing
import numpy as np
import matplotlib.pyplot as plt
import path_utils
#from my_montage_reader import my_montage_reader
from my_mne_wrapper import my_mne_wrapper
#from event_reader import event_reader

from paths_and_constants import *

# import logging
# logging.basicConfig(filename=os.path.join(BASE_FOLDER, os.path.basename(__file__).replace('.py', '.log')), filemode='w', level=logging.DEBUG)




class noise_classifier:

    def __init__(self):

        pass


    def __gaussian(self, bins, m, sigma, alpha):

        return alpha * np.exp(-(((bins - m) ** 2) / (2 * sigma ** 2)))


    def __fit_gaussian(self, bins, h, init_m, init_sigma):

        def calc_J(bins, h, m, sigma, alpha):

            g = self.__gaussian(bins, m, sigma, alpha)
            return np.sum((h - g) ** 2)

        m, sigma = init_m, init_sigma
        alpha = h.max()
        # return m, sigma, alpha
        for i in range(10):
            J0 = calc_J(bins, h, m, sigma, alpha)
            # print(i, J0, m, sigma, alpha)
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

    def mark_bad_chans_and_segs(self, mne_copy, logger=None):

        mne_copy.filter(l_freq=0.1, h_freq=None, fir_design='firwin', filter_length='auto', phase='zero',
                        verbose=False)  # REMOVE DC swing
        ch_names =mne_copy.ch_names
        fs = mne_copy.info['sfreq']

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
                m, sigma, alpha = self.__fit_gaussian(bins=bins, h=h[0], init_m=mean_rbst, init_sigma=std_rbst)
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
                if logger:
                    logger.info('{}    channel {} {}    frac={:5.3f}  events={}'.format(subject, i_chan, ch_names[i_chan], frac_bad_windows, num_events))
                bad_channel_indicator[i_chan] = (frac_bad_windows > 0.05) or (num_events > 50)
                if bad_channel_indicator[i_chan]:  # simple_bad_ch_indicator[i_chan] or
                    if logger:
                        logger.info('  exclude channel {} {}'.format(i_chan, ch_names[i_chan]))
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
                    ax[0].plot(bins, self.__gaussian(bins, m, sigma, alpha), 'r')
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

        return mne_copy





if __name__ == '__main__':


    subject_list = path_utils.get_subject_list(BASE_FOLDER)[:2]

    noise_classifier_obj = noise_classifier()

    for subject in subject_list:
        paths = path_utils.get_paths(BASE_FOLDER, subject=subject, mode='bipolar')
        for path in paths:
            # try:
            # read .edf file and generate mne_object with annotation of bad chans and segs
            print('\n{}:'.format(path['signals']))
            mne_wrapper = my_mne_wrapper()
            mne_wrapper.read_edf_file(path['signals'])  # ), chanel_groups=electrodes)
            mne_wrapper.preprocess()
            assert mne_wrapper.get_mne().info['sfreq'] == 500
            mne_copy = mne_wrapper.get_mne().copy()
            mne_copy = noise_classifier_obj.mark_bad_chans_and_segs(mne_copy, logger=None)
            print(mne_copy.info['bads'], mne_copy.annotations)

