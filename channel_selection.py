import numpy as np
import matplotlib.pyplot as plt
import os
import mne

from epoched_analysis_wrapper import calculate_p_values # SHOULD BE MOVED ELSEWHERE
import paths_and_constants






global DTRND_FIR, SMTH_FIR
DTRND_FIR, SMTH_FIR = None, None


def check_periodicity(x, fs, period_sec):
    win = [3 * fs, 10 * fs]
    xx = x[win[0]:win[-1]]
    xx = xx - xx.mean()
    xx = xx.reshape(7, fs)
    s_ref, s_avg = xx.std(axis=1).mean(), xx.mean(axis=0).std()
    periodicity_score = s_avg / s_ref

    return periodicity_score


def estimate_periodiciy(x, fs, period_sec, ax=None):
    win = [3 * fs, 10 * fs]
    exp_k = 2 * (win[1] - win[0]) / (fs * period_sec)
    # xx = np.copy(x[win[0]:win[-1]])
    global DTRND_FIR, SMTH_FIR
    if SMTH_FIR is None:
        SMTH_FIR = scipy.signal.firls(max(int(fs / 2) + 1, 5),
                                      [0, min(0.4 * period_sec / fs, 0.4), min(0.8 * period_sec / fs, 0.5),
                                       min(4.2 * exp_k / (win[1] - win[0]), 0.8), min(4.5 * exp_k / (win[1] - win[0]), 0.9), 1],
                                      [0, 0, 1, 1, 0, 0])
        fdisp, rdisp = scipy.signal.freqz(SMTH_FIR, 1)
        plt.plot(fdisp, 10 * np.log10(np.real(rdisp * np.conj(rdisp)) + 1e-6))
        plt.show()
    xx = scipy.signal.filtfilt(SMTH_FIR, 1, x)[win[0]:win[-1]]
    # xx -= xx.mean()
    # x = scipy.signal.detrend(x)
    # exp_k = 2 * xx.size / (fs * period_sec)
    xx = (xx - xx.mean())  # * np.hamming(xx.size)
    xx = np.concatenate((np.zeros(int(xx.size / 2)), xx, np.zeros(int(xx.size / 2))))
    xc = np.convolve(xx, xx, mode='same')
    xc = (xc - xc.mean()) * np.hanning(xc.size)
    X = np.abs(np.fft.fft(xc))
    X = X / np.linalg.norm(x)
    X = X[:int(x.size / 2)]
    X = X / X.sum()  # np.linalg.norm(x)

    ave_score = check_periodicity(x, fs, 1)

    mask = np.zeros(X.shape)
    for octv in range(1, 3 + 1):
        mask += np.convolve([1, 1, 1], np.sinc(np.arange(mask.size) - octv * exp_k), mode='same')

    if ax is not None:
        # ax.plot(x, ':b')
        ax.plot(X, 'r', linewidth=2)
        idxs = np.argwhere(mask > 0.25).flatten()
        ax.scatter(idxs, X[idxs], c='r', s=16 * mask[idxs])
        ax.plot(xx[int(xx.size / 4):-int(xx.size / 4)] * X.max() / xx.max(), 'b', linewidth=0.5)
        ax.plot(xc * X.max() / xc.max(), 'k:')
        ax.text(150, 0.6 * X.max(), '{:4.2f}'.format(ave_score))

    return X, (X * mask).sum(), ave_score





def selects_contacts_by_periodicity(contact_list, fs, period_sec=1, show=False):


    #
    # PATCH: add no subset to the contact_list
    for contact in contact_list:
        tmp = contact['first'].find('subset-')
        contact['first_base'] = contact['first'][:tmp] + contact['first'][tmp+len('subset---'):]
        tmp = contact['second'].find('subset-')
        contact['second_base'] = contact['second'][:tmp] + contact['second'][tmp+len('subset---'):]

    data_mat_for_periodicity, mask, _ = read_data_single_two_sessions_single_epoch(contact_list, v_samp_per_sec=fs, active_contacts_only=False, slct=['first_base', 'second_base'], cmprs=False)
    assert data_mat_for_periodicity.ndim == 3
    num_substs, num_contacts, num_samples = data_mat_for_periodicity.shape
    assert num_substs == 2, 'currently only data with two subsets (data_mat.shape[0] == 2) are supported'
    periodicity_mask = np.zeros((2, num_contacts), dtype=bool)

    for i_cntct in range(num_contacts):
        if mask[i_cntct]:
            for i_sbst in range(num_substs):
                _, a, b = estimate_periodiciy(data_mat_for_periodicity[i_sbst, i_cntct], fs=fs, period_sec=period_sec)
                periodicity_mask[i_sbst, i_cntct] = np.sqrt(a * b) > 0.25
                if show:
                    if (i_cntct > 0) and (i_cntct % 50 == 0) and (i_sbst == 0):
                        plt.show()
                    i_ax = (i_cntct * 2 + i_sbst) % 10
                    if (i_ax == 0) and (i_sbst == 0):
                        fig, ax = plt.subplots(5, 4, sharex=False, sharey=False, figsize=(12, 10))
                        clr = ['c', 'm', 'g', 'r']
                    ax.flatten()[2*i_ax].plot(np.arange(num_samples) / fs, data_mat_for_periodicity[i_sbst, i_cntct],
                                            c=clr[i_sbst + 2 * periodicity_mask[i_sbst, i_cntct]], label=str(np.round(100*a, decimals=0)))
                    estimate_periodiciy(data_mat_for_periodicity[i_sbst, i_cntct], fs=fs, period_sec=period_sec, ax=ax.flatten()[2*i_ax+1])
                    if True:# i_sbst == num_substs - 1:
                        ax.flatten()[2*i_ax].grid(True)
                        ax.flatten()[2*i_ax+1].grid(True)
                        ax.flatten()[2*i_ax].set_yticks([])
                        ax.flatten()[2*i_ax+1].set_yticks([])
                        ax.flatten()[2*i_ax].legend()
                        ax.flatten()[2 * i_ax].set_ylabel(str(i_cntct))

    return periodicity_mask




class activation_pca:

    def __init__(self, contact_list=None):

        if contact_list:
            self.fit(contact_list)

    def fit(self, contact_list):

        data, active_contact_mask, _ = read_data_single_two_sessions_single_epoch(contact_list, v_samp_per_sec=V_SAMP_PER_SEC, active_contacts_only=False, esel=0, cmprs=False)
        data = np.expand_dims(data, axis=-1) # axes: (session, contact, time, epoch)
        done, esel = False, 1
        while not done:
            try:
                data_, active_contact_mask_, _ = read_data_single_two_sessions_single_epoch(contact_list,
                                                                                          v_samp_per_sec=V_SAMP_PER_SEC,
                                                                                          active_contacts_only=False,
                                                                                          esel=esel, cmprs=False)
                data = np.concatenate((data, np.expand_dims(data_, axis=-1)), axis=-1)
                active_contact_mask = active_contact_mask * active_contact_mask_
                esel += 1
            except:
                done = True
        data = data[:, active_contact_mask]
        self.active_contact_mask = active_contact_mask

        data = data[:, :, 2*V_SAMP_PER_SEC:-V_SAMP_PER_SEC] # remove pre- and post- seconds
        data1 = data.transpose((0, 3, 1, 2)).reshape(2, data.shape[3], data.shape[1], int(data.shape[2] / V_SAMP_PER_SEC), V_SAMP_PER_SEC)
        #data = data1.reshape(2, data.shape[1], np.prod(data.shape[-2:])) # all digits, all epochs
        #data = data.reshape(2, data.shape[1], int(data.shape[2] / V_SAMP_PER_SEC), V_SAMP_PER_SEC)
        data2 = data1.transpose((0, 1, 3, 2, 4)).reshape(data1.shape[0], data1.shape[1], data1.shape[3], data1.shape[2] * data1.shape[4])
        data3 = data2.transpose((0, 3, 2, 1)).reshape(data2.shape[0], data2.shape[3], np.prod(data2.shape[1:3]))
        self.averages = data3.mean(axis=-1)

        import sklearn.decomposition
        self.tfm0 = sklearn.decomposition.PCA(svd_solver='full')
        self.tfm0.fit(data3[0].T)
        self.tfm1 = sklearn.decomposition.PCA(svd_solver='full')
        self.tfm1.fit(data3[1].T)

        fig, ax = plt.subplots(2, 1)
        ax[0].plot(self.tfm0.mean_, label='mean')
        [ax[0].plot(self.tfm0.components_[i], label=str(i)) for i in range(3)]
        ax[0].grid(True)
        ax[0].legend()
        ax[1].plot(self.tfm1.mean_, label='mean')
        [ax[1].plot(self.tfm1.components_[i], label=str(i)) for i in range(3)]
        ax[1].grid(True)
        ax[1].legend()
        plt.show()


    def get_contact_mask(self):

        return self.active_contact_mask


    def remove_first_componenets(self, data, n=5):

        n_chan, n_sec = data.shape[2], int(data.shape[3] / V_SAMP_PER_SEC)
        data1 = data.reshape(2, data.shape[1], n_chan, n_sec, V_SAMP_PER_SEC).transpose((0, 1, 2, 4, 3))
        data1 = data1.reshape(2, data.shape[1], n_chan * V_SAMP_PER_SEC, n_sec)

        for epoch_id in range(data.shape[1]):
            weights = self.tfm0.transform(data1[0, epoch_id].T)
            weights[:, :n] = 0
            tmp = np.copy(self.tfm0.mean_)  ######
            #self.tfm0.mean_[:] = 0          ######
            data[0, epoch_id] = self.tfm0.inverse_transform(weights).T.reshape(data.shape[2], V_SAMP_PER_SEC, data1.shape[-1]).transpose((0, 2, 1)).reshape(data.shape[2:])
            data1[0, epoch_id] -= np.expand_dims(self.averages[0], axis=1)
            data[0, epoch_id] = data1[0, epoch_id].reshape(data.shape[2], V_SAMP_PER_SEC, data1.shape[-1]).transpose((0, 2, 1)).reshape(data.shape[2:])
            self.tfm0.mean_ = tmp           ######
            weights = self.tfm1.transform(data1[1, epoch_id].T)
            weights[:, :n] = 0
            tmp = np.copy(self.tfm1.mean_)  ######
            #self.tfm1.mean_[:] = 0          ######
            data[1, epoch_id] = self.tfm1.inverse_transform(weights).T.reshape(data.shape[2], V_SAMP_PER_SEC, data1.shape[-1]).reshape(data.shape[2:])
            data1[1, epoch_id] -= np.expand_dims(self.averages[1], axis=1)
            data[1, epoch_id] = data1[1, epoch_id].reshape(data.shape[2], V_SAMP_PER_SEC, data1.shape[-1]).transpose((0, 2, 1)).reshape(data.shape[2:])
            self.tfm1.mean_ = tmp           ######

        return data


def read_evoked_data_two_sessions(contact_list, v_samp_per_sec, esel_list=[]):

    data_mat = np.zeros((2, len(esel_list), len(contact_list), int(v_samp_per_sec * 12)))
    boundries_sec = np.linspace(start=0 - 1, stop=11, num=data_mat.shape[-1] + 1)
    running_first, running_second = ' ', ' '
    valid_contact_mask = np.zeros((len(contact_list), len(esel_list)), dtype=bool)
    for epoch_id, esel in enumerate(esel_list):
        for i_cntct, contact in enumerate(contact_list):
            if (running_first != contact['first'][esel]) or (running_second != contact['second'][esel]):
                running_first, running_second = contact['first'][esel], contact['second'][esel]
                #running_subject_id = subject_ids[contact['subject']]
                first_data = mne.read_evokeds(running_first, verbose=False)[0]
                # first_data.apply_baseline((-0.5, -0.1))
                second_data = mne.read_evokeds(running_second, verbose=False)[0]
                # second_data.apply_baseline((-0.5, -0.1))
                masks = np.zeros((boundries_sec.size - 1, first_data.times.shape[-1]), dtype=bool)
                for i in range(data_mat.shape[-1]):
                    masks[i] = (first_data.times >= boundries_sec[i]) * (first_data.times < boundries_sec[i + 1])
            #
            # now read the data in the requested resolution
            src_idx1 = np.argwhere([contact['name'] == c for c in first_data.ch_names]).squeeze()
            src_idx2 = np.argwhere([contact['name'] == c for c in second_data.ch_names]).squeeze()
            not_bad1 = np.logical_not(np.any([contact['name'] == b for b in first_data.info['bads']]))
            not_bad2 = np.logical_not(np.any([contact['name'] == b for b in second_data.info['bads']]))
            not_bad1 = not_bad1 and np.any(first_data._data[src_idx1] != 0)  # !!!   PATCH    !!!
            not_bad2 = not_bad2 and np.any(second_data._data[src_idx2] != 0)  # !!!   PATCH    !!!
            not_bad1 = not_bad1 and np.abs(first_data._data[src_idx1]).max() < 3  # !!!   PATCH    !!!
            not_bad2 = not_bad2 and np.abs(second_data._data[src_idx2]).max() < 3  # !!!   PATCH    !!!
            if (src_idx1.size == 1) and (src_idx2.size == 1) and not_bad1 and not_bad2:
                id_in_mat = i_cntct
                for i in range(data_mat.shape[-1]):
                    data_mat[0, epoch_id, id_in_mat, i] = (first_data._data[src_idx1] * masks[i]).sum() / masks[i].sum()
                    data_mat[1, epoch_id, id_in_mat, i] = (second_data._data[src_idx1] * masks[i]).sum() / masks[i].sum()
                #id_vector[id_in_mat] = running_subject_id
                valid_contact_mask[i_cntct, epoch_id] = True

    valid_contact_mask = np.prod(valid_contact_mask, axis=1)

    return data_mat, valid_contact_mask


def select_channels_by_correlation1(data_mat, valid_contact_mask, v_samp_per_sec, show=False):

    num_chans = data_mat.shape[2]
    num_sess, num_epochs, seg_len = data_mat.shape[0], data_mat.shape[1], v_samp_per_sec * 8
    assert data_mat.shape[1] >= 6
    assert data_mat.shape[1] % 2 == 0
    xcorr_vec = np.zeros((num_sess, num_chans))
    for i_chan in range(num_chans):
        if valid_contact_mask[i_chan]:
            chan_data = data_mat[:, :, i_chan, 3*v_samp_per_sec:-v_samp_per_sec]
            for i_sess in range(num_sess):
                for i_epoch in range(num_epochs):
                    chan_data[i_sess, i_epoch] -= chan_data[i_sess, i_epoch].mean()
                    chan_data[i_sess, i_epoch] /= np.linalg.norm(chan_data[i_sess, i_epoch])
            c0 = chan_data[:, ::2].reshape(num_sess, int(num_epochs * seg_len / 2))
            c1 = chan_data[:, 1::2].reshape(num_sess, int(num_epochs * seg_len / 2))
            mid = int(num_epochs * seg_len / 2) - 1
            # for i_sess in range(num_sess):
            #     xcorr_vec[i_sess, i_chan] = np.max(np.convolve(c0[i_sess], c1[i_sess])[mid-1:mid+2])
            for i_sess in range(num_sess):
                c0[i_sess] -= c0[i_sess].mean()
                c1[i_sess] -= c1[i_sess].mean()
                xcorr_vec[i_sess, i_chan] = (c0[i_sess] * c1[i_sess]).sum() / (np.linalg.norm(c0[i_sess]) * np.linalg.norm(c1[i_sess]))
            #
            if show and (xcorr_vec[i_chan] > 0.3):
                fig, ax = plt.subplots(num_sess, 3)
                for i_sess in range(num_sess):
                    ax[i_sess, 0].plot(c0[i_sess])
                    ax[i_sess, 1].plot(c1[i_sess])
                    ax[i_sess, 2].plot(np.convolve(c0[i_sess], c1[i_sess]))
                    [ax[i_sess, i].grid(True) for i in range(3)]
                fig.suptitle('cntct {}  mark={:4.2f}'.format(str(i_chan), xcorr_vec[i_chan]))
                plt.show()
            #
    plt.plot(xcorr_vec.T)
    xcorr_vec = np.min(xcorr_vec, axis=0)
    plt.plot(xcorr_vec)
    plt.show()
    thd = np.sort(xcorr_vec)[::-1][100]
    return valid_contact_mask * (xcorr_vec > thd)


def select_channels_by_correlation(data_mat, valid_contact_mask, v_samp_per_sec, show=False):

    cstart, cstop = 1, 11

    num_chans = data_mat.shape[2]
    num_sess, num_epochs, seg_len = data_mat.shape[0], data_mat.shape[1], v_samp_per_sec * (cstop - cstart)
    assert data_mat.shape[1] >= 6
    assert data_mat.shape[1] % 2 == 0
    xcorr_vec = np.zeros((num_sess, num_chans))
    #xcorr_vec = np.zeros((num_sess, num_chans, 10 * v_samp_per_sec * 2 - 1))
    for i_chan in range(num_chans):
        if valid_contact_mask[i_chan]:
            chan_data = data_mat[:, :, i_chan, cstart*v_samp_per_sec:cstop*v_samp_per_sec]
            for i_sess in range(num_sess):
                for i_epoch in range(num_epochs):
                    chan_data[i_sess, i_epoch] -= chan_data[i_sess, i_epoch].mean()
                    chan_data[i_sess, i_epoch] /= np.linalg.norm(chan_data[i_sess, i_epoch])
                for i1 in range(num_epochs - 1):
                    for i2 in range(i1+1, num_epochs):
                        xcorr_vec[i_sess, i_chan] += (chan_data[i_sess, i1] * chan_data[i_sess, i2]).sum()
                        #xcorr_vec[i_sess, i_chan] += np.convolve(chan_data[i_sess, i1, ::-1], chan_data[i_sess, i2])

    xcorr_vec /= num_epochs * (num_epochs - 1) / 2

    xcorr_vec_m = np.min(xcorr_vec, axis=0)
    #xcorr_vec_m = np.min(np.max(xcorr_vec, axis=2), axis=0)

    def calc_xcor_for_chan(data_mat, ch_id, num_sess, num_epochs, v_samp_per_sec, cstart, cstop):

        chan_data = data_mat[:, :, ch_id, cstart*v_samp_per_sec:cstop*v_samp_per_sec]
        avg_cnt = 0
        #xcorr_f = np.zeros((num_sess, 10*v_samp_per_sec*2-1))
        xcorr_f = np.zeros((num_sess, int(chan_data.shape[-1] * 2 -1)))
        for i_sess in range(num_sess):
            for i_epoch in range(num_epochs):
                chan_data[i_sess, i_epoch] -= chan_data[i_sess, i_epoch].mean()
                chan_data[i_sess, i_epoch] /= np.linalg.norm(chan_data[i_sess, i_epoch])
            for i1 in range(num_epochs - 1):
                for i2 in range(i1 + 1, num_epochs):
                    if avg_cnt == 0:
                        xcorr_f[i_sess] = np.convolve(chan_data[i_sess, i1, ::-1], chan_data[i_sess, i2])
                    else:
                        xcorr_f[i_sess] += np.convolve(chan_data[i_sess, i1, ::-1], chan_data[i_sess, i2])
                    avg_cnt += 1
        #xcorr_f = xcorr_f[1] if xcorr_f[0].max() > xcorr_f[1].max() else xcorr_f[0]
        if xcorr_f[0].max() > xcorr_f[1].max():
            xcorr_f = xcorr_f[::-1]

        return xcorr_f / (num_epochs * (num_epochs - 1) / 2)

    if show:
        #plt.plot(xcorr_vec.T)
        #plt.plot(xcorr_vec_m)

        # now show xcorr for 50 best and 50 worst
        show_thd_good = np.sort(xcorr_vec_m)[::-1][40]
        good_sel = np.argwhere(xcorr_vec_m >= show_thd_good).flatten()
        show_thd_bad = np.sort(xcorr_vec_m[valid_contact_mask])[::-1][-40]
        bad_sel = np.argwhere((xcorr_vec_m <= show_thd_bad) * valid_contact_mask).flatten()
        fig_good, ax_good = plt.subplots(5, 8, figsize=(16, 12), sharey=True)
        fig_good.suptitle('best 40 contacts')
        fig_bad, ax_bad = plt.subplots(5, 8, figsize=(16, 12), sharey=True)
        fig_bad.suptitle('worst 40 contacts')
        span = [xcorr_vec_m.min(), xcorr_vec_m.max()]
        for i in range(40):
            ax_good.flatten()[i].grid(True)
            ax_bad.flatten()[i].grid(True)
            ax_good.flatten()[i].set_ylim(span)
            ax_bad.flatten()[i].set_ylim(span)
            good_xc = calc_xcor_for_chan(data_mat=data_mat, ch_id=good_sel[i], num_sess=num_sess, num_epochs=num_epochs, v_samp_per_sec=v_samp_per_sec, cstart=cstart, cstop=cstop)
            bad_xc = calc_xcor_for_chan(data_mat=data_mat, ch_id=bad_sel[i], num_sess=num_sess, num_epochs=num_epochs, v_samp_per_sec=v_samp_per_sec, cstart=cstart, cstop=cstop)
            ax_good.flatten()[i].plot(good_xc[::-1].T)
            ax_bad.flatten()[i].plot(bad_xc[::-1].T)
            ax_good.flatten()[i].set_xticks([0, int(good_xc.shape[1] / 2), good_xc.shape[1]  -1])
            ax_bad.flatten()[i].set_xticks([0, int(bad_xc.shape[1]  / 2), bad_xc.shape[1]  -1])
            ax_good.flatten()[i].set_ylabel(str(good_sel[i]))
            ax_bad.flatten()[i].set_ylabel(str(bad_sel[i]))
            # print('channel {} xc={:4.2f}'.format(good_sel[i], xcorr_vec_m[good_sel[i]]))
            # print('channel {} xc={:4.2f}'.format(bad_sel[i], xcorr_vec_m[bad_sel[i]]))

        plt.show()

    # thd0 = np.sort(xcorr_vec[0])[::-1][100]
    # slct0 = xcorr_vec[0] >= thd0
    # thd1 = np.sort(xcorr_vec[1])[::-1][100]
    # slct1 = xcorr_vec[1] >= thd1
    # return valid_contact_mask * np.logical_or(slct0, slct1)

    thd = np.sort(xcorr_vec_m)[::-1][100]
    return valid_contact_mask * (xcorr_vec_m > thd)

