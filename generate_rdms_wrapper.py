import pickle

import numpy as np
import matplotlib.pyplot as plt
import os
import mne
import scipy.signal
import seaborn as sns
import tqdm

from data_availability import data_availability
from epoched_analysis_wrapper import calculate_p_values # SHOULD BE MOVED ELSEWHERE



def consistant_random_grouping(data, num_groups=2, pindex=2, axis=0, padding=False):

    # if padding is True, the output vectors sizes will be number of contacts, the "missing" contacts are not omitted but zeroed
    n = data.shape[axis]
    grp_size = int(n / num_groups)
    dim_perm = np.arange(data.ndim)
    dim_perm[0], dim_perm[axis] = axis, 0
    data = data.transpose(dim_perm) # bring the axis of grouping to the 0
    store_seed = np.random.seed()
    np.random.seed(pindex)
    p = np.random.permutation(n)
    np.random.seed(store_seed)
    groups, i0 = [], 0
    for i_grp in range(num_groups):
        if padding:
            groups.append(np.zeros(data.shape))
            groups[-1][p[i0:i0+grp_size]] = data[p[i0:i0+grp_size]]
            groups[-1] = groups[-1].transpose(dim_perm)
        else:
            groups.append(data[p[i0:i0+grp_size]].transpose(dim_perm))
        i0 += grp_size

    return groups




def pierson(x, y):

    x1, y1 = x - x.mean(), y - y.mean()
    return (x1 * y1).sum() / (np.linalg.norm(x1) * np.linalg.norm(y1) + 1e-16)

def calc_rdm(data, rdm_size, pre_ignore, delta_time_smple):

    # input data should be of size (2, #contacts, data_bins)
    assert data.ndim == 3
    if data.shape[0] == 1:
        src0, src1 = 0, 0 # "auto" correlation, diagonal should be 1
    elif data.shape[0] == 2:
        src0, src1 = 0, 1 # "cross" correlation, diagonal ususaly < 1
    else:
        assert False
    rdm = np.zeros((rdm_size, rdm_size))  # {random group, time(1), time(2)}
    for t1 in range(rdm_size):
        for t2 in range(rdm_size):
            sig1 = data[src0, :, pre_ignore + t1 * delta_time_smple:pre_ignore + (t1 + 1) * delta_time_smple].flatten()
            sig2 = data[src1, :, pre_ignore + t2 * delta_time_smple:pre_ignore + (t2 + 1) * delta_time_smple].flatten()
            rdm[t1, t2] = pierson(sig1, sig2)

    return rdm



def visualize_rdms(rdms, title='', dst_idx=' ', show_bars=True, show_hists=True, show_hmaps=True, show=True):

    num_splits = rdms.shape[0]
    rdm_size = rdms.shape[-1]

    # generating the correlation bars
    if show_bars:
        ylow, yhigh = min(-0.01, max(-0.1, np.floor(np.quantile(rdms, 0.05) * 10) / 10)), np.ceil(np.quantile(rdms, 0.95) * 10) / 10
        fig_bars, ax_bars = plt.subplots(rdm_size, 1, figsize=(6, 10))
        fig_bars.suptitle(title)
        for t1 in range(rdm_size):
            ax_bars[t1].grid(True)
            ax_bars[t1].set_yticks(np.arange(start=-1, stop=1, step=0.1))
            ax_bars[t1].set_ylim([ylow, yhigh])
            ax_bars[t1].set_xlim([-1.5, rdm_size - 1.5])
            ax_bars[t1].set_ylabel(str(t1-1))
            ax_bars[t1].yaxis.set_tick_params(labelleft=False)
            ax_bars[t1].plot([-2, rdm_size-1], [0, 0], c='k', linewidth=2)
            ax_bars[t1].plot([-2, rdm_size-1], [0.2, 0.2], c='b', linewidth=1)
            ax_bars[t1].plot([-2, rdm_size-1], [0.4, 0.4], c='c', linewidth=1)
            ax_bars[t1].plot([-2, rdm_size-1], [0.6, 0.6], c='y', linewidth=1)
            ax_bars[t1].plot([-2, rdm_size-1], [0.8, 0.8], c='m', linewidth=1)
            for t2 in range(rdm_size):
                ax_bars[t1].bar(t2-1, rdms[:, t1, t2].mean(), width=0.25, color='k' if t1==t2 else 'b')
                ax_bars[t1].errorbar(t2-1, rdms[:, t1, t2].mean(), yerr=3*rdms[:, t1, t2].std(), elinewidth=2.5, ecolor='r')
        #plt.show()

    # generating the correlation histograms
    bin_boundaries = np.linspace(start=rdms.min(), stop=rdms.max(), num=36 if data_mat.shape[1] < 600 else 72)
    bins = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    hist_0_0, hist_0_n, hist_1_1, hist_1_n, hist_n_n, hist_n_m = \
        np.zeros(bins.size), np.zeros(bins.size), np.zeros(bins.size), np.zeros(bins.size), np.zeros(bins.size), np.zeros(bins.size)

    if show_hists:
        fig_husts, ax_hists = plt.subplots(1, 1)
        ax_hists.set_title(title)
        for t1 in range(rdm_size):
            for t2 in range(t1, rdm_size):
                h1 = np.histogram(rdms[:, t1, t2], bins=bin_boundaries, density=True)[0]
                h2 = np.histogram(rdms[:, t2, t1], bins=bin_boundaries, density=True)[0]
                h = (h1 + h2) / 2
                if (t1 > 1) and (t2 > 1) and (t1 == t2):
                    c, w = 'r', 1
                    hist_n_n += h
                if (t1 > 1) and (t2 > 1) and (t1 != t2):
                    c, w = 'k', 1
                    hist_n_m += h
                if (min(t1, t2) == 1) and (t1 != t2):
                    c, w = 'b', 1
                    hist_1_n += h
                if (t1 == 1) and (t2 == 1):
                    c, w = 'g', 3
                    hist_1_1 += h
                if (min(t1, t2) == 0) and (t1 != t2):
                    c, w = 'c', 1
                    hist_0_n += h
                if (t1 == 0) and (t2 == 0):
                    c, w = 'm', 2
                    hist_0_0 += h
                ax_hists.plot(bins, h, c=c, linewidth=w)
        #plt.show()

        fig_havg, ax_havg = plt.subplots(1, 1)
        ax_havg.plot(bins, hist_0_0 / hist_0_0.sum(), c='m', linewidth=2, label='baseline / baseline')
        ax_havg.plot(bins, hist_0_n / hist_0_n.sum(), c='c', linewidth=1, label='baseline / digit')
        ax_havg.plot(bins, hist_1_n / hist_1_n.sum(), c='b', linewidth=1, label='1 / 2-9')
        ax_havg.plot(bins, hist_1_1 / hist_1_1.sum(), c='g', linewidth=3, label='1 / 1')
        ax_havg.plot(bins, hist_n_m / hist_n_m.sum(), c='k', linewidth=1, label='2-9 / other 2-9')
        ax_havg.plot(bins, hist_n_n / hist_n_n.sum(), c='r', linewidth=3, label='2-9 / 2-9')
        ax_havg.legend()
        ax_havg.grid(True)
        fig_havg.suptitle('{}\n{} contacts, {} split permutations'.format(title, dst_idx, num_splits))
        #plt.show()

    if show_hmaps:
        fig_pc, ax_pc = plt.subplots(1, 1, figsize=(6, 6))
        fig_folded, ax_folded = plt.subplots(1, 1, figsize=(6, 6))
        fig_pc.suptitle('{}\n{} contacts, {} split permutations'.format(title, dst_idx, num_splits))
        fig_folded.suptitle('{}\n{} contacts, {} split permutations'.format(title, dst_idx, num_splits))
        sns.heatmap(np.round(rdms.mean(axis=0), decimals=2), vmin=-1, vmax=1, ax=ax_pc, annot=True, square=True, cbar=False)
        havg = rdms.mean(axis=0)
        havg = (havg + havg.T) / 2
        for i in range(1, rdm_size):
            havg[i, :i] = 0
        sns.heatmap(np.round(havg, decimals=2), vmin=-1, vmax=1, ax=ax_folded, annot=True, square=True, cbar=False)

        if show:
            plt.show()


def read_data_single_two_sessions_single_epoch(contact_list, v_samp_per_sec, active_contacts_only=False, slct=['first', 'second'], cmprs=True):

    id_vector = np.zeros(len(contact_list), dtype=int)
    # resolution_sec = 0.5
    # data_mat = np.zeros((2, len(contact_list), int(V_SAMP_PER_SEC * 11)))
    # boundries_sec = np.linspace(start=0-1, stop=11-1, num=data_mat.shape[-1] + 1)
    data_mat = np.zeros((2, len(contact_list), int(v_samp_per_sec * 12)))
    boundries_sec = np.linspace(start=0 - 1, stop=11, num=data_mat.shape[-1] + 1)
    running_first, running_second = ' ', ' '
    dst_idx = 0
    active_contact_list = []
    active_contact_mask = np.zeros(len(contact_list), dtype=bool)
    for i_cntct, contact in enumerate(contact_list):
        if (running_first != contact[slct[0]]) or (running_second != contact[slct[1]]):
            running_first, running_second = contact[slct[0]], contact[slct[1]]
            running_subject_id = subject_ids[contact['subject']]
            first_data = mne.read_evokeds(running_first, verbose=False)[0]
            # first_data.apply_baseline((-0.5, -0.1))
            first_p_vals, first_increase, _ = calculate_p_values(first_data.copy(), show=False)
            second_data = mne.read_evokeds(running_second, verbose=False)[0]
            # second_data.apply_baseline((-0.5, -0.1))
            second_p_vals, second_increase, _ = calculate_p_values(second_data.copy(), show=False)
            masks = np.zeros((boundries_sec.size - 1, first_data.times.shape[-1]), dtype=bool)
            for i in range(data_mat.shape[-1]):
                masks[i] = (first_data.times >= boundries_sec[i]) * (first_data.times < boundries_sec[i + 1])
            contact_activity_1 = (first_p_vals < 0.05) * first_increase
            contact_activity_2 = (second_p_vals < 0.05) * second_increase
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
        contact_is_active = (contact_activity_1[src_idx1] and contact_activity_2[src_idx2])
        if ((contact_is_active or (not active_contacts_only)) and
                (src_idx1.size == 1) and (src_idx2.size == 1) and not_bad1 and not_bad2):
            id_in_mat = dst_idx if cmprs else i_cntct
            for i in range(data_mat.shape[-1]):
                data_mat[0, id_in_mat, i] = (first_data._data[src_idx1] * masks[i]).sum() / masks[i].sum()
                data_mat[1, id_in_mat, i] = (second_data._data[src_idx1] * masks[i]).sum() / masks[i].sum()
            id_vector[id_in_mat] = running_subject_id
            active_contact_list.append({'subject': contact['subject'], 'name': contact['name']})
            active_contact_mask[i_cntct] = True
            dst_idx += 1

    if cmprs:
        data_mat = data_mat[:, :dst_idx]

    return data_mat, active_contact_mask, dict({'active_contact_list': active_contact_list, 'p_values_first': first_p_vals, 'p_values_second': second_p_vals})




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
        SMTH_FIR = scipy.signal.firls(int(fs / 2) + 1,
                                      [0, 0.4 * period_sec / fs, 0.8 * period_sec / fs,
                                       4.2 * exp_k / (win[1] - win[0]), 4.5 * exp_k / (win[1] - win[0]), 1],
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


def relative_codes(cmat, first=0, last=-1, remove_diag=True, normalize=False):

    # converting the correlation matrix ("rdm") into relational coding
    # remove diagonal: if True, the diagonal is first removed
    # remove columns: if default values are overridden, it will be used to select which columns in the original matrix participating in the codes

    assert cmat.shape[0] == cmat.shape[1]
    num_cw = cmat.shape[0]

    code = np.copy(cmat)
    if normalize:
        for i_cw in range(num_cw):
            code[i_cw] = code[i_cw] / code[i_cw, i_cw]

    if remove_diag:
        code -= np.diag(np.diag(code))

    return code[:, first:last]




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



if __name__ == '__main__':

    V_SAMP_PER_SEC = 4
    CORR_WINDOW_SEC = 1
    SHOW_TIME_PER_CONTACT = False
    #NUM_SPLITS = 500#1250
    ACTIVE_CONTACTS_ONLY = False
    AUTO_OR_CROSS_ACTIVATION = "CROSS"  # "AUTO": generate session rdm from single epoch set (diagonal = 1); "CROSS": cross-correlate two epoch sets
    EPOCH_SUBSET = 0#None#
    OTHER_EPOCH_SUBSET = 1# if AUTO_OR_CROSS_ACTIVATION == "CROSS" else EPOCH_SUBSET# None#for making self-session rdms
    MIN_TGAP, MAX_TGAP = 72, 96#60, 160#
    SELECT_CONTACTS_BY_PERIODICITY = False
    CONTACT_SPLIT = None # None: use all, 0: even contacts only, 1: odd contacts only
    PROCESS_QUADS = False
    if PROCESS_QUADS:
        V_SAMP_PER_SEC = V_SAMP_PER_SEC * 4
        CORR_WINDOW_SEC = CORR_WINDOW_SEC / 4

    data_availability_obj = data_availability()
    contact_list = data_availability_obj.get_get_contacts_for_2_session_gap_epoch_splits(min_timegap_hrs=MIN_TGAP, max_timegap_hrs=MAX_TGAP,
                                                                                         event_type='CNTDWN', sub_event_type='CNTDWN',
                                                                                         epoch_subset=EPOCH_SUBSET, second_epoch_subset=OTHER_EPOCH_SUBSET)

    # make a dictionary og integer subject id's
    subject_ids = dict()
    id = 0
    for contact in contact_list:
        if not contact['subject'] in subject_ids.keys():
            subject_ids[contact['subject']] = id
            id += 1

    data_mat, active_contact_mask, _ = read_data_single_two_sessions_single_epoch(contact_list, v_samp_per_sec=V_SAMP_PER_SEC, active_contacts_only=ACTIVE_CONTACTS_ONLY, cmprs=False)
    data_mat2, active_contact_mask2, _ = read_data_single_two_sessions_single_epoch(contact_list, v_samp_per_sec=V_SAMP_PER_SEC, active_contacts_only=ACTIVE_CONTACTS_ONLY, slct=['first2', 'second2'], cmprs=False)
    keep = active_contact_mask * active_contact_mask2
    contact_list = [contact_list[i] for i in np.argwhere(keep).flatten()]
    data_mat = np.concatenate((np.expand_dims(data_mat, axis=1), np.expand_dims(data_mat2, axis=1)), axis=1)[:, :, keep]

    if SELECT_CONTACTS_BY_PERIODICITY:
        pmask = selects_contacts_by_periodicity(contact_list=contact_list, fs=16, period_sec=1, show=False)
        #pmask = np.logical_not(pmask)
        keep = pmask[0] * pmask[1]
        data_mat = data_mat[:, :, keep]

    # THE DIMENSIONS OF THE DATA MAT ARE: (session, epoch_group, contact, time_bin)

    # with open('E:/epoched/contact_sel', 'wb') as fd:
    #     pickle.dump({'active_contact_list': active_contact_list}, fd)

    if CONTACT_SPLIT is not None:
        data_mat = data_mat[:, :, CONTACT_SPLIT::2]

    if AUTO_OR_CROSS_ACTIVATION == 'AUTO':
        data_mat[:, 1] = data_mat[:, 0]

    if SHOW_TIME_PER_CONTACT:
        for i in range(data_mat.shape[2]):
            if i % 20 == 0:
                figc, axc = plt.subplots(4, 5, figsize=(12, 8), sharex=True, sharey=True)
                figc.suptitle('autocorrelation')
                fig, ax = plt.subplots(4, 5, figsize=(12, 8), sharex=True, sharey=True)
            i_ax = i % 20
            #
            tmp1, a1 = estimate_periodiciy(data_mat[0, 0, i], V_SAMP_PER_SEC)
            tmp2, a2 = estimate_periodiciy(data_mat[0, 1, i], V_SAMP_PER_SEC)
            selbld0 = (max(a1, a2) > 0.2) or (a1 + a2 > 0.32)
            tmp3, a3 = estimate_periodiciy(data_mat[1, 0, i], V_SAMP_PER_SEC)
            tmp4, a4 = estimate_periodiciy(data_mat[1, 1, i], V_SAMP_PER_SEC)
            selbld1 = (max(a3, a4) > 0.2) or (a3 + a4 > 0.32)
            #
            ax.flatten()[i_ax].plot(data_mat[0, 0, i], linewidth=0.5+selbld0)
            ax.flatten()[i_ax].plot(data_mat[0, 1, i], linewidth=0.5+selbld0)
            ax.flatten()[i_ax].plot(data_mat[1, 0, i], linewidth=0.5+selbld1)
            ax.flatten()[i_ax].plot(data_mat[1, 1, i], linewidth=0.5+selbld1)
            ax.flatten()[i_ax].grid(True)
            ax.flatten()[i_ax].set_ylabel(str(i))
            ax.flatten()[i_ax].set_ylim((0.5, 2))
            #
            axc.flatten()[i_ax].plot(tmp1, linewidth=0.5+selbld0)
            axc.flatten()[i_ax].plot(tmp2, linewidth=0.5+selbld0)
            axc.flatten()[i_ax].plot(tmp3, linewidth=0.5+selbld1)
            axc.flatten()[i_ax].plot(tmp4, linewidth=0.5+selbld1)
            axc.flatten()[i_ax].set_ylabel('{:3.0f}   {:3.0f}   {:3.0f}   {:3.0f}'.format(100 * a1, 100 * a2, 100 * a3, 100 * a4))
            axc.flatten()[i_ax].grid(True)
            #axc.flatten()[i_ax].set_ylabel(str(i))
            #axc.flatten()[i_ax].set_ylim((0.5, 2))
            #
            if i % 200 == 199:
                plt.show()


    # show the signals
    fig, ax = plt.subplots(4, 1, sharex=True, sharey=True)
    for i in range(2):
        ax[i].plot(data_mat[0, i])
        ax[i].set_ylabel('early\nsession')
        ax[2+i].plot(data_mat[1, i])
        ax[2+i].set_ylabel('succeeding\nsession')
    plt.suptitle(str(data_mat.shape[2]) + '   contacts')
    plt.show()

    # NOW DO THE RDM ANALYSIS
    pre_ignore = 0#V_SAMP_PER_SEC # 1 second in the begining
    delta_time_smple = int(V_SAMP_PER_SEC * CORR_WINDOW_SEC)

    if PROCESS_QUADS:
        # make avg 1-sec
        first_digit_to_avg, last_digit_to_avg = 3, 8#2, 7
        first_smple_to_avg = (1 + first_digit_to_avg) * V_SAMP_PER_SEC
        last_sample_to_avg = (1 + last_digit_to_avg) * V_SAMP_PER_SEC
        num_avg = last_digit_to_avg - first_digit_to_avg
        revised_data_mat = np.zeros((2, 2, data_mat.shape[2], delta_time_smple * 4))
        # make the 1-sec avg
        for i_cntct in range(data_mat.shape[2]):
            src_smp = first_smple_to_avg
            for i_rpt in range(num_avg):
                for i_quad in range(4):
                    revised_data_mat[:, :, i_cntct, (i_quad * delta_time_smple):((i_quad + 1) * delta_time_smple)] += data_mat[:, :, i_cntct, src_smp:src_smp + delta_time_smple] / num_avg
                    src_smp += delta_time_smple
        #     i_ax = i_cntct % 16
        #     if i_ax == 0:
        #         fig, ax = plt.subplots(4, 4)
        #     ax.flatten()[i_ax].plot(revised_data_mat[0, 0, i_cntct])
        #     ax.flatten()[i_ax].plot(revised_data_mat[0, 1, i_cntct])
        #     ax.flatten()[i_ax].plot(revised_data_mat[1, 0, i_cntct])
        #     ax.flatten()[i_ax].plot(revised_data_mat[1, 1, i_cntct])
        #     i_ax += 1
        # plt.show()
        data_mat = revised_data_mat


    rdm_size = int((data_mat.shape[-1] - pre_ignore) / delta_time_smple)

    rdm0 = calc_rdm(data_mat[0], rdm_size, pre_ignore, delta_time_smple)
    rdm1 = calc_rdm(data_mat[1], rdm_size, pre_ignore, delta_time_smple)
    visualize_rdms(np.expand_dims(rdm0, axis=0), title=' early session', show_hists=False, show=False)
    visualize_rdms(np.expand_dims(rdm1, axis=0), title=' subsequent session', show_hists=False, show=False)

    # cross_session activity correlation
    for i_sbst in range(2 if AUTO_OR_CROSS_ACTIVATION=='CROSS' else 1):
        csac = calc_rdm(data_mat[:, 0], rdm_size, pre_ignore, delta_time_smple)
        visualize_rdms(np.expand_dims(csac, axis=0), title='cross-session correlation of Activity vectors (sbst {})'.format(i_sbst+1), show_hists=False, show_bars=False, show=False)



    if  PROCESS_QUADS:
        R0 = relative_codes(rdm0, first=0, last=4, remove_diag=True, normalize=False)
        R1 = relative_codes(rdm1, first=0, last=4, remove_diag=True, normalize=False)
    else:
        R0 = relative_codes(rdm0, first=1, remove_diag=True, normalize=False)
        R1 = relative_codes(rdm1, first=1, remove_diag=True, normalize=False)

    # for i in range(rdm_size):
    #     print('\n', i)
    #     print(rdm0[i])
    #     print(R0[i])

    rep_pcors = np.zeros((rdm_size, rdm_size))
    for digit_1 in range(rdm_size):
        for difit_2 in range(rdm_size):
            v1, v2 = R0[digit_1], R1[difit_2]
            rep_pcors[digit_1, difit_2] = pierson (v1, v2) #(v1 * v2).sum() / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-18)
    visualize_rdms(np.expand_dims(rep_pcors, axis=0), title='full vector relative representation correlation', show_hists=False, show_bars=False, show=True)





    # EXAMPLE ON GENERATING SPLITS
    # # print('collecting statistics')
    # # for split_id in tqdm.tqdm(range(1, NUM_SPLITS)):
    # #     split_data = consistant_random_grouping(data_mat, pindex=split_id, axis=1)
    # #     split_data = np.array(split_data) # axes: {contact group, session, contact (within group), time)
    # #     # correlation between corresponding seconds is different sessions
    # #     rdms = generate_rdm(split_data, rdm_size, pre_ignore, delta_time_smple, ses=[0, 1])
    # #     rdm_results[split_id * 2] = rdms[0]
    # #     rdm_results[split_id * 2 + 1] = rdms[1]
    # #     #
    # #     SHOW_EACH_RDM = False
    # #     if SHOW_EACH_RDM:
    # #         fig, ax = plt.subplots(1, 2, figsize=(10, 6), num='split rdm')
    # #         sns.heatmap(np.round(rdms[0], decimals=2), vmin=-1, vmax=1, ax=ax[0], annot=True, square=True, cbar=False)
    # #         ax[0].set_title('random contact sel {} / {}'.format(split_id, 1))
    # #         sns.heatmap(np.round(rdms[1], decimals=2), vmin=-1, vmax=1, ax=ax[1], annot=True, square=True, cbar=False)
    # #         ax[1].set_title('random contact sel {} / {}'.format(split_id, 2))
    # #         plt.show(block=False)
    # #         plt.pause(0.1)
    # # visualize_rdms(rdm_results)



