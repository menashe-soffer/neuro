import pickle

import numpy as np
import matplotlib.pyplot as plt
import os
import mne
import scipy.signal
import seaborn as sns
import tqdm
import copy
import sklearn

from data_availability import data_availability
from epoched_analysis_wrapper import calculate_p_values # SHOULD BE MOVED ELSEWHERE
from channel_selection import *



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




def pierson(x, y, remove_nans=False, mode='p'):

    assert mode in ['p', 'c', 'd'] # [ierson, cosine, distance]
    if remove_nans:
        rmv_cols = np.concatenate((np.argwhere(np.isnan(x)).flatten(), np.argwhere(np.isnan(y)).flatten()))
        if rmv_cols.size > 0:
            x, y = np.copy(x), np.copy(y)
            rmv_cols = np.sort(np.unique(rmv_cols))
            while rmv_cols.size > 0:
                x = np.concatenate((x[:rmv_cols[0]], x[rmv_cols[0]+1:]))
                y = np.concatenate((y[:rmv_cols[0]], y[rmv_cols[0]+1:]))
                rmv_cols = rmv_cols[1:] - 1

    x1, y1 = x - x.mean() * (mode == 'p'), y - y.mean() * (mode == 'p')
    if mode == 'd':
        return 1 - 2 * np.linalg.norm(x1 - y1)  / (np.linalg.norm(x1) + np.linalg.norm(y1) + 1e-16)
    else:
        return (x1 * y1).sum() / (np.linalg.norm(x1) * np.linalg.norm(y1) + 1e-16)



def calc_rdm(data, rdm_size, pre_ignore, delta_time_smple, corr_mode='p'):

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



def visualize_rdms(rdms, title='', dst_idx=' ', show_bars=True, show_hists=True, show_hmaps=True, show=True, ovrd_bar_scale=None, ovrd_heat_scale=None):

    num_splits = rdms.shape[0]
    rdm_size = rdms.shape[-1]

    # generating the correlation bars
    if show_bars:
        if ovrd_bar_scale:
            ylow, yhigh =ovrd_bar_scale[0], ovrd_bar_scale[1]
        else:
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
    bin_boundaries = np.linspace(start=rdms.min(), stop=rdms.max(), num=36)# if data_mat.shape[1] < 600 else 72)
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
        if ovrd_heat_scale:
            vmin, vmax = ovrd_heat_scale[0], ovrd_heat_scale[1]
        else:
            vmin, vmax = -1, 1
        fig_pc, ax_pc = plt.subplots(1, 1, figsize=(6, 6))
        fig_folded, ax_folded = plt.subplots(1, 1, figsize=(6, 6))
        fig_pc.suptitle('{}\n{} contacts, {} split permutations'.format(title, dst_idx, num_splits))
        fig_folded.suptitle('{}\n{} contacts, {} split permutations'.format(title, dst_idx, num_splits))
        xticks, yticks = np.arange(-1, rdms.shape[1] - 1), np.arange(-1, rdms.shape[2] - 1)
        sns.heatmap(np.round(rdms.mean(axis=0), decimals=2), vmin=vmin, vmax=vmax, ax=ax_pc, annot=True, square=True, cbar=False, xticklabels=xticks, yticklabels=yticks)
        havg = rdms.mean(axis=0)
        havg = (havg + havg.T) / 2
        for i in range(1, rdm_size):
            havg[i, :i] = 0
        sns.heatmap(np.round(havg, decimals=2), vmin=vmin, vmax=vmax, ax=ax_folded, annot=True, square=True, cbar=False, xticklabels=xticks, yticklabels=yticks)



    if show:
        plt.show()


def read_data_single_two_sessions_single_epoch(contact_list, v_samp_per_sec, active_contacts_only=False, esel=0, cmprs=True):

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
        if (running_first != contact['first'][esel]) or (running_second != contact['second'][esel]):
            running_first, running_second = contact['first'][esel], contact['second'][esel]
            running_subject_id = subject_ids[contact['subject']]
            first_data = mne.read_evokeds(running_first, verbose=False)[0]
            # first_data.apply_baseline((-0.5, -0.1))
            first_p_vals, first_increase, _ = calculate_p_values(first_data.copy(), show=False, pre_intvl=[-0.95, -0.1], post_intval=[0.1+1*2, 0.6+1*8.5])
            second_data = mne.read_evokeds(running_second, verbose=False)[0]
            # second_data.apply_baseline((-0.5, -0.1))
            second_p_vals, second_increase, _ = calculate_p_values(second_data.copy(), show=False, pre_intvl=[-0.95, -0.1], post_intval=[0.1+1*2, 0.6+1*8.5])
            masks = np.zeros((boundries_sec.size - 1, first_data.times.shape[-1]), dtype=bool)
            for i in range(data_mat.shape[-1]):
                masks[i] = (first_data.times >= boundries_sec[i]) * (first_data.times < boundries_sec[i + 1])
            contact_activity_1 = (first_p_vals < 0.05) #* first_increase
            contact_activity_2 = (second_p_vals < 0.05) #* second_increase
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
        contact_is_active = (contact_activity_1[src_idx1] or contact_activity_2[src_idx2])
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
        for i in range(num_cw):
            code[i, i] = np.nan

    return code[:, first:last]




def do_analysis_for_two_epoch_sets(contact_list, esel0, esel1,
                                   V_SAMP_PER_SEC, SHOW_TIME_PER_CONTACT, ACTIVE_CONTACTS_ONLY,
                                   CORR_WINDOW_SEC, AUTO_OR_CROSS_ACTIVATION,
                                   CONTACT_SPLIT, PROCESS_QUADS, tfm=None, SHOW=True, ccorr_mode='p'):


    data_mat, active_contact_mask, _ = read_data_single_two_sessions_single_epoch(contact_list, v_samp_per_sec=V_SAMP_PER_SEC, active_contacts_only=ACTIVE_CONTACTS_ONLY, esel=esel0, cmprs=False)
    data_mat2, active_contact_mask2, _ = read_data_single_two_sessions_single_epoch(contact_list, v_samp_per_sec=V_SAMP_PER_SEC, active_contacts_only=ACTIVE_CONTACTS_ONLY, esel=esel1, cmprs=False)
    keep = active_contact_mask * active_contact_mask2 if not tfm else tfm.get_contact_mask()
    contact_list = [contact_list[i] for i in np.argwhere(keep).flatten()]
    data_mat = np.concatenate((np.expand_dims(data_mat, axis=1), np.expand_dims(data_mat2, axis=1)), axis=1)[:, :, keep]

    #data_mat -= np.tile(np.expand_dims(data_mat[:, :, :, 1:-1].mean(axis=-1), axis=-1), 12)
    #
    if SHOW and tfm:
        data_mat_show = np.copy(data_mat)
        data_mat_show = tfm.remove_first_componenets(data=data_mat_show, n=0)
        for i_ch in range(data_mat_show.shape[2]):
            if i_ch % 20 == 0:
                fig, ax = plt.subplots(4, 5, figsize=(16, 12))
            ax.flatten()[i_ch % 20].plot(data_mat[0, 0, i_ch])
            ax.flatten()[i_ch % 20].plot(data_mat_show[0, 0, i_ch] + data_mat[0, 0, i_ch].mean() - data_mat_show[0, 0, i_ch].mean())
            if i_ch % 400 == 399:
                plt.show()
        plt.show()
    #
    #data_mat = tfm.remove_first_componenets(data=data_mat, n=0) if tfm else data_mat

    # #
    # fig, ax = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
    # data_mat1 = np.copy(data_mat)
    # for i1 in range(2):
    #     for i2 in range(2):
    #         for i3 in range(data_mat1.shape[2]):
    #             data_mat1[i1, i2, i3] -=  data_mat1[i1, i2, i3].mean()
    #             data_mat1[i1, i2, i3] /= data_mat1[i1, i2, i3].std()
    #         sns.heatmap(data_mat1[i1, i2], ax=ax[i1, i2], vmin=-1.6, vmax=1.6)
    # plt.show()
    # #

    # #
    # fig, ax = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
    # data_mat1 = np.copy(data_mat)
    # for i1 in range(2):
    #     for i2 in range(2):
    #         for i3 in range(data_mat1.shape[2]):
    #             data_mat1[i1, i2, i3] -=  data_mat1[i1, i2, i3].mean()
    #             data_mat1[i1, i2, i3] /= np.linalg.norm(data_mat1[i1, i2, i3])
    #         c = data_mat1[i1, i2] @  data_mat1[i1, i2].T
    #         sns.heatmap(c, ax=ax[i1, i2], vmin=-0.25, vmax=0.25)
    # plt.show()
    # #

    if SELECT_CONTACTS_BY_PERIODICITY != 0:
        pmask = selects_contacts_by_periodicity(contact_list=contact_list, fs=16, period_sec=1, show=False)
        #pmask = np.logical_not(pmask)
        keep = pmask[0] * pmask[1]
        keep = np.logical_not(keep) if SELECT_CONTACTS_BY_PERIODICITY < 0 else keep
        data_mat = data_mat[:, :, keep]

    # THE DIMENSIONS OF THE DATA MAT ARE: (session, epoch_group, contact, time_bin)

    # with open('E:/epoched/contact_sel', 'wb') as fd:
    #     pickle.dump({'active_contact_list': active_contact_list}, fd)

    if CONTACT_SPLIT is not None:
        data_mat = data_mat[:, :, CONTACT_SPLIT::2]

    if AUTO_OR_CROSS_ACTIVATION == 'AUTO':
        data_mat[:, 1] = data_mat[:, 0]

    if SHOW_TIME_PER_CONTACT and SHOW:
        for i in range(data_mat.shape[2]):
            if i % 20 == 0:
                figc, axc = plt.subplots(4, 5, figsize=(12, 8), sharex=True, sharey=True)
                figc.suptitle('autocorrelation')
                fig, ax = plt.subplots(4, 5, figsize=(12, 8), sharex=True, sharey=True)
            i_ax = i % 20
            #
            tmp1, a1, _ = estimate_periodiciy(data_mat[0, 0, i], V_SAMP_PER_SEC, 1)
            tmp2, a2, _ = estimate_periodiciy(data_mat[0, 1, i], V_SAMP_PER_SEC, 1)
            selbld0 = (max(a1, a2) > 0.2) or (a1 + a2 > 0.32)
            tmp3, a3, _ = estimate_periodiciy(data_mat[1, 0, i], V_SAMP_PER_SEC, 1)
            tmp4, a4, _ = estimate_periodiciy(data_mat[1, 1, i], V_SAMP_PER_SEC, 1)
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
    if SHOW:
        fig, ax = plt.subplots(4, 1, sharex=True, sharey=True)
        for i in range(2):
            ax[i].plot(data_mat[0, i, :, :])
            ax[i].set_ylabel('early\nsession')
            ax[2+i].plot(data_mat[1, i, :, :])
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
    if SHOW:
        visualize_rdms(np.expand_dims(rdm0, axis=0), title=' early session', show_hists=False, show=False)
        visualize_rdms(np.expand_dims(rdm1, axis=0), title=' subsequent session', show_hists=False, show=False)

    # # cross_session activity correlation
    # for i_sbst in range(2 if AUTO_OR_CROSS_ACTIVATION=='CROSS' else 1):
    #     if i_sbst == 0:
    #         csac = calc_rdm(data_mat[:, i_sbst], rdm_size, pre_ignore, delta_time_smple)
    #     else:
    #         csac = (csac + calc_rdm(data_mat[:, i_sbst], rdm_size, pre_ignore, delta_time_smple)) / 2
    #     if SHOW:
    #         visualize_rdms(np.expand_dims(csac, axis=0), title='cross-session correlation of Activity vectors (sbst {})'.format(i_sbst+1), show_hists=False, show_bars=False, show=False)

    # cross-session activity correlation, by average of the two epochs
    csac = calc_rdm(data_mat.mean(axis=1), rdm_size, pre_ignore, delta_time_smple, corr_mode=ccorr_mode)

    if  PROCESS_QUADS:
        R0 = relative_codes(rdm0, first=0, last=4, remove_diag=True, normalize=False)
        R1 = relative_codes(rdm1, first=0, last=4, remove_diag=True, normalize=False)
    else:
        R0 = relative_codes(rdm0, first=1, remove_diag=True, normalize=False)
        R1 = relative_codes(rdm1, first=1, remove_diag=True, normalize=False)


    return rdm_size, rdm0, rdm1, csac, R0, R1, contact_list


def show_corr_diagonals(csac_list, rep_pcors_list, show=False):


    # show the diagonals
    if csac_list.ndim == 2:
        csac_list = np.expand_dims(csac_list, axis=0)
    if rep_pcors_list.ndim == 2:
        rep_pcors_list = np.expand_dims(rep_pcors_list, axis=0)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    c_act_avg, c_act_sem = np.mean(csac_list, axis=0), np.std(csac_list, axis=0) / np.sqrt(csac_list.shape[0])
    c_rep_avg, c_rep_sem = np.mean(rep_pcors_list, axis=0), np.std(rep_pcors_list, axis=0) / np.sqrt(csac_list.shape[0])
    #c_rep_avg, c_rep_std = rep_pcors, np.zeros(rep_pcors.shape)
    ax.bar(np.arange(c_act_avg.shape[0]) - 0.2 - 1, np.diag(c_act_avg), width=0.2, label='activations')
    ax.bar(np.arange(c_act_avg.shape[0]) - 0.2 - 1, 2 * np.diag(c_act_sem), bottom=np.diag(c_act_avg) - np.diag(c_act_sem), width=0.05, color='k')
    ax.bar(np.arange(c_rep_avg.shape[0]) + 0.2 - 1, np.diag(c_rep_avg), width=0.2, label='relational codes')
    ax.bar(np.arange(c_rep_avg.shape[0]) + 0.2 - 1, 2 * np.diag(c_rep_sem), bottom=np.diag(c_rep_avg) - np.diag(c_rep_sem), width=0.05, color='k')
    ax.set_xticks(np.arange(c_rep_avg.shape[0]) - 1)
    ax.grid(True)
    ax.set_ylim([-1.1, 1.1])
    ax.legend()
    if show:
        plt.show()
    #


def show_relational_codes(R0_list, R1_list, show=False):

    if R0_list.ndim == 2:
        R0_list = np.expand_dims(R0_list, axis=0)
    if R1_list.ndim == 2:
        R1_list = np.expand_dims(R1_list, axis=0)
    avg_size, num_codes, code_size = R0_list.shape
    assert R0_list.shape == R1_list.shape

    fig, ax = plt.subplots(num_codes, 2, figsize=(10, 10))
    fig.suptitle('relational codes')
    R0_avg, R0_sem = np.mean(R0_list, axis=0), np.std(R0_list, axis=0) / np.sqrt(avg_size)
    R1_avg, R1_sem = np.mean(R1_list, axis=0), np.std(R1_list, axis=0) / np.sqrt(avg_size)
    # lower = min(R0_list[np.logical_not(np.isnan(R0_list))].min(), R1_list[np.logical_not(np.isnan(R1_list))].min())
    # upper = max(R0_list[np.logical_not(np.isnan(R0_list))].max(), R1_list[np.logical_not(np.isnan(R1_list))].max())
    lower = min((R0_avg - R0_sem)[np.logical_not(np.isnan(R0_avg))].min(), (R1_avg - R1_sem)[np.logical_not(np.isnan(R1_avg))].min())
    upper = max((R0_avg + R1_sem)[np.logical_not(np.isnan(R0_avg))].max(), (R1_avg + R1_sem)[np.logical_not(np.isnan(R1_avg))].max())
    ystep = 0.1
    lower, upper = np.floor(lower / ystep) * ystep, np.ceil(upper / ystep) * ystep
    for i in range(num_codes):
        for ii in range(2):
            ax[i, ii].set_ylim([lower-0.05, upper+0.05])
            ax[i, ii].set_xlim([-2, code_size])
            ax[i, ii].axis(False)
            ax[i, ii].plot([-1, code_size], [0, 0], c='k', linewidth=2)

            for y in np.linspace(start=lower, stop=upper, num=int(1 + (upper - lower) / ystep)):
                if y != 0:
                    ax[i, ii].plot([-1, code_size], [y, y], linewidth=1, color=(1-y, 0.5, y) if y>0 else (0, 0, 0))
        ax[i, 0].bar(np.arange(code_size), R0_avg[i], width=0.4)
        ax[i, 1].bar(np.arange(code_size), R1_avg[i], width=0.4)
        ax[i, 0].bar(np.arange(code_size), 2 * R0_sem[i], bottom=R0_avg[i] - R0_sem[i], width=0.2)
        ax[i, 1].bar(np.arange(code_size), 2 * R1_sem[i], bottom=R1_avg[i] - R1_sem[i], width=0.2)
        ax[i, 0].text(-3, 0, str(i-1))

    if show:
        plt.show()




def show_region_distribution(contact_list, title=None):

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    sel_regions = [c['location'][0]['region'] for c in contact_list] + [c['location'][1]['region'] for c in contact_list]
    regions = np.unique(sel_regions)
    counter = np.zeros(len(regions), dtype=int)
    for region in sel_regions:
        region_id = np.argwhere([r == region for r in regions]).squeeze()
        counter[region_id] += 1
    reord = np.argsort(counter)
    counter = counter[reord]
    regions = [regions[i] for i in reord]
    ax.bar(regions, 100 * counter / counter.sum())
    #ax.grid(True)
    ax.set_xticks(np.arange(len(regions)))  # Explicitly set ticks if needed (though ax.bar often sets them)
    ax.set_xticklabels(regions, rotation=70, ha='right')
    ax.set_ylabel('contact percentage')
    fig.tight_layout()
    if title:
        fig.suptitle(title)

    return fig





if __name__ == '__main__':

    V_SAMP_PER_SEC = 1
    CORR_WINDOW_SEC = 1
    SHOW_TIME_PER_CONTACT = False
    ACTIVE_CONTACTS_ONLY = False
    AUTO_OR_CROSS_ACTIVATION = "CROSS"  # "AUTO": generate session rdm from single epoch set (diagonal = 1); "CROSS": cross-correlate two epoch sets
    EPOCH_SUBSET = 'e0-e5'#
    OTHER_EPOCH_SUBSET = 'e6-e11' if AUTO_OR_CROSS_ACTIVATION == "CROSS" else EPOCH_SUBSET# None#for making self-session rdms
    AVG_MANY_EPOCHS = []#['e0-e0', 'e1-e1', 'e2-e2', 'e3-e3', 'e4-e4', 'e5-e5']
    MIN_TGAP, MAX_TGAP = 72, 96#60, 160#10, 500#
    SELECT_CONTACTS_BY_PERIODICITY = 0 # 0: ignore periodicity, 1: select periodic contacts, -1: select NON-periodic contacts
    CONTACT_SPLIT = None # None: use all, 0: even contacts only, 1: odd contacts only
    PROCESS_QUADS = False
    event_type = 'CNTDWN' # one of: 'CNTDWN', 'RECALL', 'DSTRCT', 'REST'
    CROSS_SESSION_CMODE = 'p'
    #
    if PROCESS_QUADS:
        V_SAMP_PER_SEC = V_SAMP_PER_SEC * 4
        CORR_WINDOW_SEC = CORR_WINDOW_SEC / 4
    #assert (AUTO_OR_CROSS_ACTIVATION == "CROSS") or (not AVG_MANY_EPOCHS)
    #
    SELECT_CONTACTS_BY_CORR = False
    V_SAMP_FOR_SLCT = 4
    SAVE_CONTACT_LIST = False
    USE_CONTACT_SELECTION_FROM_FILE = True
    CONTACT_SELECTION_FILE_NAME = 'C:/Users/menas/OneDrive/Desktop/openneuro/temp/contact_list_4d_4d_sel'



    data_availability_obj = data_availability()
    epoch_subsets =  [EPOCH_SUBSET, OTHER_EPOCH_SUBSET] if not AVG_MANY_EPOCHS else AVG_MANY_EPOCHS
    contact_list = data_availability_obj.get_get_contacts_for_2_session_gap_epoch_splits(min_timegap_hrs=MIN_TGAP, max_timegap_hrs=MAX_TGAP,
                                                                                         event_type=event_type, sub_event_type=event_type,
                                                                                         epoch_subsets=epoch_subsets)

    # #
    # temp_list = ['sub-R1065J', 'sub-R1083J', 'sub-R1111M', 'sub-R1112M', 'sub-R1118N', 'sub-R1161E', 'sub-R1168T', 'sub-R1172E', 'sub-R1196N',
    #              'sub-R1283T', 'sub-R1308T', 'sub-R1315T', 'sub-R1325C', 'sub-R1336T', 'sub-R1338T', 'sub-R1355T', 'sub-R1542J']
    # revised = []
    # for c in contact_list:
    #     if c['subject'] in temp_list:
    #         revised.append(c)
    # print(len(contact_list), len(revised))
    # #assert False
    # contact_list = revised
    # #



    # make a dictionary og integer subject id's
    subject_ids = dict()
    id = 0
    for contact in contact_list:
        if not contact['subject'] in subject_ids.keys():
            subject_ids[contact['subject']] = id
            id += 1


    # for i in range(rdm_size):
    #     print('\n', i)
    #     print(rdm0[i])
    #     print(R0[i])

    #
    # SELECT ACTIVE CONTACTS BASED ON ACTIVITY OF TOTAL
    if ACTIVE_CONTACTS_ONLY:
        # generate temporary contact list for averages (on which we calulate activity)
        tmp_contact_list = copy.deepcopy(contact_list)
        for contact in tmp_contact_list:
            tmp_first = contact['first'][0].replace('-bipolar_*--CNTDWN', '-bipolar_-CNTDWN')
            tmp_second = contact['first'][0].replace('-bipolar_*--CNTDWN', '-bipolar_-CNTDWN')
            contact['first'], contact['second'] = [tmp_first], [tmp_second]
        # now
        _, active_contact_mask, _ = read_data_single_two_sessions_single_epoch(tmp_contact_list, v_samp_per_sec=V_SAMP_PER_SEC, active_contacts_only=ACTIVE_CONTACTS_ONLY, esel=0, cmprs=False)
        _, active_contact_mask2, _ = read_data_single_two_sessions_single_epoch(contact_list, v_samp_per_sec=V_SAMP_PER_SEC, active_contacts_only=ACTIVE_CONTACTS_ONLY, esel=1, cmprs=False)
        keep = active_contact_mask * active_contact_mask2
        # the selected contact
        print('SELECTING {} ACTIVE CONTACTS OUT OF {}'.format(keep.sum(), len(contact_list)))
        contact_list = [contact_list[i] for i in np.argwhere(keep).flatten()]
    #

    projector = None#activation_pca(contact_list=contact_list)#

    if SELECT_CONTACTS_BY_CORR:
        data_mat, valid_contact_mask = read_evoked_data_two_sessions(contact_list, V_SAMP_FOR_SLCT, esel_list=np.arange(len(epoch_subsets)))
        mask = select_channels_by_correlation(data_mat, valid_contact_mask, V_SAMP_FOR_SLCT, show=True)
        contact_list = [contact_list[i] for i in np.argwhere(mask).flatten()]

    # if SAVE_CONTACT_LIST:
    #     with open(CONTACT_SELECTION_FILE_NAME, 'wb') as fd:
    #         pickle.dump(contact_list, fd)

    if USE_CONTACT_SELECTION_FROM_FILE:
        with open(CONTACT_SELECTION_FILE_NAME, 'rb') as fd:
            contact_list_ref = pickle.load(fd)
        combined_list = []
        for cid, c in enumerate(contact_list):
            for rid, r in enumerate(contact_list_ref):
                #print(c, r)
                if (c['subject'] == r['subject']) and (c['name'] == r['name']) and (len(c['first']) >= len(AVG_MANY_EPOCHS)) and(len(c['second']) >= len(AVG_MANY_EPOCHS)):
                    print(c['subject'], c['name'], r['subject'], r['name'], cid, rid, len(combined_list))
                    combined_list.append(c)
        contact_list = combined_list[:len(contact_list_ref)]#[:100]

    from temp_plots import remove_double_contacts
    contact_list = remove_double_contacts(contact_list)


    pair_cnt = 0
    contact_list_ = []
    csac_list, R0_list, R1_list, rdm0_list, rdm1_list = [], [], [], [], []
    for i_sbst0 in range(len(epoch_subsets) - (1 if AUTO_OR_CROSS_ACTIVATION=='CROSS' else 0)):
        for i_sbst1 in range(i_sbst0 + 1, len(epoch_subsets)) if AUTO_OR_CROSS_ACTIVATION=='CROSS' else [i_sbst0] :

            rdm_size_, rdm0_, rdm1_, csac_, R0_, R1_, contact_list__ = do_analysis_for_two_epoch_sets(contact_list, i_sbst0, i_sbst1,
                                               V_SAMP_PER_SEC, SHOW_TIME_PER_CONTACT, False, CORR_WINDOW_SEC,
                                               AUTO_OR_CROSS_ACTIVATION, CONTACT_SPLIT, PROCESS_QUADS, tfm=projector, SHOW=(i_sbst0 + i_sbst1 == 111), ccorr_mode=CROSS_SESSION_CMODE)
            contact_list_ = contact_list_ + contact_list__
            if pair_cnt == 0:
                rdm_size, rdm0, rdm1, csac, R0, R1 = rdm_size_, rdm0_, rdm1_, csac_, R0_, R1_
            else:
                rdm0 += rdm0_
                rdm1 += rdm1_
                csac += csac_
                R0 += R0_
                R1 += R1_
            pair_cnt += 1
            print('pair no. {},  {} {}'.format(pair_cnt, epoch_subsets[i_sbst0], epoch_subsets[i_sbst1]))
            csac_list.append(csac_)
            R0_list.append(R0_)
            R1_list.append(R1_)
            rdm0_list.append(rdm0_)
            rdm1_list.append(rdm1_)

    rdm0 /= pair_cnt
    rdm1 /= pair_cnt
    csac /= pair_cnt
    R0 /= pair_cnt
    R1 /= pair_cnt



if SAVE_CONTACT_LIST:
    with open(CONTACT_SELECTION_FILE_NAME, 'wb') as fd:
        pickle.dump(contact_list_, fd)

show_region_distribution(contact_list, title='{} contacts , delta=T = {} hrs to {} hrs'.format(len(contact_list), MIN_TGAP, MAX_TGAP))
#plt.show()


# VISUALIZE RESULTS FOR PLAIN AVERAGING
DISPLAY_PLAIN_AVERAGING = len(AVG_MANY_EPOCHS) == 0
if DISPLAY_PLAIN_AVERAGING:
    visualize_rdms(np.expand_dims(rdm0, axis=0), title=' early session ', show_hists=False, show_bars=False, show=False, ovrd_bar_scale=[-0.1, 0.2], ovrd_heat_scale=[-0.1, 0.2])
    visualize_rdms(np.expand_dims(rdm1, axis=0), title=' subsequent session', show_hists=False, show_bars=False, show=False, ovrd_bar_scale=[-0.1, 0.2], ovrd_heat_scale=[-0.1, 0.2])
    #for i_sbst in range(range(2 if AUTO_OR_CROSS_ACTIVATION=='CROSS' else 1):
    visualize_rdms(np.expand_dims(csac, axis=0),
                   #title='cross-session correlation of Activity vectors (sbst {})'.format(i_sbst + 1),
                   title='cross-session correlation of Activity vectors among sessions',
                   show_hists=False, show_bars=False, show=False)

    show_relational_codes(R0, R1, show=False)
    rep_pcors = np.zeros((rdm_size, rdm_size))
    for digit_1 in range(rdm_size):
        for digit_2 in range(rdm_size):
            v1, v2 = R0[digit_1], R1[digit_2]
            #rep_pcors[digit_1, digit_2] = pierson (v1, v2, remove_nans=True) #(v1 * v2).sum() / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-18)
            rep_pcors[digit_1, digit_2] = pierson(v1, v2, remove_nans=True, mode=CROSS_SESSION_CMODE)  # (v1 * v2).sum() / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-18)
    visualize_rdms(np.expand_dims(rep_pcors, axis=0), title='full vector relative representation correlation', show_hists=False, show_bars=False, show=False)
    show_corr_diagonals(csac, rep_pcors, show=True)

    with open(os.path.join(os.path.dirname(CONTACT_SELECTION_FILE_NAME), 'cs_corrs_1'), 'wb') as fd:
        pickle.dump(dict({'csac': csac, 'rep_pcorr': rep_pcors}), fd)


    #
    # redu everything with lists
DISPLAY_5_3_2 = len(AVG_MANY_EPOCHS) >= 6
if DISPLAY_5_3_2:
    csac_list, R0_list, R1_list = np.array(csac_list), np.array(R0_list), np.array(R1_list)
    # # re-disply activation correlations with error bars
    # visualize_rdms(np.expand_dims(np.mean(csac_list, axis=0), axis=0),
    #                title='cross-session correlation of Activity vectors recalculated', show_hists=False, show_bars=False, show=False)
    # visualize_rdms(np.expand_dims(np.std(csac_list, axis=0), axis=0),
    #                title='cross-session correlation of Activity vectors recalculated, STDEV', show_hists=False, show_bars=False, show=False)
    # # no generate rep_pcoers for seperate reps
    # partial averagings
    if AUTO_OR_CROSS_ACTIVATION == 'CROSS':
        sbgrps = np.array((1, 10, 15, 2, 8, 14, 3, 9, 11, 4, 7, 12, 5, 6, 13)).reshape(5, 3)
        sub_cnt = 5
    if AUTO_OR_CROSS_ACTIVATION == 'AUTO':
        sbgrps = np.array((1, 2, 3, 4, 5, 6)).reshape(3, 2)
        sub_cnt = 3
    Cact_list_ = np.zeros((sub_cnt, csac_list.shape[1], csac_list.shape[2]))
    R0_list_, R1_list_ = np.zeros((sub_cnt, R0_list.shape[1], R0_list.shape[2])), np.zeros((sub_cnt, R0_list.shape[1], R0_list.shape[2]))
    for i_sub in range(sub_cnt):
        Cact_list_[i_sub] = csac_list[sbgrps[i_sub] - 1].mean(axis=0)
        R0_list_[i_sub] = R0_list[sbgrps[i_sub] - 1].mean(axis=0)
        R1_list_[i_sub] = R1_list[sbgrps[i_sub] - 1].mean(axis=0)
    csac_list, R0_list, R1_list = Cact_list_, R0_list_, R1_list_

    rep_pcors_list = np.zeros((sub_cnt, rdm_size, rdm_size))
    for i_pair in range(sub_cnt):
        for digit_1 in range(rdm_size):
            for digit_2 in range(rdm_size):
                v1, v2 = R0_list[i_pair, digit_1], R1_list[i_pair, digit_2]
                #rep_pcors_list[i_pair, digit_1, digit_2] = pierson (v1, v2, remove_nans=True) #(v1 * v2).sum() / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-18)
                rep_pcors_list[i_pair, digit_1, digit_2] = pierson(v1, v2, remove_nans=True, mode=CROSS_SESSION_CMODE)  # (v1 * v2).sum() / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-18)
    visualize_rdms(np.expand_dims(np.mean(csac_list, axis=0), axis=0),
                   title='ACTIVATION CROSS-SESSION AVG CORR', show_hists=False, show_bars=False, show=False, ovrd_bar_scale=[-0.1, 0.2], ovrd_heat_scale=[-0.2, 0.3])
    visualize_rdms(np.expand_dims(np.mean(rep_pcors_list, axis=0), axis=0),
                   title='RELATIVE REPRESANTATION AVG CORR', show_hists=False, show_bars=False, show=False, ovrd_bar_scale=[-0,1, 0.2], ovrd_heat_scale=[-1, 1])
    # visualize_rdms(np.expand_dims(np.std(rep_pcors_list, axis=0), axis=0),
    #                title='RELATIVE REPRESANTATION STDEV CORR', show_hists=False, show_bars=False, show=False, ovrd_bar_scale=[-0,1, 0.2], ovrd_heat_scale=[-0.2, 0.3])
    show_relational_codes(R0_list, R1_list, show=False)
    show_corr_diagonals(csac_list, rep_pcors_list, show=True)


    with open(os.path.join(os.path.dirname(CONTACT_SELECTION_FILE_NAME), 'cs_corrs_5_3'), 'wb') as fd:
        pickle.dump(dict({'csac': csac_list, 'rep_pcorr': rep_pcors_list}), fd)


    #
    rep_pcors = rep_pcors_list.mean(axis=0)



    # # show the diagonals
    # fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    # c_act_avg, c_act_std = np.mean(csac_list, axis=0), np.std(csac_list, axis=0)
    # #c_rep_avg, c_rep_std = np.mean(rep_pcors_list, axis=0), np.std(rep_pcors_list, axis=0)
    # c_rep_avg, c_rep_std = rep_pcors, np.zeros(rep_pcors.shape)
    # ax.bar(np.arange(c_act_avg.shape[0]) - 0.2, np.diag(c_act_avg), width=0.2, label='activations')
    # ax.bar(np.arange(c_act_avg.shape[0]) - 0.2, 2 * np.diag(c_act_std), bottom=np.diag(c_act_avg) - np.diag(c_act_std), width=0.05, color='k')
    # ax.bar(np.arange(c_rep_avg.shape[0]) + 0.2, np.diag(c_rep_avg), width=0.2, label='relational codes')
    # ax.bar(np.arange(c_rep_avg.shape[0]) + 0.2, 2 * np.diag(c_rep_std), bottom=np.diag(c_rep_avg) - np.diag(c_rep_std), width=0.05, color='k')
    # ax.grid(True)
    # ax.set_ylim([-1.1, 1.1])
    # ax.legend()
    # plt.show()
    # #



    # statistics
    tmp0, tmp1, tmpcs, tmprel = rdm0[1:-1, 1:-1], rdm1[1:-1, 1:-1], csac[1:-1, 1:-1], rep_pcors[1:-1, 1:-1]
    off_diag = np.concatenate((tmp0[~np.eye(tmp0.shape[0],dtype=bool)], tmp1[~np.eye(tmp1.shape[0],dtype=bool)]))
    on_diag = np.concatenate((np.diag(tmp0), np.diag(tmp1)))
    print('\n\t\t\t\t\t\tdiagonal\t\t\toff diag')
    print('within session:\t\t {:4.2f} (dev {:4.2f}) \t{:4.2f} (dev {:4.2f}) '.format(on_diag.mean(), on_diag.std(), off_diag.mean(), off_diag.std()))
    off_diag = tmpcs[~np.eye(tmpcs.shape[0],dtype=bool)]
    on_diag = np.diag(csac)
    print('across sessions:\t {:4.2f} (dev {:4.2f}) \t{:4.2f} (dev {:4.2f}) '.format(on_diag.mean(), on_diag.std(), off_diag.mean(), off_diag.std()))
    off_diag = tmprel[~np.eye(tmprel.shape[0],dtype=bool)]
    on_diag = np.diag(tmprel)
    print('relative :\t\t\t {:4.2f} (dev {:4.2f}) \t{:4.2f} (dev {:4.2f}) '.format(on_diag.mean(), on_diag.std(), off_diag.mean(), off_diag.std()))


    # # show the diagonals
    # fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    # ax.bar(np.arange(csac.shape[0]) - 0.2, np.diag(csac), width=0.2, label='activations')
    # ax.bar(np.arange(rep_pcors.shape[0]) + 0.2, np.diag(rep_pcors), width=0.2, label='relational codes')
    # ax.grid(True)
    # ax.set_ylim([-1.1, 1.1])
    # ax.legend()
    # if not PROCESS_QUADS:
    #     ax.set_xticks(np.arange(12), ['pre\ncnt', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'post\ncnt'])
    # fig.suptitle('correlations')
    # plt.show()
    # #
    # fname = 'C:/Users/menas/OneDrive/Desktop/openneuro/tmpres-sbst'# + str(EPOCH_SUBSET)
    # with open(fname, 'wb') as fd:
    #     pickle.dump({'csac': csac, 'rep_pcors': rep_pcors}, fd)
    # #


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



