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

from data_availability_new import data_availability, contact_list_services
from epoched_analysis_wrapper import calculate_p_values # SHOULD BE MOVED ELSEWHERE
from channel_selection import *

import logging
# logging.basicConfig(filename='C:/Users/menas/OneDrive/Desktop/openneuro/temp/generate_rdms_wrapper.log', filemode='w', level=logging.DEBUG)


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
        fig_pc.savefig(os.path.join(os.path.expanduser('~'), 'figs', title + '_pc.pdf'))
        #fig_folded.savefig(os.path.join(os.path.expanduser('~'), 'figs', title + '_folded.pdf'))

    
    if show:
        plt.show()






def read_epoch_files_by_list(epoch_file_list, first_epoch=0, last_epoch=1, boundary_sec=np.arange(start=-1, stop=12, step=1), norm_baseline=[-0.5, -0.05], random_shift=False):
    
    # the returned array has dimensions (epoch, contact, time)
    
    def resample_epoch(data, fs, tscale, boundary_sec):
        
        resampled = np.zeros((data.shape[0], data.shape[1], boundary_sec.size - 1))
        for i, (start, stop) in enumerate(zip(boundary_sec[:-1], boundary_sec[1:])):
            mask = (tscale >= start) * (tscale < stop)
            resampled[:, :, i] = data[:, :, mask].mean(axis=-1)
        
        return resampled
    
        
    for i_subject in range(epoch_file_list.shape[0]):
        subject_data = epoch_file_list.iloc[i_subject]
        fname = subject_data['filename']
        contacts = subject_data['contacts']
        mne_obj = mne.read_epochs(fname, verbose=False)
        fs = mne_obj.info['sfreq']
        tscale = mne_obj.times
        cmask = [name in contacts for name in mne_obj.ch_names]
        subject_signals = mne_obj.get_data()[first_epoch:last_epoch, cmask]
        nmask = (tscale >= norm_baseline[0]) * (tscale <= norm_baseline[-1])
        #
        #
        if random_shift:
            tol = tscale[-1] - boundary_sec[-1]
            tshift = np.random.uniform(low=0, high=tol)
        else:
            tshift = 0
        resampled = resample_epoch(subject_signals, fs, tscale, boundary_sec + tshift)
        #
        if i_subject == 0:
            data = np.copy(resampled)
            #nmask = (tscale >= norm_baseline[0]) * (tscale <= norm_baseline[-1])
            #norms = subject_signals[:, :, nmask].std(axis=-1)
            norms = np.linalg.norm(subject_signals[:, :, nmask], axis=-1) / np.sqrt(nmask.sum())
        else:
            data = np.concatenate((data, np.copy(resampled)), axis=1)
            norms = np.concatenate((norms, np.linalg.norm(subject_signals[:, :, nmask], axis=-1) / np.sqrt(nmask.sum())), axis=1)
        
    # normalize
    nominal_nf = 1 / np.median(norms)
    max_nf = 3 * nominal_nf
    nf = np.minimum(1 / norms, max_nf)
    data = np.repeat(nf[:, :, np.newaxis], data.shape[-1], axis=2) * data
        

    return data



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




def do_analysis_for_two_epoch_sets(contact_list, subject_ids, esel0, esel1,
                                   V_SAMP_PER_SEC, SHOW_TIME_PER_CONTACT, ACTIVE_CONTACTS_ONLY,
                                   CORR_WINDOW_SEC, AUTO_OR_CROSS_ACTIVATION,
                                   CONTACT_SPLIT, PROCESS_QUADS, tfm=None, SHOW=True, ccorr_mode='p'):


    data_mat, active_contact_mask, _ = read_data_single_two_sessions_single_epoch(contact_list, subject_ids, v_samp_per_sec=V_SAMP_PER_SEC, active_contacts_only=ACTIVE_CONTACTS_ONLY, esel=esel0, cmprs=False)
    data_mat2, active_contact_mask2, _ = read_data_single_two_sessions_single_epoch(contact_list, subject_ids, v_samp_per_sec=V_SAMP_PER_SEC, active_contacts_only=ACTIVE_CONTACTS_ONLY, esel=esel1, cmprs=False)
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

    SELECT_CONTACTS_BY_PERIODICITY = 0
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
    return fig


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
    
    return fig




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


