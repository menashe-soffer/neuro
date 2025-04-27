import pickle

import numpy as np
import matplotlib.pyplot as plt
import os
import mne
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


# def generate_rdm(split_data, rdm_size, pre_ignore, delta_time_smple, ses):
#     num_grps = split_data.shape[0]
#     assert num_grps <= 2
#     rdms = np.zeros((num_grps, rdm_size, rdm_size))  # {random group, time(1), time(2)}
#     for t1 in range(rdm_size):
#         for t2 in range(rdm_size):
#             for grp in range(num_grps):
#                 sig1 = split_data[grp, ses[0], :, pre_ignore + t1 * delta_time_smple:pre_ignore + (t1 + 1) * delta_time_smple].flatten()
#                 sig2 = split_data[grp, ses[1], :, pre_ignore + t2 * delta_time_smple:pre_ignore + (t2 + 1) * delta_time_smple].flatten()
#                 sig1 -= sig1.mean()
#                 sig2 -= sig2.mean()
#                 if True:  # t1 <= t2:#
#                     rdms[grp, t1, t2] = (sig1 * sig2).sum() / (np.linalg.norm(sig1) * np.linalg.norm(sig2) + 1e-16)
#                 # if t1 == t2:
#                 #     plt.plot(sig1)
#                 #     plt.plot(sig2)
#                 #     plt.title('t1: {}   t2: {}'.format(t1, t2))
#                 #     plt.show()
#
#     return rdms


def pierson(x, y):

    x1, y1 = x - x.mean(), y - y.mean()
    return (x1 * y1).sum() / (np.linalg.norm(x1) * np.linalg.norm(y1) + 1e-16)

def calc_rdm(data, rdm_size, pre_ignore, delta_time_smple):

    # input data should be of size (2, #contacts, data_bins)
    assert data.ndim == 3
    assert data.shape[0] == 2
    rdm = np.zeros((rdm_size, rdm_size))  # {random group, time(1), time(2)}
    for t1 in range(rdm_size):
        for t2 in range(rdm_size):
            sig1 = data[0, :, pre_ignore + t1 * delta_time_smple:pre_ignore + (t1 + 1) * delta_time_smple].flatten()
            sig2 = data[1, :, pre_ignore + t2 * delta_time_smple:pre_ignore + (t2 + 1) * delta_time_smple].flatten()
            # sig1 -= sig1.mean()
            # sig2 -= sig2.mean()
            # rdm[t1, t2] = (sig1 * sig2).sum() / (np.linalg.norm(sig1) * np.linalg.norm(sig2) + 1e-16)
            rdm[t1, t2] = pierson(sig1, sig2)

    return rdm



def visualize_rdms(rdms, title='', dst_idx=' '):

    num_splits = rdms.shape[0]

    # generating the correlation bars
    ylow, yhigh = max(-0.1, np.floor(np.quantile(rdms, 0.05) * 10) / 10), np.ceil(np.quantile(rdms, 0.95) * 10) / 10
    fig_bars, ax_bars = plt.subplots(12, 1, figsize=(6, 10))
    for t1 in range(12):
        ax_bars[t1].grid(True)
        ax_bars[t1].set_yticks(np.arange(start=-1, stop=1, step=0.1))
        ax_bars[t1].set_ylim([ylow, yhigh])
        ax_bars[t1].set_xlim([-1.5, 10.5])
        ax_bars[t1].set_ylabel(str(t1-1))
        ax_bars[t1].yaxis.set_tick_params(labelleft=False)
        ax_bars[t1].plot([-2, 11], [0, 0], c='k', linewidth=2)
        for t2 in range(12):
            ax_bars[t1].bar(t2-1, rdms[:, t1, t2].mean(), width=0.25, color='k' if t1==t2 else 'b')
            ax_bars[t1].errorbar(t2-1, rdms[:, t1, t2].mean(), yerr=3*rdms[:, t1, t2].std(), elinewidth=2.5, ecolor='r')
    #plt.show()

    # generating the correlation histograms
    bin_boundaries = np.linspace(start=rdms.min(), stop=rdms.max(), num=36 if data_mat.shape[1] < 600 else 72)
    bins = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    hist_0_0, hist_0_n, hist_1_1, hist_1_n, hist_n_n, hist_n_m = \
        np.zeros(bins.size), np.zeros(bins.size), np.zeros(bins.size), np.zeros(bins.size), np.zeros(bins.size), np.zeros(bins.size)

    fig_husts, ax_hists = plt.subplots(1, 1)
    for t1 in range(12):
        for t2 in range(t1, 12):
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

    fig_pc, ax_pc = plt.subplots(1, 1, figsize=(6, 6))
    fig_folded, ax_folded = plt.subplots(1, 1, figsize=(6, 6))
    fig_pc.suptitle('{}\n{} contacts, {} split permutations'.format(title, dst_idx, num_splits))
    fig_folded.suptitle('{}\n{} contacts, {} split permutations'.format(title, dst_idx, num_splits))
    sns.heatmap(np.round(rdms.mean(axis=0), decimals=2), vmin=-1, vmax=1, ax=ax_pc, annot=True, square=True, cbar=False)
    havg = rdms.mean(axis=0)
    havg = (havg + havg.T) / 2
    for i in range(1, 12):
        havg[i, :i] = 0
    sns.heatmap(np.round(havg, decimals=2), vmin=-1, vmax=1, ax=ax_folded, annot=True, square=True, cbar=False)
    plt.show()


def read_data_single_two_sessions_single_epoch(contact_list, active_contacts_only=False, slct=['first', 'second']):

    id_vector = np.zeros(len(contact_list), dtype=int)
    # resolution_sec = 0.5
    # data_mat = np.zeros((2, len(contact_list), int(V_SAMP_PER_SEC * 11)))
    # boundries_sec = np.linspace(start=0-1, stop=11-1, num=data_mat.shape[-1] + 1)
    data_mat = np.zeros((2, len(contact_list), int(V_SAMP_PER_SEC * 12)))
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
            for i in range(data_mat.shape[-1]):
                data_mat[0, dst_idx, i] = (first_data._data[src_idx1] * masks[i]).sum() / masks[i].sum()
                data_mat[1, dst_idx, i] = (second_data._data[src_idx1] * masks[i]).sum() / masks[i].sum()
            id_vector[dst_idx] = running_subject_id
            active_contact_list.append({'subject': contact['subject'], 'name': contact['name']})
            active_contact_mask[i_cntct] = True
            dst_idx += 1

    data_mat = data_mat[:, :dst_idx]

    return data_mat, active_contact_mask, dict({'active_contact_list': active_contact_list, 'p_values_first': first_p_vals, 'p_values_second': second_p_vals})




if __name__ == '__main__':

    V_SAMP_PER_SEC = 4
    CORR_WINDOW_SEC = 1
    SHOW_TIME_PER_CONTACT = False
    NUM_SPLITS = 500#1250
    ACTIVE_CONTACTS_ONLY = False
    EPOCH_SUBSET = 0#None#
    OTHER_EPOCH_SUBSET = 1 # None#for making self-session rdms
    MIN_TGAP, MAX_TGAP = 60, 160# 72, 96

    data_availability_obj = data_availability()
    _, contact_list = data_availability_obj.get_contacts_for_2_session_gap(min_timegap_hrs=MIN_TGAP, max_timegap_hrs=MAX_TGAP,
                                                                           event_type='cntdwn', sub_event_type='CNTDWN', epoch_subset=EPOCH_SUBSET)
    if OTHER_EPOCH_SUBSET is not None:
        _, contact_list2 = data_availability_obj.get_contacts_for_2_session_gap(min_timegap_hrs=MIN_TGAP, max_timegap_hrs=MAX_TGAP,
                                                                                event_type='cntdwn', sub_event_type='CNTDWN', epoch_subset=OTHER_EPOCH_SUBSET)
        # TBD find intersect of two lists
        combined_list = []
        rplc_pattrn_1, rplc_pattern2 = '_subset-' + str(EPOCH_SUBSET), '_subset-' + str(OTHER_EPOCH_SUBSET)
        for i_cntct, contact in tqdm.tqdm(enumerate(contact_list)):
            # check if the same contact apears in the second list
            exist2 = False
            for i_cntct2, contact2 in enumerate(contact_list2):
                if contact['subject'] == contact2['subject']:
                    if contact['name'] == contact2['name']:
                        ok1 = contact['first'].replace(rplc_pattrn_1, rplc_pattern2) == contact2['first']
                        ok2 = contact['second'].replace(rplc_pattrn_1, rplc_pattern2) == contact2['second']
                        if ok1 and ok2:
                            contact['first2'] = contact2['first']
                            contact['second2'] = contact2['second']
                            combined_list.append(contact)
                            break
        contact_list = combined_list

    # make a dictionary og integer subject id's
    subject_ids = dict()
    id = 0
    for contact in contact_list:
        if not contact['subject'] in subject_ids.keys():
            subject_ids[contact['subject']] = id
            id += 1

    data_mat, active_contact_mask, _ = read_data_single_two_sessions_single_epoch(contact_list, active_contacts_only=ACTIVE_CONTACTS_ONLY)
    data_mat2, active_contact_mask2, _ = read_data_single_two_sessions_single_epoch(contact_list, active_contacts_only=ACTIVE_CONTACTS_ONLY, slct=['first2', 'second2'])
    keep1, keep2 = np.zeros(data_mat.shape[1], dtype=bool), np.zeros(data_mat2.shape[1], dtype=bool)
    idx1, idx2 = 0, 0
    for (a1, a2) in zip(active_contact_mask, active_contact_mask2):
        if a1 and a2:
            keep1[idx1], keep2[idx2] = True, True
        idx1 += a1
        idx2 += a2
    data_mat = data_mat[:, keep1]
    data_mat2 = data_mat2[:, keep2]
    data_mat = np.concatenate((np.expand_dims(data_mat, axis=1), np.expand_dims(data_mat2, axis=1)), axis=1)


    # THE DIMENSIONS OF THE DATA MAT ARE: (sesseion, epoch_group, contact, time_bin)

    # with open('E:/epoched/contact_sel', 'wb') as fd:
    #     pickle.dump({'active_contact_list': active_contact_list}, fd)

    if SHOW_TIME_PER_CONTACT:
        for i in range(data_mat.shape[1]):
            if i % 20 == 0:
                fig, ax = plt.subplots(4, 5, figsize=(12, 8), sharex=True, sharey=True)
            i_ax = i % 20
            ax.flatten()[i_ax].plot(data_mat[0, i])
            ax.flatten()[i_ax].plot(data_mat[1, i])
            ax.flatten()[i_ax].grid(True)
            ax.flatten()[i_ax].set_ylabel(str(i))
            ax.flatten()[i_ax].set_ylim((0.5, 2))
            if i % 400 == 399:
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
    rdm_size = int((data_mat.shape[-1] - pre_ignore) / delta_time_smple)
    rdm_results = np.zeros((2 * NUM_SPLITS, rdm_size, rdm_size))


    #data_mat = data_mat.transpose((1, 0, 2, 3)) # control: is it realy rel-rep, or plain similarity?
    rdm0 = calc_rdm(data_mat[0], rdm_size, pre_ignore, delta_time_smple)
    rdm1 = calc_rdm(data_mat[1], rdm_size, pre_ignore, delta_time_smple)
    visualize_rdms(np.expand_dims(rdm0, axis=0), title='RDM for early session')
    visualize_rdms(np.expand_dims(rdm1, axis=0), title='RDM for subsequent session')

    relreps = np.zeros((2, 12, 10))
    relreps[0] = rdm0[:, 1:-1]
    relreps[1] = rdm1[:, 1:-1]
    rep_pcors = np.zeros((rdm_size, rdm_size))
    for digit_1 in range(12):
        for difit_2 in range(12):
            v1, v2 = relreps[0, digit_1], relreps[1, difit_2]
            rep_pcors[digit_1, difit_2] = (v1 * v2).sum() / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-18)
    visualize_rdms(np.expand_dims(rep_pcors, axis=0), title='full vector relative representation correlation')

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
    #
    #
    #
    # # generating a single rdm from all the data
    # # rdm = generate_rdm(np.expand_dims(data_mat, axis=0), rdm_size, pre_ignore, delta_time_smple, ses=[0, 1])
    # # visualize_rdms(rdm)
    #
    #
    # # NOW DO RELATIVE REPRASENTATION ANALISYS
    # # A RELATIVE REPRESENTATION IS A LINE IN THE "SELF" RDM (MAYBE EXCLUDING THE DIAGONAL)
    #
    # # rdm results per split
    # #
    # PADDING = False
    # #
    # rdm_results = np.zeros((2, 2, NUM_SPLITS, rdm_size, rdm_size))
    # print('now making in-session rdm''s to generate relative representations')
    # for split_id in tqdm.tqdm(range(1, NUM_SPLITS)):
    #     split_data = consistant_random_grouping(data_mat, pindex=split_id, axis=1, padding=PADDING)
    #     split_data = np.array(split_data) # axes: {contact group, session, contact (within group), time)
    #     for sess in range(2):
    #         rdms = generate_rdm(split_data, rdm_size, pre_ignore, delta_time_smple, ses=[sess, sess])
    #         rdm_results[0, sess, split_id] = rdms[0]
    #         rdm_results[1, sess, split_id] = rdms[1]
    # #
    # # #acc_split = acc_split.reshape(2, 2, acc_split.shape[2] * acc_split.shape[3], NUM_SPLITS)
    # # fig, ax = plt.subplots(2, 1, sharex=True)
    # # for i in range(2):
    # #     for split_id in range(1, 9):
    # #         ax[i].plot(acc_split.reshape(2, 2, acc_split.shape[2] * acc_split.shape[3], NUM_SPLITS)[0, i, :, split_id] + (split_id - 4) / 25, label=str(split_id))
    # #     ax[i].plot(acc_split.reshape(2, 2, acc_split.shape[2] * acc_split.shape[3], NUM_SPLITS)[0, i, :].mean(axis=-1), linewidth=4, label='avg')
    # #     ax[i].legend()
    # # acc_split = acc_split.mean(axis=-1)
    # # plt.show()
    # # rdm_results = [generate_rdm(acc_split.transpose((1, 0, 2, 3)), rdm_size, pre_ignore, delta_time_smple, ses=[0, 1]),
    # #                generate_rdm(acc_split.transpose((1, 0, 2, 3)), rdm_size, pre_ignore, delta_time_smple, ses=[0, 1])]
    # # rdm_results = np.expand_dims(np.array((rdm_results[0][0], rdm_results[0][1])), axis=0)
    # #
    # visualize_rdms(rdm_results[0, 0:1], title='RDMS for early session')
    # visualize_rdms(rdm_results[0, 1:2], title='RDMS for subsequent session')
    # rdm_results = np.array(rdm_results).transpose((1, 0, 2, 3))
    # rdm_results = np.concatenate((rdm_results, rdm_results), axis=1)
    # NUM_SPLITS = 1
    #
    # # #
    # # # RDM's from avg. vectors instead of average of RDM's'
    # # rdms = generate_rdm(np.expand_dims(split_data.mean(axis=2), axis=2), rdm_size, pre_ignore, delta_time_smple, ses=[sess, sess])
    # # visualize_rdms(rdms[0], title='RDMS for early session')
    # # visualize_rdms(rdms[1], title='RDMS for subsequent session')
    # # print('here')
    # # #
    #
    # # relreps = np.zeros((2, 2 * NUM_SPLITS, rdm_size, rdm_size))
    # # # remove the 1 correlation between a digit and itself
    # # for i in range(rdm_size):
    # #     relreps[:, :, i, :i] = rdm_results[:, :, i, :i]
    # #     relreps[:, :, i, i:] = rdm_results[:, :, i, i+1:]
    # # remove the before countdown and after countdown
    # # relreps = relreps[:, :, 1:-1, 1:-1]
    # #relreps = rdm_results[:, :, 1:-1, 1:-1]
    # relreps = rdm_results[:, :, :, 1:-1]
    #
    # rep_pcors = np.zeros((2 * NUM_SPLITS, rdm_size, rdm_size))
    # for i_split in range(2 * NUM_SPLITS):
    #     for digit_1 in range(12):
    #         for difit_2 in range(12):
    #             v1, v2 = relreps[0, i_split, digit_1], relreps[1, i_split, difit_2]
    #             rep_pcors[i_split, digit_1, difit_2] = (v1 * v2).sum() / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-18)
    # # for i_split in range(2 * NUM_SPLITS):
    # #     rep_pcors[i_split] = (relreps[0, i_split] @ relreps[1, i_split].T) / np.sqrt(((relreps[0, i_split] @ relreps[0, i_split].T) * (relreps[1, i_split] @ relreps[1, i_split].T) + 1e-12))
    # visualize_rdms(rep_pcors, title='relative representation p-correlations')
    #
    # # now use the average relative representation
    # rep_pcors = np.zeros((rdm_size, rdm_size))
    # relreps = relreps.mean(axis=1)
    # for digit_1 in range(12):
    #     for difit_2 in range(12):
    #         v1, v2 = relreps[0, digit_1], relreps[1, difit_2]
    #         rep_pcors[digit_1, difit_2] = (v1 * v2).sum() / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-18)
    # visualize_rdms(np.expand_dims(rep_pcors, axis=0), title='using average of relative representation among splits')
    #
    # # now just use a single vector, no splits
    # rdm0 = generate_rdm(np.expand_dims(data_mat, axis=0), rdm_size, pre_ignore, delta_time_smple, ses=[0, 0])
    # rdm1 = generate_rdm(np.expand_dims(data_mat, axis=0), rdm_size, pre_ignore, delta_time_smple, ses=[1, 1])
    # visualize_rdms(rdm0, title='RDM for early session')
    # visualize_rdms(rdm1, title='RDM RDMS for subsequent session')
    # relreps = np.zeros((2, 12, 10))
    # relreps[0] = rdm0[:, :, 1:-1]
    # relreps[1] = rdm1[:, :, 1:-1]
    # rep_pcors = np.zeros((rdm_size, rdm_size))
    # for digit_1 in range(12):
    #     for difit_2 in range(12):
    #         v1, v2 = relreps[0, digit_1], relreps[1, difit_2]
    #         rep_pcors[digit_1, difit_2] = (v1 * v2).sum() / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-18)
    # visualize_rdms(np.expand_dims(rep_pcors, axis=0), title='full vector relative representation correlation')
    #
    # print('here')




