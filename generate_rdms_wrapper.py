import numpy as np
import matplotlib.pyplot as plt
import os
import mne
import seaborn as sns
import tqdm

from data_availability import data_availability
from epoched_analysis_wrapper import calculate_p_values # SHOULD BE MOVED ELSEWHERE



def consistant_random_grouping(data, num_groups=2, pindex=2, axis=0):

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
        groups.append(data[p[i0:i0+grp_size]].transpose(dim_perm))
        i0 += grp_size

    return groups



if __name__ == '__main__':

    V_SAMP_PER_SEC = 4
    CORR_WINDOW_SEC = 1
    SHOW_TIME_PER_CONTACT = False
    NUM_SPLITS = 1250

    data_availability_obj = data_availability()
    _, contact_list = data_availability_obj.get_contacts_for_2_session_gap(min_timegap_hrs=72, max_timegap_hrs=96, event_type='cntdwn', sub_event_type='CNTDWN')

    # make a dictionary og integer subject id's
    subject_ids = dict()
    id = 0
    for contact in contact_list:
        if not contact['subject'] in subject_ids.keys():
            subject_ids[contact['subject']] = id
            id += 1

    id_vector = np.zeros(len(contact_list), dtype=int)
    resolution_sec = 0.5
    data_mat = np.zeros((2, len(contact_list), int(V_SAMP_PER_SEC * 11)))
    boundries_sec = np.linspace(start=0-1, stop=11-1, num=data_mat.shape[-1] + 1)
    running_first, running_second = ' ', ' '
    dst_idx = 0
    for contact in contact_list:
        if (running_first != contact['first']) or (running_second != contact['second']):
            running_first, running_second = contact['first'], contact['second']
            running_subject_id = subject_ids[contact['subject']]
            first_data = mne.read_evokeds(running_first, verbose=False)[0]
            #first_data.apply_baseline((-0.5, -0.1))
            first_p_vals, first_increase, _ = calculate_p_values(first_data.copy(), show=False)
            second_data = mne.read_evokeds(running_second, verbose=False)[0]
            #second_data.apply_baseline((-0.5, -0.1))
            second_p_vals, second_increase, _ = calculate_p_values(second_data.copy(), show=False)
            masks = np.zeros((boundries_sec.size - 1, first_data.times.shape[-1]), dtype=bool)
            for i in range(data_mat.shape[-1]):
                masks[i] = (first_data.times >= boundries_sec[i]) * (first_data.times < boundries_sec[i+1])
            contact_activity_1 = (first_p_vals < 0.05) * first_increase# + True
            contact_activity_2 = (second_p_vals < 0.05) * second_increase# + True
        #
        # now read the data in the requsted resolution
        src_idx1 = np.argwhere([contact['name'] == c for c in first_data.ch_names]).squeeze()
        src_idx2 = np.argwhere([contact['name'] == c for c in second_data.ch_names]).squeeze()
        not_bad1 = np.logical_not(np.any([contact['name'] == b for b in first_data.info['bads']]))
        not_bad2 = np.logical_not(np.any([contact['name'] == b for b in second_data.info['bads']]))
        not_bad1 = not_bad1 and np.any(first_data._data[src_idx1] != 0) # !!!   PATCH    !!!
        not_bad2 = not_bad1 and np.any(second_data._data[src_idx1] != 0) # !!!   PATCH    !!!
        not_bad1 = not_bad1 and np.abs(first_data._data[src_idx1]).max() < 3 # !!!   PATCH    !!!
        not_bad2 = not_bad2 and np.abs(second_data._data[src_idx1]).max() < 3 # !!!   PATCH    !!!
        ok1 = first_p_vals[src_idx1] < 0.05
        if (src_idx1.size == 1) and (src_idx2.size == 1) and not_bad1 and not_bad2 and (contact_activity_1[src_idx1] and contact_activity_2[src_idx2]):
            for i in range(data_mat.shape[-1]):
                data_mat[0, dst_idx, i] = (first_data._data[src_idx1] * masks[i]).sum() / masks[i].sum()
                data_mat[1, dst_idx, i] = (second_data._data[src_idx1] * masks[i]).sum() / masks[i].sum()
            id_vector[dst_idx] = running_subject_id
            dst_idx += 1
    data_mat = data_mat[:, :dst_idx]

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

    pre_ignore = 0#V_SAMP_PER_SEC # 1 second in the begining
    delta_time_smple = int(V_SAMP_PER_SEC * CORR_WINDOW_SEC)
    rdm_size = int((data_mat.shape[-1] - pre_ignore) / delta_time_smple)
    rdm_results = np.zeros((2 * NUM_SPLITS, rdm_size, rdm_size))

    print('collecting tqdm statistics')
    for split_id in tqdm.tqdm(range(1, NUM_SPLITS)):
        split_data = consistant_random_grouping(data_mat, pindex=split_id, axis=1)
        split_data = np.array(split_data) # axes: {contact group, session, contact (within group), time)
        # correlation between corresponding seconds is different sessions
        rdms = np.zeros((2, rdm_size, rdm_size)) # {random group, time(1), time(2)}
        for t1 in range(rdm_size):
            for t2 in range(rdm_size):
                for grp in range(2):
                    #sig1 = split_data[grp, 0, :, t1*2*4:(t1+1)*2*4].sum(axis=1)
                    #sig2 = split_data[grp, 1, :, t2*2*4:(t2+1)*2*4].sum(axis=1)
                    sig1 = split_data[grp, 0, :, pre_ignore + t1*delta_time_smple:pre_ignore + (t1+1)*delta_time_smple].flatten()
                    sig2 = split_data[grp, 1, :, pre_ignore + t2*delta_time_smple:pre_ignore + (t2+1)*delta_time_smple].flatten()
                    sig1 -= sig1.mean()
                    sig2 -= sig2.mean()
                    if True:#t1 <= t2:#
                        rdms[grp, t1, t2] = (sig1 * sig2).sum() / (np.linalg.norm(sig1) * np.linalg.norm(sig2))
                    # if t1 == t2:
                    #     plt.plot(sig1)
                    #     plt.plot(sig2)
                    #     plt.title('t1: {}   t2: {}'.format(t1, t2))
                    #     plt.show()
        #
        rdm_results[split_id * 2] = rdms[0]
        rdm_results[split_id * 2 + 1] = rdms[1]
        #
        SHOW_EACH_RDM = False
        if SHOW_EACH_RDM:
            fig, ax = plt.subplots(1, 2, figsize=(10, 6), num='split rdm')
            sns.heatmap(np.round(rdms[0], decimals=2), vmin=-1, vmax=1, ax=ax[0], annot=True, square=True, cbar=False)
            ax[0].set_title('random contact sel {} / {}'.format(split_id, 1))
            sns.heatmap(np.round(rdms[1], decimals=2), vmin=-1, vmax=1, ax=ax[1], annot=True, square=True, cbar=False)
            ax[1].set_title('random contact sel {} / {}'.format(split_id, 2))
            plt.show(block=False)
            plt.pause(0.1)


    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
    ax[0].plot(data_mat[0])
    ax[1].plot(data_mat[1])
    plt.show()

    # generating the correlation histograms
    bin_boundaries = np.linspace(start=rdm_results.min(), stop=rdm_results.max(), num=36 if data_mat.shape[1] < 600 else 72)
    bins = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    hist_0_0, hist_0_n, hist_1_1, hist_1_n, hist_n_n, hist_n_m = \
        np.zeros(bins.size), np.zeros(bins.size), np.zeros(bins.size), np.zeros(bins.size), np.zeros(bins.size), np.zeros(bins.size)
    fig, ax = plt.subplots(1, 1)
    for t1 in range(11):
        for t2 in range(t1, 11):
            h1 = np.histogram(rdm_results[:, t1, t2], bins=bin_boundaries, density=True)[0]
            h2 = np.histogram(rdm_results[:, t2, t1], bins=bin_boundaries, density=True)[0]
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
            ax.plot(bins, h, c=c, linewidth=w)
    plt.show()
    plt.plot(bins, hist_0_0 / hist_0_0.sum(), c='m', linewidth=2, label='baseline / baseline')
    plt.plot(bins, hist_0_n / hist_0_n.sum(), c='c', linewidth=1, label='baseline / digit')
    plt.plot(bins, hist_1_n / hist_1_n.sum(), c='b', linewidth=1, label='1 / 2-9')
    plt.plot(bins, hist_1_1 / hist_1_1.sum(), c='g', linewidth=3, label='1 / 1')
    plt.plot(bins, hist_n_m / hist_n_m.sum(), c='k', linewidth=1, label='2-9 / other 2-9')
    plt.plot(bins, hist_n_n / hist_n_n.sum(), c='r', linewidth=3, label='2-9 / 2-9')
    plt.legend()
    plt.grid(True)
    plt.suptitle('{} contacts, {} split permutations'.format(dst_idx, NUM_SPLITS))
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    fig.suptitle('{} contacts, {} split permutations'.format(dst_idx, NUM_SPLITS))
    havg = rdm_results.mean(axis=0)
    havg = (havg + havg.T) / 2
    for i in range(1, 11):
        havg[i, :i] = 0
    sns.heatmap(np.round(havg, decimals=2), vmin=-1, vmax=1, ax=ax, annot=True, square=True, cbar=False)
    plt.show()




