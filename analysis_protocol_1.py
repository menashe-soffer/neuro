import os

import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from path_utils import get_subject_list, get_paths
from event_reader import event_reader
from my_montage_reader import my_montage_reader
from my_mne_wrapper import my_mne_wrapper
from my_subband_processings import calc_HFB
from my_tfr_wrapper_1 import calc_tfr

def make_data_availability_list(base_folder, region_list, hemisphere_sel, read_dates=False):

    subject_list = get_subject_list(base_folder=base_folder)
    data = dict()

    for subject in subject_list:

        paths = get_paths(base_folder=base_folder, subject=subject, sess_slct=None, mode='bipolar')
        # PATCH
        if len(paths) == 0:
            continue

        try:
            #
            subject_data = dict()

            # add contacts info
            montage = my_montage_reader(fname=paths[0]['electrodes'])
            electrode_list = montage.get_electrode_list_by_region(region_list=region_list, hemisphere_sel=hemisphere_sel)
            #subject_data['contact count'] = dict()
            total_contacts = 0
            for region in electrode_list:
                n = len(electrode_list[region])
                #subject_data['contact count'][region] = n
                subject_data[region] = n
                total_contacts += n

            # add session info
            subject_data['numsessions'] = len(paths)
            subject_data['num countdowns'] = []
            subject_data['meas dates'] = []
            for i_sess, path in enumerate(paths):
                event_reader_obj = event_reader(path['events'])
                subject_data['num countdowns'].append(len(event_reader_obj.get_countdowns()))
                #
                if read_dates:
                    subject_data['meas dates'].append(mne.io.read_raw_edf(path['signals'], preload=False).info['meas_date'])

            if total_contacts > 0:
                data[subject] = subject_data
                print(subject)
            else:
                print('no contacts in ROI for', subject)

        except:
            print('failed for', subject)

    # generate excel file
    df = pd.DataFrame.from_dict(data).T
    # "flatening" the contact counts - failed to do it so I made it flat in the first place
    writer = pd.ExcelWriter('E:/ds004789-download/data_availability_for_protocol_1.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='.'.join([r[::2][:4] for r in region_list])[:31])
    writer.close()



def calc_event_responces(paths, get_event_func, electrode_list, scan_files_only=False,
                         subs_centers=[45, 55, 65, 75, 85, 95], subs_bw=10, tscope=[-1, 10], timebin_sec=1,
                         min_events_in_sess=26, max_events_in_sess=26, min_sess=3,
                         max_sess=3, min_contacts=4, avg_depth=6, num_avg_groups=2, between_avg=13):

    p_list, sem_list, desc_list = [], [], []

    # check that there are enough valid sessions
    valid_sessions = 0
    for i_path, path in enumerate(paths):
        event_reader_obj = event_reader(path['events'])
        events = get_event_func(event_reader_obj)
        if (len(events) >= min_events_in_sess) and (len(events) <= max_events_in_sess):
            valid_sessions += 1
    if valid_sessions < min_sess:
        rc = 'not enough sessions'
        return rc, None
    #event_list, path_id_list = events[:max_sess], path_id_list[:max_sess]

    #print(subject, num_contacts, 'contacts', len(cntdwn_list), 'sessions')

    for path in paths:
        event_reader_obj = event_reader(path['events'])
        signals = my_mne_wrapper()
        signals.read_edf_file(fname=path['signals'], chanel_groups=electrode_list)
        if hasattr(signals, 'exceptions'):
            rc = signals.exceptions + ' for {} {}'.format(subject, group)
            return rc, None
        event_reader_obj.align_to_sampling_rate(old_sfreq=signals.original_sfreq, new_sfreq=signals.get_mne().info['sfreq'])
        signals.preprocess(powerline=60)#, passband=[60, 160])
        #
        chan_names = signals.get_mne().info['ch_names']
        if len(chan_names) <= min_contacts:
            #rc = 'not enough channels for {} {}'.format(subject, group)
            rc = 'not enough channels'
            #print(rc)
            return rc, None, None, None, None

        sess_events = get_event_func(event_reader_obj)
        # PATCH
        admit_list = np.argwhere([e['onset sample'] > 500 for e in sess_events]).squeeze()
        sess_events = [sess_events[i] for i in admit_list]
        #
        events = np.zeros((len(sess_events), 3), dtype=int)
        events[:, 0] = np.array([e['onset sample'] for e in sess_events])
        signals.set_events(events=events, event_glossary={0: 'cntdwn'})

        if events[:, 0].max() > signals.get_mne().get_data().shape[-1]:
            # rc = subject + ':      events[:, 0].max() > signals.get_mne().get_data().shape[-1]'
            rc = ':      events[:, 0].max() > signals.get_mne().get_data().shape[-1]'
            return rc, None, None, None, None

        # process
        if not scan_files_only:
            p_list.append([])
            sem_list.append([])
            desc_list.append([])
            eidx = np.array((0, avg_depth))
            precalc = None
            for i in range(num_avg_groups):
                #print(path, signals.get_mne().get_data().shape)
                print('>>>>>>       ', path['signals'])
                _, p, sem, precalc = calc_HFB(signals.get_mne().get_data(), dbg_markers=events[eidx[0]:eidx[1], 0], chan_names=chan_names,
                                              sub_centers=subs_centers, subs_bw=subs_bw, tscope=tscope, plot_prefix=None, gen_plots=False, precalc=precalc)
                boundaries = np.arange(start=0, stop=p.shape[1]+1, step=int(signals.get_mne().info['sfreq'] * timebin_sec))
                p_list[-1].append(np.array([p[:, b1:b2].mean(axis=1) for (b1, b2) in zip(boundaries[:-1], boundaries[1:])]).T)
                sem_list[-1].append(np.array([sem[:, b1:b2].mean(axis=1) for (b1, b2) in zip(boundaries[:-1], boundaries[1:])]).T)
                desc_list[-1].append(path['signals'][path['signals'].find('ses'):][:5] + '  events {} - {}'.format(eidx[0], eidx[1]))
                eidx += between_avg

        if len(desc_list) >= max_sess:
            break

    return 0, p_list, sem_list, desc_list, chan_names


def calc_contdwn_responces(subject, regions=None, sides=None, band=[40, 100], timebin_sec=0.5, tscope=[-1, 10],
                           avg=6, in_session_gap=13, mode='monopolar', scan_files_only=True, display_span=3):

    # configurations for the sub=band processor
    # generate the sub-band spec for the processing
    subs_centers = np.linspace(start=band[0], stop=band[-1], num=int((band[-1] - band[0]) / 10 + 1.5))
    subs_centers = (subs_centers[:-1] + subs_centers[1:]) / 2
    subs_bw = np.diff(subs_centers)[0]
    # result holders
    p_list, sem_list, desc_list = [], [], []

    paths = get_paths(base_folder=base_folder, subject=subject, sess_slct=None, mode=mode)
    if len(paths) == 0:
        rc = subject + ':   no paths'
        return rc, None

    montage = my_montage_reader(fname=paths[0]['electrodes'])
    electrode_list = montage.get_electrode_list_by_region(region_list=regions, hemisphere_sel=sides)
    contact_list = []
    for group in electrode_list:
        for e in electrode_list[group]: # adding the group indication to the list
            e['group'] = group
        contact_list = contact_list + electrode_list[group]
    num_contacts = len(contact_list)
    if num_contacts < 2:#4:
        rc = 'not enough contacts for {} {}'.format(subject, group)
        #print(rc)
        return rc, None

    rc, p_list, sem_list, desc_list, chan_names = calc_event_responces(paths=paths, get_event_func=event_reader.get_countdowns,
                                                                       electrode_list=electrode_list, scan_files_only=scan_files_only,
                                                                       subs_centers=subs_centers, subs_bw=subs_bw, tscope=tscope, timebin_sec=timebin_sec)

    if rc:
        return subject + ':  ' + rc, None
    # cntdwn_list = []
    # for path in paths:
    #     event_reader_obj =  event_reader(path['events'])
    #     cntdwns = event_reader_obj.get_countdowns()
    #     if len(cntdwns) == 26:#avg + in_session_gap:
    #         cntdwn_list.append(cntdwns)
    # if len(cntdwn_list) < 2:
    #     rc = 'not enough valid sessions for {} {}'.format(subject, group)
    #     #print(rc)
    #     return rc, None
    #
    # print(subject, num_contacts, 'contacts', len(cntdwn_list), 'sessions')
    #
    # for path in paths:
    #     event_reader_obj = event_reader(path['events'])
    #     signals = my_mne_wrapper()
    #     signals.read_edf_file(fname=path['signals'], chanel_groups=electrode_list)
    #     if hasattr(signals, 'exceptions'):
    #         rc = signals.exceptions + ' for {} {}'.format(subject, group)
    #         return rc, None
    #     event_reader_obj.align_to_sampling_rate(old_sfreq=signals.original_sfreq, new_sfreq=signals.get_mne().info['sfreq'])
    #     signals.preprocess(powerline=60)#, passband=[60, 160])
    #     #
    #     chan_names = signals.get_mne().info['ch_names']
    #     if len(chan_names) < 4:
    #         rc = 'not enough channels for {} {}'.format(subject, group)
    #         #print(rc)
    #         return rc, None
    #
    #     cntdwn_events = event_reader_obj.get_countdowns()
    #     # PATCH
    #     admit_list = np.argwhere([e['onset sample'] > 500 for e in cntdwn_events]).squeeze()
    #     cntdwn_events = [cntdwn_events[i] for i in admit_list]
    #     #
    #     events = np.zeros((len(cntdwn_events), 3), dtype=int)
    #     events[:, 0] = np.array([e['onset sample'] for e in cntdwn_events])
    #     signals.set_events(events=events, event_glossary={0: 'cntdwn'})
    #
    #     if events[:, 0].max() > signals.get_mne().get_data().shape[-1]:
    #         rc = subject + ':      events[:, 0].max() > signals.get_mne().get_data().shape[-1]'
    #         #print(rc)
    #         return rc, None
    #
    #     # process
    #     if not scan_files_only:
    #         p_list.append([])
    #         sem_list.append([])
    #         desc_list.append([])
    #         eidx = np.array((0, avg))
    #         precalc = None
    #         for i in range(2):
    #             #print(path, signals.get_mne().get_data().shape)
    #             print('>>>>>>       ', path['signals'])
    #             _, p, sem, precalc = calc_HFB(signals.get_mne().get_data(), dbg_markers=events[eidx[0]:eidx[1], 0], chan_names=chan_names,
    #                                           sub_centers=subs_centers, subs_bw=subs_bw, tscope=[-5, 18], plot_prefix=None, gen_plots=False, precalc=precalc)
    #             boundaries = np.arange(start=0, stop=p.shape[1]+1, step=int(signals.get_mne().info['sfreq'] * timebin_sec))
    #             p_list[-1].append(np.array([p[:, b1:b2].mean(axis=1) for (b1, b2) in zip(boundaries[:-1], boundaries[1:])]).T)
    #             sem_list[-1].append(np.array([sem[:, b1:b2].mean(axis=1) for (b1, b2) in zip(boundaries[:-1], boundaries[1:])]).T)
    #             desc_list[-1].append(path['signals'][path['signals'].find('ses'):][:5] + '  events {} - {}'.format(eidx[0], eidx[1]))
    #             eidx += in_session_gap
    #
    #     MAX_SESSIONS_TO_PROCESS = 3
    #     if len(desc_list) >= MAX_SESSIONS_TO_PROCESS:
    #         break


    #rc = '{}   {} contacts   {} channels   {} sessions'.format(subject, num_contacts, len(chan_names), len(cntdwn_list))
    rc = '{}   {} contacts'.format(subject, num_contacts)

    if scan_files_only:
        return rc, None
    else:
        fig, ax = plt.subplots(2, 4, figsize=(15, 10), num='calc_contdwn_responces')
        fig.clf()
        fig, ax = plt.subplots(2, 4, figsize=(15, 10), num='calc_contdwn_responces')
        #fig.suptitle(subject)
        for i_ses in range(4):
            for i_pos in range(2):
                if i_ses < len(desc_list):
                    sns.heatmap(p_list[i_ses][i_pos], ax=ax[i_pos, i_ses], vmin=0.0, vmax=display_span, cbar=False, square=False,
                                yticklabels=np.arange(p_list[i_ses][i_pos].shape[0]), xticklabels=(np.arange(start=-1, stop=20) + 0.5) / 2)
                    ax[i_pos, i_ses].set_xlabel(desc_list[i_ses][i_pos])
                else:
                    ax[i_pos, i_ses].axis('off')
        plt.show(block=False)
        plt.pause(0.25)

        if mode == 'bipolar':
            # generate the bipolat contact list
            used_contacts = []
            for name in chan_names:
                mask1 = np.array([c['name'] == name.split('-')[0] for c in contact_list])
                mask2 = np.array([c['name'] == name.split('-')[1] for c in contact_list])
                used_contacts.append([contact_list[np.argwhere(mask1).squeeze()], contact_list[np.argwhere(mask2).squeeze()]])
        contact_list = used_contacts

        return rc, dict({'p': p_list, 'sem': sem_list, 'desc': desc_list, 'fig': fig, 'contact list': contact_list})


def run_analysis_protocol_1(subject_list, mode='bipolar', region_list=[], avg_depth=13, ovld_thd=4):
    display_span = ovld_thd * 0.75

    logfile_name = os.path.join(base_folder, 'logfile.txt')
    logfile_fd = open(logfile_name, 'wt')

    for subject in subject_list:
        rc, data = calc_contdwn_responces(subject, regions=region_list, sides=hemisphere_sel, mode=mode,
                                          scan_files_only=False, avg=avg_depth, display_span=display_span, tscope=[-5, 10], timebin_sec=0.1)
        print(subject, rc)
        logfile_fd.write(rc + '\n')
        logfile_fd.flush()
        if data is not None:
            data['fig'].suptitle(subject + '\navg={}'.format(avg_depth))
            fname = os.path.join(base_folder, 'plots', subject + '_' + mode + '_raw_avgs_' + str(avg_depth) + '.png')
            data['fig'].savefig(fname)
            if mode == 'monopolar':
                pass
            else:
                contacts = ['{}-{}  ({} , {})   {} {}'.format(d[0]['name'], d[1]['name'], d[0]['group'], d[1]['group'],
                                                               str(np.round(d[0]['coords'].squeeze(), decimals=2)),
                                                               str(np.round(d[1]['coords'].squeeze(), decimals=2))) for d in data['contact list']]
            fname = os.path.join(base_folder, 'plots', subject + '_' + mode)
            np.savez(fname, p=data['p'], sem=data['sem'], contacts=np.array(contacts))
            #
            #
            #
            #
            # visualize
            nchans, nses = data['p'][0][0].shape[0], len(data['p'])
            num_lags = 4  # how many time taps to visualize
            vectors = np.zeros((2, nses, nchans, num_lags))
            #
            fig, ax = plt.subplots(2, num_lags, num='pierson', figsize=(15, 10))
            fig.clf()
            fig, ax = plt.subplots(2, num_lags, num='pierson', figsize=(15, 10))
            for i_lag in range(num_lags):
                ax[0, i_lag].grid(True)
                ax[0, i_lag].set_ylim([0.5, display_span])
                ax[0, i_lag].set_title('{:2.1f} - {:2.1f} sec.'.format(i_lag / 2, (i_lag + 1) / 2))
            #
            for i_ses, (p_ses, sem_ses) in enumerate(zip(data['p'], data['sem'])):
                for i_rpt in range(2):
                    # normalize the data
                    nf = np.tile(p_ses[i_rpt][:, 0], (p_ses[i_rpt].shape[1], 1)).T
                    # p_ses[i_rpt] /= nf
                    # sem_ses[i_rpt] /= nf
                    vectors[i_rpt, i_ses] = p_ses[i_rpt][:, 1:num_lags + 1] / nf[:, 1:num_lags + 1]
                    for i_lag in range(num_lags):
                        ax[0, i_lag].plot(vectors[i_rpt, i_ses, :, i_lag], label='s{}r{}'.format(i_ses + 1, i_rpt + 1))
            # pierson correlation

            n1, n2, nchans, nlags = vectors.shape
            ur_vectors = vectors.reshape(n1 * n2, nchans, nlags)
            pc = np.zeros((n1 * n2, n1 * n2))
            for i_lag in range(nlags):
                tap_vectors = ur_vectors[:, :, i_lag]
                mu = tap_vectors.mean(axis=0)
                ax[0, i_lag].plot(mu, ':', linewidth=2, label='avg')
                ax[0, i_lag].legend()
                # plain covariance
                c = (tap_vectors - mu) @ (tap_vectors - mu).T
                sigma = np.sqrt(np.diag(c))
                for i in range(c.shape[0]):
                    c[:, i] /= sigma[i]
                    c[i, :] /= sigma[i]
                # pairwise pierson
                for i1 in range(n1 * n2):
                    for i2 in range(n1 * n2):
                        tap_vector_ij = tap_vectors[(i1, i2), :]
                        # tap_vector_ij = tap_vector_ij * (tap_vector_ij < ovld_thd) # PATCH!!! FOR HIGH LEVEL NOISES
                        muij = tap_vector_ij.mean(axis=1)
                        dij = tap_vector_ij - muij.reshape(2, 1)
                        pc[i1, i2] = (dij[0].reshape(1, nchans) @ dij[1].reshape(nchans, 1)).squeeze() / (
                                    np.linalg.norm(dij[0]) * np.linalg.norm(dij[1]) + 1e-12)
                        # pc[i1, i2] = np.corrcoef(tap_vector_ij)[0, 1]

                sns.heatmap(np.round(pc, decimals=2), ax=ax[1, i_lag], vmin=-1, vmax=1, annot=True, cbar=False,
                            square=True)

            fig.suptitle('PAIRWISE pearson correlation    ' + subject + '\navg={}'.format(avg_depth))
            fname = os.path.join(base_folder, 'plots', subject + '-' + mode + '_pearson' + str(avg_depth) + '.png')
            fig.savefig(fname)
            plt.show(block=False)
            plt.pause(0.25)

    logfile_fd.close()


base_folder = 'E:/ds004789-download'

# area of interest definition
region_list = ['fusiform', 'inferiortemporal', 'lateraloccipital', 'lingual']
hemisphere_sel = ['LR', 'LR', 'LR', 'both']

# create table of available data
if False:
    subjects_list = get_subject_list(base_folder=base_folder)
    make_data_availability_list(base_folder=base_folder, region_list=region_list, hemisphere_sel=hemisphere_sel)
    assert False

# parameters for the protocol
# regions = ['fusiform', 'inferiortemporal', 'lateraloccipital', 'lingual']
# sides = ['LR', 'LR', 'LR', 'both']
subject_list = ['sub-R1243T', 'sub-R1281E', 'sub-R1334T', 'sub-R1338T', 'sub-R1346T', 'sub-R1355T', 'sub-R1425D',
                'sub-R1299T', 'sub-R1065J', 'sub-R1060M', 'sub-R1092J', 'sub-R1292E', 'sub-R1308T', 'sub-R1315T', 'sub-R1350D',
                'sub-R1094T', 'sub-R1123C', 'sub-R1153T', 'sub-R1154D', 'sub-R1156D', 'sub-R1161E', 'sub-R1168T', 'sub-R1223E']
mode =  'bipolar'#'monopolar'#
subject_list = ['sub-R1060M', 'sub-R1065J', 'sub-R1092J', 'sub-R1094T', 'sub-R1123C', 'sub-R1145J', 'sub-R1153T',
                'sub-R1154D', 'sub-R1161E', 'sub-R1168T', 'sub-R1195E', 'sub-R1223E', 'sub-R1243T', 'sub-R1281E', 'sub-R1292E',
                'sub-R1299T', 'sub-R1308T', 'sub-R1315T', 'sub-R1334T', 'sub-R1338T', 'sub-R1341T', 'sub-R1350D', 'sub-R1355T', 'sub-R1425D']

region_list, hemisphere_sel = region_list[3:4], hemisphere_sel[3:4]
if False:
    run_analysis_protocol_1(subject_list, mode='bipolar', region_list=region_list)


def pairwise_pearson(data, axis=-1):

    if axis != -1:
        data = np.copy(data.T)
    n_codes = data.shape[0]
    n_chans = data.shape[-1]
    c = np.zeros((n_codes, n_codes))
    for i1 in range(n_codes):
        for i2 in range(i1, n_codes):
            vij = data[(i1, i2), :]
            vij = vij - vij.mean(axis=-1).reshape(2, 1)
            test = np.linalg.norm(vij[1] - vij[0]) / min(np.linalg.norm(vij[0]), np.linalg.norm(vij[1])) < 3.5
            c[i1, i2] = ((vij[0]).reshape(1, n_chans) @ vij[1].reshape(n_chans, 1)).squeeze() / (np.linalg.norm(vij[0]) * np.linalg.norm(vij[1]) + 1e-12)
            NORMDIST = False
            if NORMDIST:
                a, b = vij[0] / np.linalg.norm(vij[0]), vij[1] / np.linalg.norm(vij[1])
                c[i1, i2] = 1 - np.linalg.norm(a - b)
            c[i1, i2] = c[i1, i2] * test - 9 * (1 - test)
            # print(i1, i2, np.linalg.norm(vij[1] - vij[0]) , np.sqrt((np.linalg.norm(vij[0]) * np.linalg.norm(vij[1]) + 1e-12)),
            #       np.linalg.norm(vij[0]), np.linalg.norm(vij[1]), c[i1, i2], np.linalg.norm(vij[1] - vij[0]) / min(np.linalg.norm(vij[0]), np.linalg.norm(vij[1])))

    return c


def remove_one_contact(p, mask):

    active_list = np.argwhere(mask).flatten()
    base_mat = pairwise_pearson(p[:, mask, 1])
    base_score = min(base_mat[0, 1], base_mat[2, 3], base_mat[4, 5])
    best_score = base_score
    for i in active_list:
        mask_i = np.copy(mask)
        mask_i[i] = False
        mat = pairwise_pearson(p[:, mask_i, 1])
        score = min(mat[0, 1], mat[2, 3], mat[4, 5])
        if score > best_score:
            best_mask = np.copy(mask_i)
            best_score = score
    if best_score > base_score:
        return best_mask
    else:
        return mask


if True:
    paggr = np.zeros((3, 2, 0, 21))
    paggr = np.zeros((3, 2, 0, 150))
    contacts = np.zeros(0, dtype=str)
    for subject in subject_list:
        fname = os.path.join(base_folder, 'plots', subject + '_' + mode + '.npz')
        try:
            data = np.load(fname)
            p, sem = data['p'][:3], data['sem'][:3]
            paggr = np.concatenate((paggr, p), axis=2)
            # contacts = np.concatenate((contacts, data['contacts']))
            contacts = np.concatenate((contacts, np.array([c.split()[0] for c in data['contacts']])))
            print('read', fname, 'aggr. shape:', paggr.shape)
        except:
            print('can''t read', fname)

    #paggr[:, :, :, 1] = paggr[:, :, :, (1, 2, 3, 4)].mean(axis=-1)
    #
    # calc the contact varainces
    nrpts = np.prod(paggr.shape[:2])
    paggr = paggr.reshape(nrpts, paggr.shape[2], paggr.shape[3])
    #
    REMOVE_OUTLAIRS = True
    if REMOVE_OUTLAIRS:
        paggr = paggr * (paggr < 19)
    # post-stimulus time histogram (PSTH)
    PSTH_mean = paggr.mean(axis=(0, 1))
    PSTH_std = paggr.std(axis=(0, 1))
    PSTH_sem = PSTH_std / np.sqrt(PSTH_std.shape[0])
    tscale = np.linspace(start=-1, stop=10, num=150)
    plt.bar(tscale, PSTH_mean, width=0.08, facecolor='w', edgecolor='b', log=False)
    plt.bar(tscale, 2 * PSTH_sem, bottom = PSTH_mean - PSTH_sem, width=0.04, color='k', log=False)
    plt.grid()
    plt.show()
    #
    for sec_choice in [0, 5, 9, 10, 12, 16, -1, -5]:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        bin1 = int(2 * (5.5 + sec_choice))
        p1 = paggr[:, :, bin1:bin1+2].mean(axis=-1)
        pmean = p1.mean(axis=0)
        pstd = p1.std(axis=0)
        psem = pstd / np.sqrt(nrpts)
        idx = np.argsort(pmean)
        ax.bar(np.arange(paggr.shape[1]), pmean[idx], width=0.75, facecolor='w', edgecolor='b', log=True)
        # ax.plot(pmean[idx], 'b')
        # ax.fill_between(np.arange(paggr.shape[1]), pmean[idx] - pstd[idx], pmean[idx] + pstd[idx], color='b', alpha=.15)
        # ax.bar(np.arange(paggr.shape[1]), pmean[idx] + pstd[idx], width=0.4)
        # ax.bar(np.arange(paggr.shape[1]), pmean[idx] + psem[idx], width=0.4)
        ax.bar(np.arange(paggr.shape[1]), 2 * psem[idx], bottom = pmean[idx] - psem[idx], width=0.3, color='k', log=True)
        ax.grid(True)
        if sec_choice < 0:
            ax.set_title('{} .. {} before countdown'.format(sec_choice, sec_choice + 1))
        if sec_choice == 0:
            ax.set_title('first countdown second')
        if (sec_choice > 0) and (sec_choice < 10):
            ax.set_title('countdown onset + {} to countdown + {}'.format(sec_choice, sec_choice + 1))
        if sec_choice >= 10:
            ax.set_title('{} to {} after countdown end'.format(sec_choice - 10, sec_choice - 9))
        #ax.set_ylim([-0.5, 5])
        ax.set_ylim([0.5, 15])
        plt.show()
        fig.savefig(os.path.join(base_folder, 'plots', 'variances-sec-{}'.format(sec_choice)))
    assert False
    #

    num_contacts = paggr.shape[2]
    paggr = np.concatenate((paggr[0], paggr[1], paggr[2]), axis=0)
    admit = np.zeros(num_contacts, dtype='bool')
    for i_chan in range(num_contacts):
        show = False
        if show:
            plt.clf()
            plt.plot(paggr[:, i_chan].flatten())
        sorted = np.sort(paggr[:, i_chan].flatten())
        if show:
            plt.plot(sorted, c='r')
            plt.ylim([0.5, 10])
            plt.grid(True)
        q6, q114, q119 = sorted[5], sorted[113], sorted[-1]
        admit[i_chan] = (q6 > 0.8) * (q114 > 1.6) * (q119 < (10 * q114)) * (q119 < 150)
        if show and admit[i_chan]:
            plt.title('{},  q6: {:4.1f}  q114:  {:4.1f}    max: {:4.1f}'.format(i_chan, q6, q114, q119))
            plt.show(block=False)
            plt.pause(0.5)
    print(admit.sum(), 'out of', num_contacts)

    split = (np.arange(num_contacts) % 2).astype(bool)
    # p_odd_all = paggr[:, np.logical_not(split), 1]
    # p_even_all = paggr[:, split, 1]
    # p_odd_admt = paggr[:, np.logical_not(split) * admit, 1]
    # p_even_admt = paggr[:, split * admit, 1]
    fig, ax = plt.subplots(2, 4, figsize=(12, 9))
    def show_one_case(p, col, desc):
        ax[0, col].plot(p.T)
        ax[0, col].grid(True)
        ax[0, col].set_ylim((0.5, 5))
        ax[0, col].legend({'s1r1', 's1r2', 's2r1', 's2r2', 's3r1', 's3r2'})
        ax[0, col].set_title(desc)
        sns.heatmap(np.round(pairwise_pearson(p), decimals=2), ax=ax[1, col], annot=True, vmin=0, vmax=1, cbar=False, square=True)
        for x in (2, 4):
            ax[1, col].plot([x, x], [0, 6], c='y', linewidth=2)
            ax[1, col].plot([0, 6], [x, x], c='y', linewidth=2)

    show_one_case(paggr[:, np.logical_not(split), 1], 0, 'odd, all')
    show_one_case(paggr[:, split, 1], 1, 'even, all')
    show_one_case(paggr[:, np.logical_not(split) * admit, 1], 2, 'odd, admt')
    show_one_case(paggr[:, split * admit, 1], 3, 'even, admt')

    plt.show()

    # fig, ax = plt.subplots(2, 4, figsize=(12, 9))
    # mask_odd = np.logical_not(split) * admit
    # mask_even = split * admit
    # for i in range(max(mask_odd.sum(), mask_even.sum()) - 4):
    #         mask_odd = remove_one_contact(paggr, mask_odd)
    #         mask_even = remove_one_contact(paggr, mask_even)
    # show_one_case(paggr[:, np.logical_not(split) * admit, 1], 2, 'odd, admt')
    # show_one_case(paggr[:, split * admit, 1], 3, 'even, admt')
    # show_one_case(paggr[:, mask_odd, 1], 0, 'odd, filtered')
    # show_one_case(paggr[:, mask_even, 1], 1, 'even, filterd')
    # plt.show()

    # now try to filter all admitted
    # for i_chan in range(num_contacts):
    #     sorted = np.sort(paggr[:, i_chan].flatten())
    #     q6, q114, q119 = sorted[5], sorted[113], sorted[-1]
    #     admit[i_chan] = (q6 > 0.8) * (q114 > 1.6) * (q119 < (7 * q114)) * (q119 < 90)

    fig, ax = plt.subplots(2, 4, figsize=(12, 9))
    show_one_case(paggr[:, admit, 1], 0, 'admt')
    mask = np.copy(admit)
    for i in range(admit.sum() - 4):
        mask = remove_one_contact(paggr, mask)
    show_one_case(paggr[:, mask, 1], 1, 'admt, filtered')
    show_one_case(paggr[:, mask * np.logical_not(split), 1], 2, 'odd, filtered')
    show_one_case(paggr[:, mask * split, 1], 3, 'even, filterd')
    #
    # print contacts related daya
    print('\nselected electrodes:\n', contacts[mask].T)
    clist = list([c.split(' ')[2][1:] for c in contacts])
    regions = np.unique(clist)
    clist_filtered = list([c.split(' ')[2][1:] for c in contacts[mask]])
    df = pd.DataFrame(columns=regions)
    df.loc[0] = [np.array([c == regions[i] for c in clist]).sum().astype(int) for i in range(len(regions))]
    df.loc[1] = [np.array([c == regions[i] for c in clist_filtered]).sum().astype(int) for i in range(len(regions))]
    df.loc[2] = np.round(100 * df.iloc[0] / len(clist), decimals=1)
    df.loc[3] = np.round(100 * df.iloc[1] / len(clist_filtered), decimals=1)
    pd.set_option('display.max_columns', None)
    pd.options.display.width = 120
    print(df)
    fig= plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(projection='3d')
    ccoords = [c[c.find('[') + 1:] for c in contacts]
    ccoords1 = [c[:c.find(']')] for c in ccoords]
    x1 = np.array([float(c.split()[0]) for c in ccoords1])
    y1 = np.array([float(c.split()[1]) for c in ccoords1])
    z1 = np.array([float(c.split()[2]) for c in ccoords1])
    ccoords2 = [c[c.find('[') + 1:] for c in ccoords]
    ccoords2 = [c[:c.find(']')] for c in ccoords2]
    x2 = np.array([float(c.split()[0]) for c in ccoords2])
    y2 = np.array([float(c.split()[1]) for c in ccoords2])
    z2 = np.array([float(c.split()[2]) for c in ccoords2])
    ax.scatter(x1, y1, z1,  c='k', s=8)
    ax.scatter(x2, y2, z2, c='k', s=10)
    for i in np.argwhere(admit).flatten():
        ax.plot((x1[i], x2[i]), (y1[i], y2[i]), (z1[i], z2[i]), c='k')
    for i in np.argwhere(mask).flatten():
        #print('({:4.1f} , {:4.1f} , {:4.1f})   ;   ({:4.1f} , {:4.1f} , {:4.1f})'.format(x1[i], y1[i], z1[i], x2[i], y2[i], z2[i]))
        ax.plot((x1[i], x2[i]), (y1[i], y2[i]), (z1[i], z2[i]), c='r')
    #
    plt.show()


