
from rdm_tools import *


if __name__ == '__main__':

    V_SAMP_PER_SEC = 1
    CORR_WINDOW_SEC = 1
    SHOW_TIME_PER_CONTACT = False
    ACTIVE_CONTACTS_ONLY = False
    AUTO_OR_CROSS_ACTIVATION = "CROSS"  # "AUTO": generate session rdm from single epoch set (diagonal = 1); "CROSS": cross-correlate two epoch sets
    EPOCH_SUBSET = 'e0-e5'#
    OTHER_EPOCH_SUBSET = 'e6-e11' if AUTO_OR_CROSS_ACTIVATION == "CROSS" else EPOCH_SUBSET# None#for making self-session rdms
    AVG_MANY_EPOCHS = ['e0-e0', 'e1-e1', 'e2-e2', 'e3-e3', 'e4-e4', 'e5-e5']
    MIN_TGAP, MAX_TGAP = 24, 48#108, 180#60, 108#36, 60#72, 96#60, 160#10, 500#180, 276#
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
    #CONTACT_SELECTION_FILE_NAME = 'C:/Users/menas/OneDrive/Desktop/openneuro/temp/contact_list_cntdwn_{}_{}'.format(MIN_TGAP, MAX_TGAP)
    CONTACT_SELECTION_FILE_NAME = 'C:/Users/menas/OneDrive/Desktop/openneuro/temp/contact_list_cntdwn_best'



    data_availability_obj = data_availability()
    epoch_subsets =  [EPOCH_SUBSET, OTHER_EPOCH_SUBSET] if not AVG_MANY_EPOCHS else AVG_MANY_EPOCHS
    contact_list = data_availability_obj.get_get_contacts_for_2_session_gap_epoch_splits(min_timegap_hrs=MIN_TGAP, max_timegap_hrs=MAX_TGAP,
                                                                                         event_type=event_type, sub_event_type=event_type,
                                                                                         epoch_subsets=epoch_subsets, enforce_first=True)

    # #
    # temp_list = ['sub-R1065J', 'sub-R1083J', 'sub-R1111M', 'sub-R1112M', 'sub-R1118N', 'sub-R1161E', 'sub-R1168T', 'sub-R1172E', 'sub-R1196N',
    #              'sub-R1283T', 'sub-R1308T', 'sub-R1315T', 'sub-R1325C', 'sub-R1336T', 'sub-R1338T', 'sub-R1355T', 'sub-R1542J']
    # revised = []
    # for c in contact_list:
    #     if c['subject'] in temp_list:
    #         revised.append(c)
    # print(len(contact_list), len(revised))
    # #assert False
    # contact_list = revised2d
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

    services = contact_list_services()
    contact_list = services.remove_double_contacts(contact_list)


    pair_cnt = 0
    contact_list_ = np.copy(contact_list)
    services = contact_list_services()
    csac_list, R0_list, R1_list, rdm0_list, rdm1_list = [], [], [], [], []
    for i_sbst0 in range(len(epoch_subsets) - (1 if AUTO_OR_CROSS_ACTIVATION=='CROSS' else 0)):
        for i_sbst1 in range(i_sbst0 + 1, len(epoch_subsets)) if AUTO_OR_CROSS_ACTIVATION=='CROSS' else [i_sbst0] :

            rdm_size_, rdm0_, rdm1_, csac_, R0_, R1_, contact_list__ = do_analysis_for_two_epoch_sets(contact_list_, subject_ids, i_sbst0, i_sbst1,
                                               V_SAMP_PER_SEC, SHOW_TIME_PER_CONTACT, False, CORR_WINDOW_SEC,
                                               AUTO_OR_CROSS_ACTIVATION, CONTACT_SPLIT, PROCESS_QUADS, tfm=projector, SHOW=(i_sbst0 + i_sbst1 == 111), ccorr_mode=CROSS_SESSION_CMODE)
            contact_list_ = services.intersect_lists(contact_list_, contact_list__)
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
    # srvices = contact_list_services()
    # sess_list = services.get_sesseion_list(contact_list)
    # print(sess_list)

show_region_distribution(contact_list_, title='{} contacts , delta=T = {} hrs to {} hrs'.format(len(contact_list_), MIN_TGAP, MAX_TGAP))
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



