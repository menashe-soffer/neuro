
from rdm_tools_new import *

def split_session_to_fake_sessions(contact_list, segment_size=3, pair_idx=1):

    for i_contact, contact in enumerate(contact_list):
        contact['second'] = contact['first'][int(pair_idx*segment_size):int((pair_idx+1)*segment_size)]
        contact['first'] = contact['first'][:segment_size]

    return contact_list



if __name__ == '__main__':

    V_SAMP_PER_SEC = 1
    CORR_WINDOW_SEC = 1
    SHOW_TIME_PER_CONTACT = False
    ACTIVE_CONTACTS_ONLY = False
    AUTO_OR_CROSS_ACTIVATION = "CROSS"  # "AUTO": generate session rdm from single epoch set (diagonal = 1); "CROSS": cross-correlate two epoch sets
    EPOCH_SUBSET = 'e0-e5'#
    OTHER_EPOCH_SUBSET = 'e6-e11' if AUTO_OR_CROSS_ACTIVATION == "CROSS" else EPOCH_SUBSET# None#for making self-session rdms
    AVG_MANY_EPOCHS = ['e0-e0', 'e1-e1', 'e2-e2', 'e3-e3', 'e4-e4', 'e5-e5', 'e6-e6', 'e7-e7', 'e8-e8', 'e9-e9', 'e10-e10', 'e11-e11', 'e12-e12', 'e13-e13', 'e14-e14', 'e15-e15', 'e16-e16', 'e17-e17']
    #AVG_MANY_EPOCHS = ['e0-e2', 'e3-e5', 'e6-e8', 'e9-e11', 'e12-e14', 'e15-e17']
    MIN_TGAP, MAX_TGAP = 60, 144#144, 336#24, 48
    SELECT_CONTACTS_BY_PERIODICITY = 0 # 0: ignore periodicity, 1: select periodic contacts, -1: select NON-periodic contacts
    CONTACT_SPLIT = None # None: use all, 0: even contacts only, 1: odd contacts only
    PROCESS_QUADS = False
    event_type = 'CNTDWN' # one of: 'CNTDWN', 'RECALL', 'DSTRCT', 'REST'
    CROSS_SESSION_CMODE = 'p'
    #
    WITHIN_SESSION_PROCESS = True
    if WITHIN_SESSION_PROCESS:
        MIN_TGAP, MAX_TGAP = 1, 1000
        WITHIN_SESSION_SEGMENT_SIZE, WITHIN_SESSION_PAIR_IDX = 4, 1
        #WITHIN_SESSION_SEGMENT_SIZE, WITHIN_SESSION_PAIR_IDX = 3, 1
    else:
        WITHIN_SESSION_SEGMENT_SIZE = len(AVG_MANY_EPOCHS)
    #
    # if PROCESS_QUADS:
    #     V_SAMP_PER_SEC = V_SAMP_PER_SEC * 4
    #     CORR_WINDOW_SEC = CORR_WINDOW_SEC / 4
    # #assert (AUTO_OR_CROSS_ACTIVATION == "CROSS") or (not AVG_MANY_EPOCHS)
    # #
    # SELECT_CONTACTS_BY_CORR = False
    # V_SAMP_FOR_SLCT = 4
    SAVE_CONTACT_LIST = False
    USE_CONTACT_SELECTION_FROM_FILE = True
    #CONTACT_SELECTION_FILE_NAME = 'C:/Users/menas/OneDrive/Desktop/openneuro/temp/contact_list_cntdwn_{}_{}'.format(MIN_TGAP, MAX_TGAP)
    CONTACT_SELECTION_FILE_NAME = 'C:/Users/menas/OneDrive/Desktop/openneuro/temp/contact_list_cntdwn_3_5'



    data_availability_obj = data_availability()
    epoch_subsets =  [EPOCH_SUBSET, OTHER_EPOCH_SUBSET] if not AVG_MANY_EPOCHS else AVG_MANY_EPOCHS
    # contact_list = data_availability_obj.get_get_contacts_for_2_session_gap_epoch_splits(min_timegap_hrs=MIN_TGAP, max_timegap_hrs=MAX_TGAP,
    #                                                                                      event_type=event_type, sub_event_type=event_type,
    #                                                                                      epoch_subsets=epoch_subsets, enforce_first=True, single_session=WITHIN_SESSION_PROCESS)
    
    list_1C, list_2C = data_availability_obj.get_suitable_epoch_files_and_contacts(min_timegap_hrs=MIN_TGAP, max_timegap_hrs=MAX_TGAP,
                                                                                           proc_type='gamma_c_60_160', event_list=['CNTDWN'], 
                                                                                           num_epochs=18, enforce_first=True, single_session=WITHIN_SESSION_PROCESS)
    
    list_1R, list_2R = data_availability_obj.get_suitable_epoch_files_and_contacts(min_timegap_hrs=MIN_TGAP, max_timegap_hrs=MAX_TGAP,
                                                                                           proc_type='gamma_c_60_160', event_list=['RECALL'], 
                                                                                           num_epochs=18, enforce_first=True, single_session=WITHIN_SESSION_PROCESS)
    
    (list_1C, list_2C, list_1R, list_2R) = data_availability_obj.intersect_epoch_files_and_contact_lists([list_1C, list_2C, list_1R, list_2R])
    
    if event_type == 'CNTDWN':
        list_1, list_2 = list_1C, list_2C
        RANDOM_SHIFT = False
    if event_type == 'RECALL':
        list_1, list_2 = list_1R, list_2R
        RANDOM_SHIFT = True

    
    # #
    # keep_list = ['sub-R1001P', 'sub-R1018P', 'sub-R1020J', 'sub-R1026D', 'sub-R1030J',
    #         'sub-R1031M', 'sub-R1036M', 'sub-R1042M', 'sub-R1048E', 'sub-R1049J',
    #         'sub-R1051J', 'sub-R1056M', 'sub-R1060M', 'sub-R1063C', 'sub-R1066P',
    #         'sub-R1069M', 'sub-R1076D', 'sub-R1084T', 'sub-R1098D', 'sub-R1104D',
    #         'sub-R1106M', 'sub-R1111M', 'sub-R1112M', 'sub-R1115T', 'sub-R1123C',
    #         'sub-R1124J', 'sub-R1125T', 'sub-R1130M', 'sub-R1134T', 'sub-R1138T',
    #         'sub-R1142N', 'sub-R1148P', 'sub-R1149N', 'sub-R1173J', 'sub-R1175N',
    #         'sub-R1232N', 'sub-R1234D', 'sub-R1247P', 'sub-R1264P', 'sub-R1274T',
    #         'sub-R1286J', 'sub-R1291M', 'sub-R1292E', 'sub-R1299T', 'sub-R1313J',
    #         'sub-R1332M', 'sub-R1338T', 'sub-R1342M', 'sub-R1346T', 'sub-R1355T',
    #         'sub-R1415T']
    # tmp_list = list(list_1['subject'])
    # mask = np.zeros(len(tmp_list), dtype='bool')
    # for i in range(len(tmp_list)):
    #     if tmp_list[i] in keep_list:
    #         mask[i] = True
    # list_1 = list_1[mask]
    
    # #
    
    contact_info = data_availability_obj.get_contact_info(list_1)
    
    boundary_sec = np.arange(start=-1, stop=11+1e-6, step=1/V_SAMP_PER_SEC)

    data_1 = read_epoch_files_by_list(list_1, first_epoch=0, last_epoch=18, boundary_sec=boundary_sec, random_shift=RANDOM_SHIFT)
    mask_ovf = data_1 > 3
    bad_epoch_contact = mask_ovf.sum(axis=-1) > 0
    bad_contact = bad_epoch_contact.sum(axis=0) > 0
    for i_epoch in range(data_1.shape[0]):
        for i_cntct in range(data_1.shape[1]):
            data_1[i_epoch, i_cntct] *= (1. - bad_epoch_contact[i_epoch, i_cntct])
    good_contact = np.logical_not(bad_contact)
    data_1 = data_1[:, good_contact, :]
    print('use {:5.1f} percent of contacts'.format(100 * good_contact.mean()))
                          

    if WITHIN_SESSION_PROCESS:
        scnd_start_idx = WITHIN_SESSION_SEGMENT_SIZE * WITHIN_SESSION_PAIR_IDX
        data_2 = data_1[scnd_start_idx:scnd_start_idx+WITHIN_SESSION_SEGMENT_SIZE]
        data_1 = data_1[:WITHIN_SESSION_SEGMENT_SIZE]
    else:
        data_2 = read_epoch_files_by_list(list_2, first_epoch=0, last_epoch=18, boundary_sec=boundary_sec, random_shift=RANDOM_SHIFT)



    # make a dictionary og integer subject id's
    subject_ids = dict()
    id = 0
    for contact in contact_info:
        if not contact['subject'] in subject_ids.keys():
            subject_ids[contact['subject']] = id
            id += 1



    data_mat = np.concatenate((data_1[np.newaxis, :], data_2[np.newaxis, :]))
    pair_cnt = 0
    csac_list, R0_list, R1_list, rdm0_list, rdm1_list = [], [], [], [], []
    #epoch_count = len(contact_list[0]['first']) #len(epoch_subsets)
    epoch_count = 4
    for i_sbst0 in range(epoch_count - (1 if AUTO_OR_CROSS_ACTIVATION=='CROSS' else 0)):
        for i_sbst1 in range(i_sbst0 + 1, epoch_count) if AUTO_OR_CROSS_ACTIVATION=='CROSS' else [i_sbst0] :
            
            #
            # this should be the new content of do_analysis_for_two_epoch_sets:
            epoch_sbst_0 = [int(i) for i in epoch_subsets[i_sbst0].replace('e', '').split('-')]
            epoch_sbst_1 = [int(i) for i in epoch_subsets[i_sbst1].replace('e', '').split('-')]
            
            data_mat_0 = data_mat[:, epoch_sbst_0[0]:epoch_sbst_0[-1] + 1].mean(axis=1)
            data_mat_1 = data_mat[:, epoch_sbst_1[0]:epoch_sbst_1[-1] + 1].mean(axis=1)
            data_mat_pair = np.concatenate((data_mat_0[np.newaxis, :], data_mat_1[np.newaxis, :])).transpose((1, 0, 2, 3))
            
            pre_ignore = 0
            delta_time_sec = 1 / V_SAMP_PER_SEC
            rdm_size = int((data_mat_pair.shape[-1] - pre_ignore) / delta_time_sec)
            
            rdm0_ = calc_rdm(data=data_mat_pair[0], rdm_size=rdm_size, pre_ignore=pre_ignore, delta_time_smple=int(delta_time_sec * V_SAMP_PER_SEC))
            rdm1_ = calc_rdm(data=data_mat_pair[1], rdm_size=rdm_size, pre_ignore=pre_ignore, delta_time_smple=int(delta_time_sec * V_SAMP_PER_SEC))

            SHOW = False
            if SHOW:
                visualize_rdms(np.expand_dims(rdm0_, axis=0), title=' early session', show_hists=False, show=False)
                visualize_rdms(np.expand_dims(rdm1_, axis=0), title=' subsequent session', show_hists=False, show=False)
            
            csac_ = calc_rdm(data_mat_pair.mean(axis=1), rdm_size, pre_ignore, int(delta_time_sec * V_SAMP_PER_SEC), corr_mode='p')
            
            R0_ = relative_codes(rdm0_, first=0, remove_diag=True, normalize=False)
            R1_ = relative_codes(rdm1_, first=0, remove_diag=True, normalize=False)
           #
            

            # rdm_size_, rdm0_, rdm1_, csac_, R0_, R1_, contact_list__ = do_analysis_for_two_epoch_sets(contact_list_, subject_ids, i_sbst0, i_sbst1,
            #                                    V_SAMP_PER_SEC, SHOW_TIME_PER_CONTACT, False, CORR_WINDOW_SEC,
            #                                    AUTO_OR_CROSS_ACTIVATION, CONTACT_SPLIT, PROCESS_QUADS, tfm=None, SHOW=(i_sbst0 + i_sbst1 == 111), ccorr_mode=CROSS_SESSION_CMODE)
            # contact_list_ = services.intersect_lists(contact_list_, contact_list__)
            if pair_cnt == 0:
                rdm_size, rdm0, rdm1, csac, R0, R1 = rdm_size, rdm0_, rdm1_, csac_, R0_, R1_
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



# if SAVE_CONTACT_LIST:
#     with open(CONTACT_SELECTION_FILE_NAME, 'wb') as fd:
#         pickle.dump(contact_list_, fd)
#     # srvices = contact_list_services()
#     # sess_list = services.get_sesseion_list(contact_list)
#     # print(sess_list)

fig = show_region_distribution(contact_info, title='{} contacts , delta=T = {} hrs to {} hrs'.format(len(contact_info), MIN_TGAP, MAX_TGAP))
fig.savefig(os.path.join(os.path.expanduser('~'), 'figs', 'region_distribution.pdf'))
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

    fig = show_relational_codes(R0, R1, show=False)
    rep_pcors = np.zeros((rdm_size, rdm_size))
    for digit_1 in range(rdm_size):
        for digit_2 in range(rdm_size):
            v1, v2 = R0[digit_1], R1[digit_2]
            #rep_pcors[digit_1, digit_2] = pierson (v1, v2, remove_nans=True) #(v1 * v2).sum() / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-18)
            rep_pcors[digit_1, digit_2] = pierson(v1, v2, remove_nans=True, mode=CROSS_SESSION_CMODE)  # (v1 * v2).sum() / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-18)
    fig = visualize_rdms(np.expand_dims(rep_pcors, axis=0), title='full vector relative representation correlation', show_hists=False, show_bars=False, show=False)
    fig = show_corr_diagonals(csac, rep_pcors, show=True)

    # with open(os.path.join(os.path.dirname(CONTACT_SELECTION_FILE_NAME), 'cs_corrs_1'), 'wb') as fd:
    #     pickle.dump(dict({'csac': csac, 'rep_pcorr': rep_pcors}), fd)


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
        if pair_cnt == 15:
            sbgrps = np.array((1, 10, 15, 2, 8, 14, 3, 9, 11, 4, 7, 12, 5, 6, 13)).reshape(5, 3)
            sub_cnt = 5
        if pair_cnt == 6: # for the 4,1 option (reading 2 epocj avgs.) - relative codes with each epoch appear only once
            sbgrps = np.array((1, 6)).reshape(2, 1)
            sub_cnt = 2
        if pair_cnt == 3:
            sbgrps = np.array((1, 2, 3)).reshape(1, 3)
            sub_cnt = 1
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
    visualize_rdms(np.expand_dims(rdm0, axis=0),
                   title='RDM 1st SESSION', show_hists=False, show_bars=False, show=False, ovrd_bar_scale=[-0.1, 0.2], ovrd_heat_scale=[-0.2, 0.3])
    visualize_rdms(np.expand_dims(rdm1, axis=0),
                   title='RDM 2nd SESSION', show_hists=False, show_bars=False, show=False, ovrd_bar_scale=[-0.1, 0.2], ovrd_heat_scale=[-0.2, 0.3])
    visualize_rdms(np.expand_dims(np.mean(csac_list, axis=0), axis=0),
                   title='ACTIVATION CROSS-SESSION AVG CORR', show_hists=False, show_bars=False, show=False, ovrd_bar_scale=[-0.1, 0.2], ovrd_heat_scale=[-0.2, 0.3])
    visualize_rdms(np.expand_dims(np.mean(rep_pcors_list, axis=0), axis=0),
                   title='RELATIVE REPRESANTATION AVG CORR', show_hists=False, show_bars=False, show=False, ovrd_bar_scale=[-0,1, 0.2], ovrd_heat_scale=[-1, 1])
    # visualize_rdms(np.expand_dims(np.std(rep_pcors_list, axis=0), axis=0),
    #                title='RELATIVE REPRESANTATION STDEV CORR', show_hists=False, show_bars=False, show=False, ovrd_bar_scale=[-0,1, 0.2], ovrd_heat_scale=[-0.2, 0.3])
    fig = show_relational_codes(R0_list, R1_list, show=False)
    fig.savefig(os.path.join(os.path.expanduser('~'), 'figs', 'relational codes.pdf'))
    fig = show_corr_diagonals(csac_list, rep_pcors_list, show=True)
    fig.savefig(os.path.join(os.path.expanduser('~'), 'figs', 'corr_diagonals.pdf'))


    # with open(os.path.join(os.path.dirname(CONTACT_SELECTION_FILE_NAME), 'cs_corrs_5_3'), 'wb') as fd:
    #     pickle.dump(dict({'csac': csac_list, 'rep_pcorr': rep_pcors_list}), fd)


    #
    rep_pcors = rep_pcors_list.mean(axis=0)




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



