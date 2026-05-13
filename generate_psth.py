import numpy as np
import pickle
import os
import time

from rdm_tools_new import *
#from permute_digits import digit_permutator
from paths_and_constants import *
from channel_selection_new import select_channels_by_regions#, get_sublist_by_importance
import tqdm
import shutil


#from itertools import combinations






def split_session_to_fake_sessions(contact_list, segment_size=3, pair_idx=1):

    for i_contact, contact in enumerate(contact_list):
        contact['second'] = contact['first'][int(pair_idx*segment_size):int((pair_idx+1)*segment_size)]
        contact['first'] = contact['first'][:segment_size]

    return contact_list

    
    
    
# def do_rdm_analisys(data_1, data_2, epoch_count, output_folder, v_samp_per_sec, contact_info):
    
#     CROSS_SESSION_CMODE = 'p'
#     AUTO_OR_CROSS_ACTIVATION = "CROSS"  # "AUTO": generate session rdm from single epoch set (diagonal = 1); "CROSS": cross-correlate two epoch sets

#     # make a dictionary og integer subject id's
#     subject_ids = dict()
#     id = 0
#     for contact in contact_info:
#         if not contact['subject'] in subject_ids.keys():
#             subject_ids[contact['subject']] = id
#             id += 1

#     data_mat = np.concatenate((data_1[np.newaxis, :], data_2[np.newaxis, :]))
#     pair_cnt = 0
#     csac_list, R0_list, R1_list, rdm0_list, rdm1_list = [], [], [], [], []

#     for i_sbst0 in range(epoch_count - (1 if AUTO_OR_CROSS_ACTIVATION=='CROSS' else 0)):
#         for i_sbst1 in range(i_sbst0 + 1, epoch_count) if AUTO_OR_CROSS_ACTIVATION=='CROSS' else [i_sbst0] :
            
#             #
#             # this should be the new content of do_analysis_for_two_epoch_sets:
#             epoch_sbst_0 = [int(i) for i in epoch_subsets[i_sbst0].replace('e', '').split('-')]
#             epoch_sbst_1 = [int(i) for i in epoch_subsets[i_sbst1].replace('e', '').split('-')]
                        
#             data_mat_0 = data_mat[:, epoch_sbst_0[0]:epoch_sbst_0[-1] + 1].mean(axis=1)
#             data_mat_1 = data_mat[:, epoch_sbst_1[0]:epoch_sbst_1[-1] + 1].mean(axis=1)
#             data_mat_pair = np.concatenate((data_mat_0[np.newaxis, :], data_mat_1[np.newaxis, :])).transpose((1, 0, 2, 3))
            
#             pre_ignore = 0
#             delta_time_sec = 1 / v_samp_per_sec
#             #
#             RDM_TIME_BIN = delta_time_sec
#             #
#             rdm_size = int((data_mat_pair.shape[-1] - pre_ignore) * (delta_time_sec / RDM_TIME_BIN))
            
#             rdm0_ = calc_rdm(data=data_mat_pair[0], rdm_size=rdm_size, pre_ignore=pre_ignore, delta_time_smple=int(delta_time_sec * v_samp_per_sec))
#             rdm1_ = calc_rdm(data=data_mat_pair[1], rdm_size=rdm_size, pre_ignore=pre_ignore, delta_time_smple=int(delta_time_sec * v_samp_per_sec))

#             SHOW = False
#             if SHOW:
#                 visualize_rdms(np.expand_dims(rdm0_, axis=0), title=' early session', show_hists=False, show=False, output_folder=output_folder)
#                 visualize_rdms(np.expand_dims(rdm1_, axis=0), title=' subsequent session', show_hists=False, show=False, output_folder=output_folder)
            
#             csac_ = calc_rdm(data_mat_pair.mean(axis=1), rdm_size, pre_ignore, int(delta_time_sec * v_samp_per_sec), corr_mode='p')
            
#             R0_ = relative_codes(rdm0_, first=0, remove_diag=True, normalize=False)
#             R1_ = relative_codes(rdm1_, first=0, remove_diag=True, normalize=False)
#            #
            

#             # rdm_size_, rdm0_, rdm1_, csac_, R0_, R1_, contact_list__ = do_analysis_for_two_epoch_sets(contact_list_, subject_ids, i_sbst0, i_sbst1,
#             #                                    V_SAMP_PER_SEC, SHOW_TIME_PER_CONTACT, False, CORR_WINDOW_SEC,
#             #                                    AUTO_OR_CROSS_ACTIVATION, CONTACT_SPLIT, PROCESS_QUADS, tfm=None, SHOW=(i_sbst0 + i_sbst1 == 111), ccorr_mode=CROSS_SESSION_CMODE)
#             # contact_list_ = services.intersect_lists(contact_list_, contact_list__)
#             if pair_cnt == 0:
#                 rdm_size, rdm0, rdm1, csac, R0, R1 = np.copy(rdm_size), np.copy(rdm0_), np.copy(rdm1_), np.copy(csac_), np.copy(R0_), np.copy(R1_)
#             else:
#                 if pair_cnt in [9, 14]:
#                     rdm0 += rdm0_
#                     rdm1 += rdm1_
#                     csac += csac_
#                     R0 += R0_
#                     R1 += R1_
#             pair_cnt += 1
#             print('pair no. {},\t  {} {}'.format(pair_cnt, epoch_subsets[i_sbst0], epoch_subsets[i_sbst1]))
#             print(pair_cnt, rdm0_.max(), rdm1_.max(), '\n')
#             csac_list.append(csac_)
#             R0_list.append(R0_)
#             R1_list.append(R1_)
#             rdm0_list.append(rdm0_)
#             rdm1_list.append(rdm1_)

#     #pair_cnt = 1#3
#     rdm0 /= pair_cnt
#     rdm1 /= pair_cnt
#     csac /= pair_cnt
#     R0 /= pair_cnt
#     R1 /= pair_cnt
#     #pair_cnt = 1#15



#     # if SAVE_CONTACT_LIST:
#     #     with open(CONTACT_SELECTION_FILE_NAME, 'wb') as fd:
#     #         pickle.dump(contact_list_, fd)
#     #     # srvices = contact_list_services()
#     #     # sess_list = services.get_sesseion_list(contact_list)
#     #     # print(sess_list)
#     #plt.show()


#     # VISUALIZE RESULTS FOR PLAIN AVERAGING
#     DISPLAY_PLAIN_AVERAGING = False#len(AVG_MANY_EPOCHS) == 0
#     if DISPLAY_PLAIN_AVERAGING:
#         visualize_rdms(np.expand_dims(rdm0, axis=0), title=' early session ', show_hists=False, show_bars=False, show=False, ovrd_bar_scale=[-0.1, 0.2], ovrd_heat_scale=[-0.1*3, 0.2*3])
#         visualize_rdms(np.expand_dims(rdm1, axis=0), title=' subsequent session', show_hists=False, show_bars=False, show=False, ovrd_bar_scale=[-0.1, 0.2], ovrd_heat_scale=[-0.1*3, 0.2*3])
#         #for i_sbst in range(range(2 if AUTO_OR_CROSS_ACTIVATION=='CROSS' else 1):
#         visualize_rdms(np.expand_dims(csac, axis=0),
#                     #title='cross-session correlation of Activity vectors (sbst {})'.format(i_sbst + 1),
#                     title='cross-session correlation of Activity vectors among sessions',
#                     show_hists=False, show_bars=False, show=False)

#         fig = show_relational_codes(R0, R1, show=False)
#         rep_pcors = np.zeros((rdm_size, rdm_size))
#         for digit_1 in range(rdm_size):
#             for digit_2 in range(rdm_size):
#                 v1, v2 = R0[digit_1], R1[digit_2]
#                 #rep_pcors[digit_1, digit_2] = pierson (v1, v2, remove_nans=True) #(v1 * v2).sum() / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-18)
#                 rep_pcors[digit_1, digit_2] = pierson(v1, v2, remove_nans=True, mode=CROSS_SESSION_CMODE)  # (v1 * v2).sum() / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-18)
#         fig = visualize_rdms(np.expand_dims(rep_pcors, axis=0), title='full vector relative representation correlation', 
#                             show_hists=False, show_bars=False, show=False, output_folder=output_folder)
#         fig = show_corr_diagonals(csac, rep_pcors, show=False)

#         # with open(os.path.join(os.path.dirname(CONTACT_SELECTION_FILE_NAME), 'cs_corrs_1'), 'wb') as fd:
#         #     pickle.dump(dict({'csac': csac, 'rep_pcorr': rep_pcors}), fd)


#     #
#     # redu everything with lists
#     DISPLAY_5_3_2 = True#len(AVG_MANY_EPOCHS) >= 6
#     if DISPLAY_5_3_2:
#         csac_list, R0_list, R1_list = np.array(csac_list), np.array(R0_list), np.array(R1_list)
#         # # re-disply activation correlations with error bars
#         # visualize_rdms(np.expand_dims(np.mean(csac_list, axis=0), axis=0),
#         #                title='cross-session correlation of Activity vectors recalculated', show_hists=False, show_bars=False, show=False)
#         # visualize_rdms(np.expand_dims(np.std(csac_list, axis=0), axis=0),
#         #                title='cross-session correlation of Activity vectors recalculated, STDEV', show_hists=False, show_bars=False, show=False)
#         # # no generate rep_pcoers for seperate reps
#         # partial averagings
#         if AUTO_OR_CROSS_ACTIVATION == 'CROSS':
#             if pair_cnt == 15:
#                 sbgrps = np.array((1, 10, 15, 2, 8, 14, 3, 9, 11, 4, 7, 12, 5, 6, 13)).reshape(5, 3)
#                 sub_cnt = 5
#             if pair_cnt == 6: # for the 4,1 option (reading 2 epocj avgs.) - relative codes with each epoch appear only once
#                 sbgrps = np.array((1, 6)).reshape(2, 1)
#                 sub_cnt = 2
#             if pair_cnt == 3:
#                 sbgrps = np.array((1, 2, 3)).reshape(1, 3)
#                 sub_cnt = 1
#             if pair_cnt == 1:
#                 sbgrps = np.array((1, )).reshape(1, 1)
#                 sub_cnt = 1
#         if AUTO_OR_CROSS_ACTIVATION == 'AUTO':
#             sbgrps = np.array((1, 2, 3, 4, 5, 6)).reshape(3, 2)
#             sub_cnt = 3
#         Cact_list_ = np.zeros((sub_cnt, csac_list.shape[1], csac_list.shape[2]))
#         R0_list_, R1_list_ = np.zeros((sub_cnt, R0_list.shape[1], R0_list.shape[2])), np.zeros((sub_cnt, R0_list.shape[1], R0_list.shape[2]))
#         for i_sub in range(sub_cnt):
#             Cact_list_[i_sub] = csac_list[sbgrps[i_sub] - 1].mean(axis=0)
#             R0_list_[i_sub] = R0_list[sbgrps[i_sub] - 1].mean(axis=0)
#             R1_list_[i_sub] = R1_list[sbgrps[i_sub] - 1].mean(axis=0)
#         csac_list, R0_list, R1_list = Cact_list_, R0_list_, R1_list_

#         # #
#         # visualize_rdms(np.expand_dims(rdm0_list[0], axis=0),
#         #             title='rdm 0 epochs 1, 2', show_hists=False, show_bars=False, show=False, 
#         #             ovrd_bar_scale=[-0.1, 0.2], ovrd_heat_scale=[-0.2, 0.3], output_folder=output_folder)
#         # visualize_rdms(np.expand_dims(rdm0_list[9], axis=0),
#         #             title='rdm 0 epochs 3, 4', show_hists=False, show_bars=False, show=False, 
#         #             ovrd_bar_scale=[-0.1, 0.2], ovrd_heat_scale=[-0.2, 0.3], output_folder=output_folder)
#         # visualize_rdms(np.expand_dims(rdm0_list[14], axis=0),
#         #             title='rdm 0 epochs 5, 6', show_hists=False, show_bars=False, show=False, 
#         #             ovrd_bar_scale=[-0.1, 0.2], ovrd_heat_scale=[-0.2, 0.3], output_folder=output_folder)
#         # visualize_rdms(np.expand_dims(rdm1_list[0], axis=0),
#         #             title='rdm 1  epochs 7, 8', show_hists=False, show_bars=False, show=False, 
#         #             ovrd_bar_scale=[-0.1, 0.2], ovrd_heat_scale=[-0.2, 0.3], output_folder=output_folder)
#         # visualize_rdms(np.expand_dims(rdm1_list[9], axis=0),
#         #             title='rdm 1  epochs 9, 10', show_hists=False, show_bars=False, show=False, 
#         #             ovrd_bar_scale=[-0.1, 0.2], ovrd_heat_scale=[-0.2, 0.3], output_folder=output_folder)
#         # visualize_rdms(np.expand_dims(rdm1_list[14], axis=0),
#         #             title='rdm 1  epochs 11, 12', show_hists=False, show_bars=False, show=False, 
#         #             ovrd_bar_scale=[-0.1, 0.2], ovrd_heat_scale=[-0.2, 0.3], output_folder=output_folder)
#         # #
#         rep_pcors_list = np.zeros((sub_cnt, rdm_size, rdm_size))
#         for i_pair in range(sub_cnt):
#             for digit_1 in range(rdm_size):
#                 for digit_2 in range(rdm_size):
#                     v1, v2 = R0_list[i_pair, digit_1], R1_list[i_pair, digit_2]
#                     #rep_pcors_list[i_pair, digit_1, digit_2] = pierson (v1, v2, remove_nans=True) #(v1 * v2).sum() / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-18)
#                     rep_pcors_list[i_pair, digit_1, digit_2] = pierson(v1, v2, remove_nans=True, mode=CROSS_SESSION_CMODE)  # (v1 * v2).sum() / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-18)
#         visualize_rdms(np.expand_dims(rdm0, axis=0),
#                     title='RDM 1st SESSION', show_hists=False, show_bars=False, show=False, 
#                     ovrd_bar_scale=[-0.1, 0.2], ovrd_heat_scale=[-0.2*3, 0.3*3], output_folder=output_folder)
#         visualize_rdms(np.expand_dims(rdm1, axis=0),
#                     title='RDM 2nd SESSION', show_hists=False, show_bars=False, show=False, 
#                     ovrd_bar_scale=[-0.1, 0.2], ovrd_heat_scale=[-0.2*3, 0.3*3], output_folder=output_folder)
#         visualize_rdms(np.expand_dims(np.mean(csac_list, axis=0), axis=0),
#                     title='ACTIVATION CROSS-SESSION AVG CORR', show_hists=False, show_bars=False, show=False, 
#                     ovrd_bar_scale=[-0.1, 0.2], ovrd_heat_scale=[-0.2*2, 0.3*2], output_folder=output_folder)
#         visualize_rdms(np.expand_dims(np.mean(rep_pcors_list, axis=0), axis=0),
#                     title='RELATIVE REPRESANTATION AVG CORR', show_hists=False, show_bars=False, show=False, 
#                     ovrd_bar_scale=[-0,1, 0.2], ovrd_heat_scale=[-1, 1], output_folder=output_folder)
#         # visualize_rdms(np.expand_dims(np.std(rep_pcors_list, axis=0), axis=0),
#         #                title='RELATIVE REPRESANTATION STDEV CORR', show_hists=False, show_bars=False, show=False, ovrd_bar_scale=[-0,1, 0.2], ovrd_heat_scale=[-0.2, 0.3])
#         fig = show_relational_codes(R0_list, R1_list, show=False)
#         #fig.savefig(os.path.join(os.path.expanduser('~'), 'figs', 'relational codes.pdf'))
#         mysavefig(fig=fig, subfolder=output_folder, name='relational codes')
#         saveok, savecnt = False, 5
#         while not saveok:
#             try:
#                 mysavedata(subfolder=output_folder, name='relational codes', data=dict({'R0_list': R0_list, 'R1_list': R1_list}))
#                 saveok = True
#             except:
#                 time.sleep(1)
#                 savecnt -= 1
#                 assert savecnt > 0
#         fig = show_corr_diagonals(csac_list, rep_pcors_list, show=False)
#         #fig.savefig(os.path.join(os.path.expanduser('~'), 'figs', 'corr_diagonals.pdf'))
#         mysavefig(fig=fig, subfolder=output_folder, name='corr diagonals')
#         saveok, savecnt = False, 5
#         while not saveok:
#             try:
#                 mysavedata(subfolder=output_folder, name='diagonals', data=dict({'csac_list': csac_list, 'rep_pcors_list': rep_pcors_list}))
#                 saveok = True
#             except:
#                 time.sleep(1)
#                 savecnt -= 1
#                 assert savecnt > 0


#     # with open(os.path.join(os.path.dirname(CONTACT_SELECTION_FILE_NAME), 'cs_corrs_5_3'), 'wb') as fd:
#     #     pickle.dump(dict({'csac': csac_list, 'rep_pcorr': rep_pcors_list}), fd)


#     #
#     rep_pcors = rep_pcors_list.mean(axis=0)



#     # statistics
#     tmp0, tmp1, tmpcs, tmprel = rdm0[1:-1, 1:-1], rdm1[1:-1, 1:-1], csac[1:-1, 1:-1], rep_pcors[1:-1, 1:-1]
#     off_diag = np.concatenate((tmp0[~np.eye(tmp0.shape[0],dtype=bool)], tmp1[~np.eye(tmp1.shape[0],dtype=bool)]))
#     on_diag = np.concatenate((np.diag(tmp0), np.diag(tmp1)))
#     print('\n\t\t\t\t\t\tdiagonal\t\t\toff diag')
#     print('within session:\t\t {:4.2f} (dev {:4.2f}) \t{:4.2f} (dev {:4.2f}) '.format(on_diag.mean(), on_diag.std(), off_diag.mean(), off_diag.std()))
#     off_diag = tmpcs[~np.eye(tmpcs.shape[0],dtype=bool)]
#     on_diag = np.diag(csac)
#     print('across sessions:\t {:4.2f} (dev {:4.2f}) \t{:4.2f} (dev {:4.2f}) '.format(on_diag.mean(), on_diag.std(), off_diag.mean(), off_diag.std()))
#     off_diag = tmprel[~np.eye(tmprel.shape[0],dtype=bool)]
#     on_diag = np.diag(tmprel)
#     print('relative :\t\t\t {:4.2f} (dev {:4.2f}) \t{:4.2f} (dev {:4.2f}) '.format(on_diag.mean(), on_diag.std(), off_diag.mean(), off_diag.std()))
    
    
#     return rdm0, rdm1, np.mean(rep_pcors_list, axis=0)






def get_contact_subset(data_1C, data_2C, data_1R, data_2R, contact_info, boundary_sec, USE='ALL', SPLIT='ALL'):

    # now find "responsive" contacts
    #if USE != 'ALL':
    psth_by_cntct = (data_1C.mean(axis=0) + data_2C.mean(axis=0)) / 2
    #rr = np.median(psth_by_cntct[:, 1:], axis=-1) / psth_by_cntct[:, 0]
    mask_0 = ((boundary_sec >= -1) * (boundary_sec < 0))[:-1]
    mask_2_5 = ((boundary_sec >= 2) * (boundary_sec < 5))[:-1]
    mask_0 = ((boundary_sec >= -0.5) * (boundary_sec < 0))[:-1]
    mask_2_5 = ((boundary_sec >= 0.25) * (boundary_sec < 0.75))[:-1]
    rr = psth_by_cntct[:, mask_2_5].mean(axis=-1) / psth_by_cntct[:, mask_0].mean(axis=-1)
    if USE == 'ALL':
        use_mask =np.ones(rr.shape, dtype='bool')
    if USE == 'NON_RESP':
        rr = np.maximum(rr, 1/rr)
        thd = np.quantile(rr, 1/3)
        use_mask = rr < thd
    if USE == 'RESP':
        thd = np.quantile(rr, 2/3)
        thd = np.sort(rr)[::-1][min(rr.size - 1, 200)]
        use_mask = rr > thd
    if USE == 'HIGH_RESP':
        thd = np.quantile(rr, 0.975)
        thd = max(np.sort(rr)[::-1][min(rr.size - 1, max(25, int(rr.size / 3)))], 1.1)
        use_mask = rr > thd
        # #
        # thd_l = np.quantile(rr, 0.95)
        # thd_h = np.quantile(rr, 1.0)
        # use_mask = (rr >= thd_l) * (rr < thd_h)
        # #
    data_1C_ = data_1C[:, use_mask]
    data_2C_ = data_2C[:, use_mask]
    data_1R_ = data_1R[:, use_mask]
    data_2R_ = data_2R[:, use_mask]
    contact_info_ = [contact_info[i] for i in np.argwhere(use_mask).flatten().astype(int)]
    
    # split
    if SPLIT == 'ODD':
        split_mask = ((-1) ** np.arange(len(contact_info_))) > 0
    if SPLIT == 'EVEN':
        split_mask = ((-1) ** np.arange(len(contact_info_))) < 0
    if SPLIT != 'ALL':
        data_1C_ = data_1C_[:, split_mask]
        data_2C_ = data_2C_[:, split_mask]
        data_1R_ = data_1R_[:, split_mask]
        data_2R_ = data_2R_[:, split_mask]
        contact_info_ = [contact_info_[i] for i in np.argwhere(split_mask).flatten().astype(int)]
    
    # re-generate mask
    mask = np.zeros(use_mask.shape, dtype=bool)
    use_idxes = np.argwhere(use_mask).flatten() if SPLIT=='ALL' else np.argwhere(use_mask).flatten()[split_mask]
    mask[use_idxes] = True
     
    return data_1C_, data_2C_, data_1R_, data_2R_, contact_info_, mask



    

if __name__ == '__main__':
    
    print('\n\n***************\n\nstarting\n\n*********************\n\n')
    
    V_SAMP_PER_SEC = 10
    V_SAMP_PER_SEC_RDM = 1
    #AUTO_OR_CROSS_ACTIVATION = "CROSS"  # "AUTO": generate session rdm from single epoch set (diagonal = 1); "CROSS": cross-correlate two epoch sets
    MIN_TGAP, MAX_TGAP = 24, 480#144, 336#24, 48
    CONTACT_SPLIT = None # None: use all, 0: even contacts only, 1: odd contacts only
    #event_type = 'CNTDWN' # one of: 'CNTDWN', 'RECALL', 'DSTRCT', 'REST'
    #
    RAW_EPOCH_AVG = 1

    NUM_SESSIONS = 3
    assert NUM_SESSIONS in [1, 2, 3]
    WITHIN_SESSION_PROCESS = NUM_SESSIONS == 1
    THIRD_SESSION = NUM_SESSIONS == 3
    if WITHIN_SESSION_PROCESS:
        MIN_TGAP, MAX_TGAP = 1, 1000
        #WITHIN_SESSION_SEGMENT_SIZE, WITHIN_SESSION_PAIR_IDX = 8, 1# 6, 1
        #WITHIN_SESSION_SEGMENT_SIZE, WITHIN_SESSION_PAIR_IDX = 3, 1
        #EPOCHS_TO_READ = 18
    else:
        #EPOCHS_TO_READ = 12
        #WITHIN_SESSION_SEGMENT_SIZE = 6#len(AVG_MANY_EPOCHS)
        pass
    EPOCHS_TO_READ = 12 
    
    PROCESS_RECALL = False

    # SAVE_CONTACT_LIST = False
    # USE_CONTACT_SELECTION_FROM_FILE = True
    # #CONTACT_SELECTION_FILE_NAME = 'C:/Users/menas/OneDrive/Desktop/openneuro/temp/contact_list_cntdwn_{}_{}'.format(MIN_TGAP, MAX_TGAP)
    # CONTACT_SELECTION_FILE_NAME = 'C:/Users/menas/OneDrive/Desktop/openneuro/temp/contact_list_cntdwn_3_5'        


    data_availability_obj = data_availability()
    epoch_subsets = [[i*RAW_EPOCH_AVG, (i+1)*RAW_EPOCH_AVG-1] for i in range(int(EPOCHS_TO_READ / RAW_EPOCH_AVG))]
    epoch_subsets = ['e{}-e{}'.format(i1, i2) for (i1, i2) in epoch_subsets]
    
    
    # prepare contact list
    # stage 1: find suitable contacts
    
    list_1C, list_2C, list_3C = data_availability_obj.get_suitable_epoch_files_and_contacts(min_timegap_hrs=MIN_TGAP, max_timegap_hrs=MAX_TGAP,
                                                                                           proc_type='gamma_c_60_160', event_list=['CNTDWN'], 
                                                                                           num_epochs=EPOCHS_TO_READ, enforce_first=True, 
                                                                                           single_session=WITHIN_SESSION_PROCESS, third_session=THIRD_SESSION)
    if NUM_SESSIONS < 3:
        list_3C = list_2C # !!! PATCH !!!
    (list_1C, list_2C, list_3C) = data_availability_obj.intersect_epoch_files_and_contact_lists([list_1C, list_2C, list_3C])

    # if PROCESS_RECALL: 
    #     list_1R, list_2R = data_availability_obj.get_suitable_epoch_files_and_contacts(min_timegap_hrs=MIN_TGAP, max_timegap_hrs=MAX_TGAP,
    #                                                                                         proc_type='gamma_c_60_160', event_list=['RECALL'], 
    #                                                                                         num_epochs=EPOCHS_TO_READ, enforce_first=True, single_session=WITHIN_SESSION_PROCESS)
    #     (list_1C, list_2C, list_1R, list_2R) = data_availability_obj.intersect_epoch_files_and_contact_lists([list_1C, list_2C, list_1R, list_2R])
    

    print('A')
    
    #list_1C, list_2C, list_1R, list_2R = list_1C[:1], list_2C[:1], list_1R[:1], list_2R[:1]
    
    contact_info = data_availability_obj.get_contact_info(list_1C)
    print('B')
    #
    SELECT_BY_REGION = True
    if SELECT_BY_REGION:
        responsive_list = ['cuneus', 'pericalcarine', 'postcentral', 'precentral', 'lingual',
                           'superiorparietal', 'inferiortemporal', 'middletemporal', 'fusiform', 'lateraloccipital']
        # early_list = ['pericalcarine-R', 'cuneus-R', 'lingual-R', 'lateraloccipital-R', 'pericalcarine-L', 'cuneus-L', 'lingual-L', 'lateraloccipital-L']
        # mid_list = ['fusiform-R', 'inferiortemporal-R', 'parahippocampal-R', 'fusiform-L', 'inferiortemporal-L', 'parahippocampal-L']
        # late_list = ['precuneus-R', 'superiorparietal-R', 'precuneus-L', 'superiorparietal-L']
        region_list = ['transversetemporal-R', 'transversetemporal-L']#['superiorparietal-R', 'superiorparietal-L']#early_list + mid_list
        # _, contact_info = select_channels_by_regions(contact_info=contact_info, region_list=['fusiform-L', 'fusiform-R'])
        _, contact_info = select_channels_by_regions(contact_info=contact_info, region_list=region_list)
        contact_info_imp = contact_info
    else:
        contact_info_imp = contact_info
    list_1C, _ = data_availability_obj.intersect_contact_list_and_contact_info(contact_list=list_1C, contact_info=contact_info_imp)
    list_2C, _ = data_availability_obj.intersect_contact_list_and_contact_info(contact_list=list_2C, contact_info=contact_info_imp)
    list_3C, _ = data_availability_obj.intersect_contact_list_and_contact_info(contact_list=list_3C, contact_info=contact_info_imp)
    # list_1R, _ = data_availability_obj.intersect_contact_list_and_contact_info(contact_list=list_1R, contact_info=contact_info_imp)
    # list_2R, contact_info = data_availability_obj.intersect_contact_list_and_contact_info(contact_list=list_2R, contact_info=contact_info_imp)
    #
    
    boundary_sec = np.arange(start=-5, stop=12+1e-6+3, step=0.02)#1/V_SAMP_PER_SEC)

    # read data
    data_1C, cntct_mask = read_epoch_files_by_list(list_1C, first_epoch=0, last_epoch=EPOCHS_TO_READ, norm_per_epoch=True,
                                                   boundary_sec=boundary_sec, random_shift=False, norm_baseline=[-5, 15])#[-0.5, -0.05])#
    print('C')

    # if WITHIN_SESSION_PROCESS:
    #     scnd_start_idx = WITHIN_SESSION_SEGMENT_SIZE * WITHIN_SESSION_PAIR_IDX
    #     data_2C = data_1C[scnd_start_idx:scnd_start_idx+WITHIN_SESSION_SEGMENT_SIZE]
    #     data_1C = data_1C[:WITHIN_SESSION_SEGMENT_SIZE]
    #     # data_2R = data_1R[scnd_start_idx:scnd_start_idx+WITHIN_SESSION_SEGMENT_SIZE]
    #     # data_1R = data_1R[:WITHIN_SESSION_SEGMENT_SIZE]
    # else:
    #     data_2C, cntct_mask_2 = read_epoch_files_by_list(list_2C, first_epoch=0, last_epoch=EPOCHS_TO_READ, norm_per_epoch=True,
    #                                                      boundary_sec=boundary_sec, random_shift=False, norm_baseline=[5, 15])#[-0.5, -0.05])#
    #     # data_2R, _ = read_epoch_files_by_list(list_2R, first_epoch=0, last_epoch=18, norm_per_epoch=True,
    #     #                                       boundary_sec=boundary_sec, random_shift=True, verbose=False, norm_baseline=[5, 15])#[-0.5, -0.05])#
    #     cntct_mask = cntct_mask * cntct_mask_2

    if NUM_SESSIONS == 1:
        data_2C = data_1C
        data_3C = data_1C
    if NUM_SESSIONS >= 2:
        data_2C, cntct_mask_2 = read_epoch_files_by_list(list_2C, first_epoch=0, last_epoch=EPOCHS_TO_READ, norm_per_epoch=True,
                                                         boundary_sec=boundary_sec, random_shift=False, norm_baseline=[5, 15])#[-0.5, -0.05])#
        cntct_mask = cntct_mask * cntct_mask_2
        data_3C = data_2C
    if NUM_SESSIONS == 3:
        data_3C, cntct_mask_3 = read_epoch_files_by_list(list_3C, first_epoch=0, last_epoch=EPOCHS_TO_READ, norm_per_epoch=True,
                                                         boundary_sec=boundary_sec, random_shift=False, norm_baseline=[5, 15])#[-0.5, -0.05])#
        cntct_mask = cntct_mask * cntct_mask_3

    data_1C = data_1C[:, cntct_mask, :]
    data_2C = data_2C[:, cntct_mask, :]
    data_3C = data_3C[:, cntct_mask, :]
    # data_1R = data_1R[:, cntct_mask, :]
    # data_2R = data_2R[:, cntct_mask, :]
    contact_info = [contact_info[i] for i in np.argwhere(cntct_mask).flatten().astype(int)]
    
        
    
    for USE in ['ALL', 'NON_RESP', 'RESP', 'HIGH_RESP']:
        for SPLIT in ['ALL', 'ODD', 'EVEN']:

            # erase all old folders
            for event in ['CNTDWN', 'RECALL']:
                folder_path = os.path.join(os.path.expanduser('~'), 'figs', '{}_USE_{}_SPLIT_{}'.format(event, USE, SPLIT))
                shutil.rmtree(folder_path, ignore_errors=True)
            
            # if (USE != 'NON_RESP') or (SPLIT != 'ODD'):
            #     continue
            if SPLIT != 'ALL':
                continue
            if USE not in ['ALL', 'HIGH_RESP']:
                continue
            
            if True:
                # data_1C_, data_2C_, data_1R_, data_2R_, contact_info_, _ = \
                #     get_contact_subset(data_1C, data_2C, data_1R, data_2R, contact_info, boundary_sec=boundary_sec, USE=USE, SPLIT=SPLIT)
                data_1C_, data_2C_, _, _, contact_info_, _ = \
                    get_contact_subset(data_1C, data_2C, data_1C, data_2C, contact_info, boundary_sec=boundary_sec, USE=USE, SPLIT=SPLIT)
                #data_1C_, data_2C_ = data_1C_[:, 20:], data_2C_[:, 20:]
                print(data_1C_.shape, data_2C_.shape, len(contact_info_))
            #
            
            if (USE != 'ALL') and (len(contact_info_) < 10):
                continue
             

            for event in ['CNTDWN', 'RECALL']:

                if event == 'CNTDWN':
                    data_1_, data_2_ = data_1C_, data_2C_
                if event == 'RECALL':
                    if PROCESS_RECALL:
                        data_1_, data_2_ = data_1R_, data_2R_
                    else:
                        continue
                    
                output_folder = '{}_USE_{}_SPLIT_{}'.format(event, USE, SPLIT)
                print('\n\n\nworking on', output_folder)
                # # output_folder = 'default'
                # psth_by_cntct = (data_1_.mean(axis=0) + data_2_.mean(axis=0)) / 2
                # psth_all = psth_by_cntct.mean(axis=0)
                # psth_all_sem = psth_by_cntct.std(axis=0) / np.sqrt(psth_by_cntct.shape[0])
                psth_by_cntct_1 = data_1_.mean(axis=0)
                psth_all_1 = psth_by_cntct_1.mean(axis=0)
                psth_all_sem_1 = psth_by_cntct_1.std(axis=0) / np.sqrt(psth_by_cntct_1.shape[0])
                psth_by_cntct_2 = data_2_.mean(axis=0)
                psth_all_2 = psth_by_cntct_2.mean(axis=0)
                psth_all_sem_2 = psth_by_cntct_2.std(axis=0) / np.sqrt(psth_by_cntct_2.shape[0])
                fig, ax = plt.subplots(1, 1)
                # ax.bar((boundary_sec[:-1] + boundary_sec[1:]) / 2, psth_all, width=1/V_SAMP_PER_SEC)
                # ax.bar((boundary_sec[:-1] + boundary_sec[1:]) / 2, 2 * psth_all_sem, bottom=psth_all - psth_all_sem, width=0.5/V_SAMP_PER_SEC, color='k')
                ax.plot((boundary_sec[:-1] + boundary_sec[1:]) / 2, np.log(psth_all_1), label='sess 1')
                ax.plot((boundary_sec[:-1] + boundary_sec[1:]) / 2, np.log(psth_all_2), label='sess 2')
                ax.set_ylim((-0.1, 0.3))
                ax.grid(True)
                ax.set_title('PSTH   ({} contacts)'.format(psth_by_cntct_1.shape[0]))
                mysavefig(name='PSTH', subfolder=output_folder, fig=fig)
                mysavedata(subfolder=output_folder, name='PSTH', data=dict({'boundary_sec': boundary_sec,
                                                                             'psth_1': psth_all_1, 'psth_sem_1': psth_all_sem_1,
                                                                             'psth_2': psth_all_2, 'psth_sem_2': psth_all_sem_2}))
                
                

                fig = show_region_distribution(contact_info_, title='{} contacts , delta=T = {} hrs to {} hrs'.format(len(contact_info_), MIN_TGAP, MAX_TGAP))
                #fig.savefig(os.path.join(os.path.expanduser('~'), 'figs', 'region_distribution.pdf'))
                mysavefig(name='region_distribution', subfolder=output_folder, fig=fig)
                with open(os.path.join(os.path.expanduser('~'), 'figs', output_folder, 'contact_data'), 'wb') as fd:
                    pickle.dump({'contact_info': contact_info_}, fd)
                

                # if COMB_ANALYSIS:
                #     from correlation_tools_comb import my_flow

                #     # s0 = 0
                #     # erdm_list, srdm_list, act_list = [], [], []
                #     use = [0, 10]
                #     keep_margin = [use[0], use[-1] + 1]
                #     fig_erdm, ax_erdm = plt.subplots(1, 1)
                #     fig_erdm.suptitle('RDM (averging over epochs) correlations, Span from {} sec to {} sec'.format(use[0], use[-1]))
                #     fig_srdm, ax_srdm = plt.subplots(1, 1)
                #     fig_srdm.suptitle('RDM (full session) correlations, Span from {} sec to {} sec'.format(use[0], use[-1]))
                #     fig_act, ax_act = plt.subplots(1, 1)
                #     fig_act.suptitle('Activation Vector (full session) correlations, Span from {} sec to {} sec'.format(use[0], use[-1]))

                #     #
                #     zoom_mask = (boundary_sec[:-1] >= 3) * (boundary_sec[:-1] < 4)
                #     cntct_max = np.concatenate((data_1C_[:, :, zoom_mask].mean(axis=0).max(axis=-1)[:, np.newaxis], 
                #                                 data_2C_[:, :, zoom_mask].mean(axis=0).max(axis=-1)[:, np.newaxis]), axis=1).max(axis=1)
                #     power_cntcts = np.argsort(cntct_max)[-5:]
                #     for i_sess, Data in enumerate([data_1C_, data_2C_]):
                #         fig_resp, ax_resp = plt.subplots(1, 1, figsize=(12, 8))
                #         fig_resp.suptitle('avg. gamma response session'.format(i_sess))
                #         ax_resp.grid(True)
                #         ax_resp.set_ylim(0.5, 1.5)
                #         #ax_resp.plot(boundary_sec[:-1][zoom_mask], Data[:, :, zoom_mask].mean(axis=0).T)
                #         for i_ctct in range(Data.shape[1]):
                #             if i_ctct in power_cntcts:
                #                 ax_resp.plot(boundary_sec[:-1][zoom_mask], Data[:, i_ctct, zoom_mask].mean(axis=0), linewidth=3)
                #             else:
                #                 ax_resp.plot(boundary_sec[:-1][zoom_mask], Data[:, i_ctct, zoom_mask].mean(axis=0))
                #         mysavefig(name='avg_resp_sees_' + str(i_sess), fig=fig_resp)
                #     #
                #     erdm, srdm, act, epoch_rdm_set, session_rdm_set, act_set = my_flow(data_1C_[:16],  data_2C_[:16], 
                #                                                                     boundary_sec=boundary_sec, use=use,
                #                                                                     keep_margin=keep_margin, add_margin=0, shift_step=0.02, interval=0.1)

                #     _, _, _, epoch_rdm_set_1, session_rdm_set_1, act_set_1 = my_flow(data_1C_[:16],  data_2C_[:16], 
                #                                                                     boundary_sec=boundary_sec, use=use,
                #                                                                     keep_margin=keep_margin, add_margin=0, shift_step=1, interval=1)
                #     # erdm_list.append(erdm)
                #     # srdm_list.append(srdm)
                #     # act_list.append(act)
                #     for sess in range(2):
                #         ax_erdm.plot(erdm[sess], label='shift {:4.2f}, sess {}'.format(0.02, sess))
                #         ax_srdm.plot(srdm[sess], label='shift {:4.2f}, sess {}'.format(0.02, sess))
                #         ax_act.plot(act[sess], label='shift {:4.2f}, sess {}'.format(0.02, sess))
                #     for ax in [ax_erdm, ax_srdm, ax_act]:
                #         ax.grid(True)
                #         ax.set_ylim(-0.2, 1)
                #         ax.legend()
                #     mysavefig(name='ERDM', fig=fig_erdm)
                #     mysavefig(name='SRDM', fig=fig_srdm)
                #     mysavefig(name='ACT', fig=fig_act)


                #     fig_erdm_erdm, ax_ee = plt.subplots(1, 1)
                #     fig_erdm_erdm.suptitle('correlations between 1sec and 0.1 sec EPOCH RDMs')
                #     fig_srdm_srdm, ax_ss = plt.subplots(1, 1)
                #     fig_srdm_srdm.suptitle('correlations between 1sec and 0.1 sec SESSION RDMs')
                #     fig_act_act, ax_aa = plt.subplots(1, 1)
                #     fig_srdm_srdm.suptitle('correlations between 1sec and 0.1 sec ACTIVATIONS')
                #     [ax.set_ylim((0, 1)) for ax in [ax_ee, ax_ss, ax_aa]]
                #     [ax.grid(True) for ax in [ax_ee, ax_ss, ax_aa]]
                    
                #     erdm_sz = epoch_rdm_set_1[0][0].shape[0]
                #     srdm_sz = session_rdm_set_1[0][0].shape[0]
                #     erdm_mask = ~np.eye(erdm_sz).astype(bool)
                #     srdm_mask = ~np.eye(srdm_sz).astype(bool)
                #     for i_sess in range(2):
                #         erdm_erdm, srdm_srdm, act_act = [], [], []
                #         for i_tshift in range(len(epoch_rdm_set[0])):
                #             erdm_erdm.append(pierson(epoch_rdm_set_1[i_sess][0][erdm_mask], epoch_rdm_set[i_sess][i_tshift][erdm_mask]))
                #             srdm_srdm.append(pierson(session_rdm_set_1[i_sess][0][srdm_mask], session_rdm_set[i_sess][i_tshift][srdm_mask]))
                #             act_act.append(pierson(act_set_1[i_sess][0], act_set[i_sess][i_tshift]))
                #         ax_ee.plot(erdm_erdm, label='sess {}'.format(i_sess))
                #         ax_ss.plot(srdm_srdm, label='sess {}'.format(i_sess))
                #         ax_aa.plot(act_act, label='sess {}'.format(i_sess))
                #     [ax.legend() for ax in [ax_ee, ax_ss, ax_aa]]
                #     mysavefig(name='ERDM-ERDM', fig=fig_erdm_erdm)
                #     mysavefig(name='SRDM-SRDM', fig=fig_srdm_srdm)
                #     mysavefig(name='ACT-ACT', fig=fig_act_act)
                    
                    
                        
                    
            
            



