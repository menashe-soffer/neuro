
from rdm_tools_new import *
from permute_digits import digit_permutator



def split_session_to_fake_sessions(contact_list, segment_size=3, pair_idx=1):

    for i_contact, contact in enumerate(contact_list):
        contact['second'] = contact['first'][int(pair_idx*segment_size):int((pair_idx+1)*segment_size)]
        contact['first'] = contact['first'][:segment_size]

    return contact_list

    
    
    
def do_rdm_analisys(data_1, data_2, epoch_count, output_folder, v_samp_per_sec, contact_info):
    
    CROSS_SESSION_CMODE = 'p'
    AUTO_OR_CROSS_ACTIVATION = "CROSS"  # "AUTO": generate session rdm from single epoch set (diagonal = 1); "CROSS": cross-correlate two epoch sets

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
            delta_time_sec = 1 / v_samp_per_sec
            rdm_size = int((data_mat_pair.shape[-1] - pre_ignore) * delta_time_sec)
            
            rdm0_ = calc_rdm(data=data_mat_pair[0], rdm_size=rdm_size, pre_ignore=pre_ignore, delta_time_smple=int(delta_time_sec * v_samp_per_sec))
            rdm1_ = calc_rdm(data=data_mat_pair[1], rdm_size=rdm_size, pre_ignore=pre_ignore, delta_time_smple=int(delta_time_sec * v_samp_per_sec))

            SHOW = False
            if SHOW:
                visualize_rdms(np.expand_dims(rdm0_, axis=0), title=' early session', show_hists=False, show=False, output_folder=output_folder)
                visualize_rdms(np.expand_dims(rdm1_, axis=0), title=' subsequent session', show_hists=False, show=False, output_folder=output_folder)
            
            csac_ = calc_rdm(data_mat_pair.mean(axis=1), rdm_size, pre_ignore, int(delta_time_sec * v_samp_per_sec), corr_mode='p')
            
            R0_ = relative_codes(rdm0_, first=0, remove_diag=True, normalize=False)
            R1_ = relative_codes(rdm1_, first=0, remove_diag=True, normalize=False)
           #
            

            # rdm_size_, rdm0_, rdm1_, csac_, R0_, R1_, contact_list__ = do_analysis_for_two_epoch_sets(contact_list_, subject_ids, i_sbst0, i_sbst1,
            #                                    V_SAMP_PER_SEC, SHOW_TIME_PER_CONTACT, False, CORR_WINDOW_SEC,
            #                                    AUTO_OR_CROSS_ACTIVATION, CONTACT_SPLIT, PROCESS_QUADS, tfm=None, SHOW=(i_sbst0 + i_sbst1 == 111), ccorr_mode=CROSS_SESSION_CMODE)
            # contact_list_ = services.intersect_lists(contact_list_, contact_list__)
            if pair_cnt == 0:
                rdm_size, rdm0, rdm1, csac, R0, R1 = np.copy(rdm_size), np.copy(rdm0_), np.copy(rdm1_), np.copy(csac_), np.copy(R0_), np.copy(R1_)
            else:
                if pair_cnt in [9, 14]:
                    rdm0 += rdm0_
                    rdm1 += rdm1_
                    csac += csac_
                    R0 += R0_
                    R1 += R1_
            pair_cnt += 1
            print('pair no. {},\t  {} {}'.format(pair_cnt, epoch_subsets[i_sbst0], epoch_subsets[i_sbst1]))
            print(pair_cnt, rdm0_.max(), rdm1_.max(), '\n')
            csac_list.append(csac_)
            R0_list.append(R0_)
            R1_list.append(R1_)
            rdm0_list.append(rdm0_)
            rdm1_list.append(rdm1_)

    pair_cnt = 3
    rdm0 /= pair_cnt
    rdm1 /= pair_cnt
    csac /= pair_cnt
    R0 /= pair_cnt
    R1 /= pair_cnt
    pair_cnt = 15



    # if SAVE_CONTACT_LIST:
    #     with open(CONTACT_SELECTION_FILE_NAME, 'wb') as fd:
    #         pickle.dump(contact_list_, fd)
    #     # srvices = contact_list_services()
    #     # sess_list = services.get_sesseion_list(contact_list)
    #     # print(sess_list)
    #plt.show()


    # VISUALIZE RESULTS FOR PLAIN AVERAGING
    DISPLAY_PLAIN_AVERAGING = False#len(AVG_MANY_EPOCHS) == 0
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
        fig = visualize_rdms(np.expand_dims(rep_pcors, axis=0), title='full vector relative representation correlation', 
                            show_hists=False, show_bars=False, show=False, output_folder=output_folder)
        fig = show_corr_diagonals(csac, rep_pcors, show=True)

        # with open(os.path.join(os.path.dirname(CONTACT_SELECTION_FILE_NAME), 'cs_corrs_1'), 'wb') as fd:
        #     pickle.dump(dict({'csac': csac, 'rep_pcorr': rep_pcors}), fd)


    #
    # redu everything with lists
    DISPLAY_5_3_2 = True#len(AVG_MANY_EPOCHS) >= 6
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
                sub_cnt = 5-4
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

        #
        visualize_rdms(np.expand_dims(rdm0_list[0], axis=0),
                    title='rdm 0 epochs 1, 2', show_hists=False, show_bars=False, show=False, 
                    ovrd_bar_scale=[-0.1, 0.2], ovrd_heat_scale=[-0.2, 0.3], output_folder=output_folder)
        visualize_rdms(np.expand_dims(rdm0_list[9], axis=0),
                    title='rdm 0 epochs 3, 4', show_hists=False, show_bars=False, show=False, 
                    ovrd_bar_scale=[-0.1, 0.2], ovrd_heat_scale=[-0.2, 0.3], output_folder=output_folder)
        visualize_rdms(np.expand_dims(rdm0_list[14], axis=0),
                    title='rdm 0 epochs 5, 6', show_hists=False, show_bars=False, show=False, 
                    ovrd_bar_scale=[-0.1, 0.2], ovrd_heat_scale=[-0.2, 0.3], output_folder=output_folder)
        visualize_rdms(np.expand_dims(rdm1_list[0], axis=0),
                    title='rdm 1  epochs 7, 8', show_hists=False, show_bars=False, show=False, 
                    ovrd_bar_scale=[-0.1, 0.2], ovrd_heat_scale=[-0.2, 0.3], output_folder=output_folder)
        visualize_rdms(np.expand_dims(rdm1_list[9], axis=0),
                    title='rdm 1  epochs 9, 10', show_hists=False, show_bars=False, show=False, 
                    ovrd_bar_scale=[-0.1, 0.2], ovrd_heat_scale=[-0.2, 0.3], output_folder=output_folder)
        visualize_rdms(np.expand_dims(rdm1_list[14], axis=0),
                    title='rdm 1  epochs 11, 12', show_hists=False, show_bars=False, show=False, 
                    ovrd_bar_scale=[-0.1, 0.2], ovrd_heat_scale=[-0.2, 0.3], output_folder=output_folder)
        #
        rep_pcors_list = np.zeros((sub_cnt, rdm_size, rdm_size))
        for i_pair in range(sub_cnt):
            for digit_1 in range(rdm_size):
                for digit_2 in range(rdm_size):
                    v1, v2 = R0_list[i_pair, digit_1], R1_list[i_pair, digit_2]
                    #rep_pcors_list[i_pair, digit_1, digit_2] = pierson (v1, v2, remove_nans=True) #(v1 * v2).sum() / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-18)
                    rep_pcors_list[i_pair, digit_1, digit_2] = pierson(v1, v2, remove_nans=True, mode=CROSS_SESSION_CMODE)  # (v1 * v2).sum() / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-18)
        visualize_rdms(np.expand_dims(rdm0, axis=0),
                    title='RDM 1st SESSION', show_hists=False, show_bars=False, show=False, 
                    ovrd_bar_scale=[-0.1, 0.2], ovrd_heat_scale=[-0.2, 0.3], output_folder=output_folder)
        visualize_rdms(np.expand_dims(rdm1, axis=0),
                    title='RDM 2nd SESSION', show_hists=False, show_bars=False, show=False, 
                    ovrd_bar_scale=[-0.1, 0.2], ovrd_heat_scale=[-0.2, 0.3], output_folder=output_folder)
        visualize_rdms(np.expand_dims(np.mean(csac_list, axis=0), axis=0),
                    title='ACTIVATION CROSS-SESSION AVG CORR', show_hists=False, show_bars=False, show=False, 
                    ovrd_bar_scale=[-0.1, 0.2], ovrd_heat_scale=[-0.2, 0.3], output_folder=output_folder)
        visualize_rdms(np.expand_dims(np.mean(rep_pcors_list, axis=0), axis=0),
                    title='RELATIVE REPRESANTATION AVG CORR', show_hists=False, show_bars=False, show=False, 
                    ovrd_bar_scale=[-0,1, 0.2], ovrd_heat_scale=[-1, 1], output_folder=output_folder)
        # visualize_rdms(np.expand_dims(np.std(rep_pcors_list, axis=0), axis=0),
        #                title='RELATIVE REPRESANTATION STDEV CORR', show_hists=False, show_bars=False, show=False, ovrd_bar_scale=[-0,1, 0.2], ovrd_heat_scale=[-0.2, 0.3])
        fig = show_relational_codes(R0_list, R1_list, show=False)
        #fig.savefig(os.path.join(os.path.expanduser('~'), 'figs', 'relational codes.pdf'))
        mysavefig(fig=fig, subfolder=output_folder, name='relational codes')
        mysavedata(subfolder=output_folder, name='relational codes', data=dict({'R0_list': R0_list, 'R1_list': R1_list}))
        fig = show_corr_diagonals(csac_list, rep_pcors_list, show=True)
        #fig.savefig(os.path.join(os.path.expanduser('~'), 'figs', 'corr_diagonals.pdf'))
        mysavefig(fig=fig, subfolder=output_folder, name='corr diagonals')
        mysavedata(subfolder=output_folder, name='diagonals', data=dict({'csac_list': csac_list, 'rep_pcors_list': rep_pcors_list}))


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
        use_mask = rr > thd
    if USE == 'HIGH_RESP':
        thd = np.quantile(rr, 0.95)
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



def make_amp_hists(data):
    
    data = np.copy(data.mean(axis=0))
    amp_for_sort = data[:, 4:8].mean(axis=-1)
    idxs = np.argsort(amp_for_sort)
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fige, axe = plt.subplots(1, 1, figsize=(12, 8))
    for i_sec in [1, 2, 3, 5, 7]:
        ax.plot(data[:, i_sec][idxs], linewidth=0.25, label=str(i_sec - 1))
        axe.plot(np.log10(data[:, i_sec][idxs] + 0.001), linewidth=0.25, label=str(i_sec/2 - 1))
    ax.plot(data[:, 0][idxs], linewidth=0.25, label='-1')
    ax.plot(amp_for_sort[idxs], linewidth=4, label='mid')
    ax.grid(True)
    ax.legend()
    axe.plot(np.log10(data[:, 0][idxs] + 0.001), linewidth=0.25, label='-1')
    axe.plot(np.log10(amp_for_sort[idxs] + 0.001), linewidth=4, label='mid')
    axe.grid(True)
    axe.legend()
    mysavefig(name='amp histograms', fig=fig)
    mysavefig(name='amp histograms log', fig=fige)



    
    

def scramble_digits(data, seed=0, permute_epochs=False, add_permute=False):
    
    num_epochs, num_contacts, num_digits = data.shape
    if permute_epochs:
        data = np.copy(data)[np.array((3, 0, 4, 1, 5, 2)), :, :]
    digitlist = np.arange(num_digits, dtype=int)
    if add_permute:
        for i in np.arange(start=2, stop=num_digits-1, step=4):
            digitlist[i:i+2] = digitlist[i:i+2][::-1]
    perm = np.arange(num_epochs, dtype=int)
    if seed < 0:
        perm = np.roll(perm, seed)
    if seed > 0:
        np.random.seed(seed=seed)
    for i_digit in range(num_digits):
        if seed < 0:
            perm = np.roll(perm, -seed)
        if seed > 0:
            perm = np.random.permutation(perm)
        # print('digit', i_digit, '\tp:', perm)
        data[:, :, digitlist[i_digit]] = data[perm, :, digitlist[i_digit]]
    
    return data
    
    

if __name__ == '__main__':

    V_SAMP_PER_SEC = 10
    V_SAMP_PER_SEC_RDM = 1
    AUTO_OR_CROSS_ACTIVATION = "CROSS"  # "AUTO": generate session rdm from single epoch set (diagonal = 1); "CROSS": cross-correlate two epoch sets
    MIN_TGAP, MAX_TGAP = 60, 144#144, 336#24, 48
    CONTACT_SPLIT = None # None: use all, 0: even contacts only, 1: odd contacts only
    #event_type = 'CNTDWN' # one of: 'CNTDWN', 'RECALL', 'DSTRCT', 'REST'
    #
    RAW_EPOCH_AVG = 1
    WITHIN_SESSION_PROCESS = True
    if WITHIN_SESSION_PROCESS:
        MIN_TGAP, MAX_TGAP = 1, 1000
        WITHIN_SESSION_SEGMENT_SIZE, WITHIN_SESSION_PAIR_IDX = 6, 1
        #WITHIN_SESSION_SEGMENT_SIZE, WITHIN_SESSION_PAIR_IDX = 3, 1
        EPOCHS_TO_READ = 18
    else:
        EPOCHS_TO_READ = 18
        WITHIN_SESSION_SEGMENT_SIZE = 6#len(AVG_MANY_EPOCHS)

    # SAVE_CONTACT_LIST = False
    # USE_CONTACT_SELECTION_FROM_FILE = True
    # #CONTACT_SELECTION_FILE_NAME = 'C:/Users/menas/OneDrive/Desktop/openneuro/temp/contact_list_cntdwn_{}_{}'.format(MIN_TGAP, MAX_TGAP)
    # CONTACT_SELECTION_FILE_NAME = 'C:/Users/menas/OneDrive/Desktop/openneuro/temp/contact_list_cntdwn_3_5'        


    data_availability_obj = data_availability()
    epoch_subsets = [[i*RAW_EPOCH_AVG, (i+1)*RAW_EPOCH_AVG-1] for i in range(int(EPOCHS_TO_READ / RAW_EPOCH_AVG))]
    epoch_subsets = ['e{}-e{}'.format(i1, i2) for (i1, i2) in epoch_subsets]
    
    
    # prepare contact list
    # stage 1: find suitable contacts
    
    list_1C, list_2C = data_availability_obj.get_suitable_epoch_files_and_contacts(min_timegap_hrs=MIN_TGAP, max_timegap_hrs=MAX_TGAP,
                                                                                           proc_type='gamma_c_60_160', event_list=['CNTDWN'], 
                                                                                           num_epochs=18, enforce_first=True, single_session=WITHIN_SESSION_PROCESS)
    
    list_1R, list_2R = data_availability_obj.get_suitable_epoch_files_and_contacts(min_timegap_hrs=MIN_TGAP, max_timegap_hrs=MAX_TGAP,
                                                                                           proc_type='gamma_c_60_160', event_list=['CNTRL'], 
                                                                                           num_epochs=18, enforce_first=True, single_session=WITHIN_SESSION_PROCESS)
    
    (list_1C, list_2C, list_1R, list_2R) = data_availability_obj.intersect_epoch_files_and_contact_lists([list_1C, list_2C, list_1R, list_2R])
    
    #list_1C, list_2C, list_1R, list_2R = list_1C[:1], list_2C[:1], list_1R[:1], list_2R[:1]
    
    contact_info = data_availability_obj.get_contact_info(list_1C)
    
    boundary_sec = np.arange(start=-1, stop=11+1e-6, step=1/V_SAMP_PER_SEC)

    # read data
    data_1C, cntct_mask = read_epoch_files_by_list(list_1C, first_epoch=0, last_epoch=EPOCHS_TO_READ, boundary_sec=boundary_sec, random_shift=False)
    data_1R, _ = read_epoch_files_by_list(list_1R, first_epoch=0, last_epoch=EPOCHS_TO_READ, boundary_sec=boundary_sec, random_shift=False, verbose=False)
    #
    # data_raw = np.copy(data_1C[:, cntct_mask, :])
    #

    #cntct_mask = np.array([c['location'][0]['region'].find('fus') > -1 for c in contact_info]) * cntct_mask
    #
    if WITHIN_SESSION_PROCESS:
        scnd_start_idx = WITHIN_SESSION_SEGMENT_SIZE * WITHIN_SESSION_PAIR_IDX
        data_2C = data_1C[scnd_start_idx:scnd_start_idx+WITHIN_SESSION_SEGMENT_SIZE]
        data_1C = data_1C[:WITHIN_SESSION_SEGMENT_SIZE]
        data_2R = data_1R[scnd_start_idx:scnd_start_idx+WITHIN_SESSION_SEGMENT_SIZE]
        data_1R = data_1R[:WITHIN_SESSION_SEGMENT_SIZE]
    else:
        data_2C, cntct_mask_2 = read_epoch_files_by_list(list_2C, first_epoch=0, last_epoch=18, boundary_sec=boundary_sec, random_shift=False)
        data_2R, _ = read_epoch_files_by_list(list_2R, first_epoch=0, last_epoch=18, boundary_sec=boundary_sec, random_shift=True, verbose=False)
        cntct_mask = cntct_mask * cntct_mask_2

    data_1C = data_1C[:, cntct_mask, :]
    data_2C = data_2C[:, cntct_mask, :]
    data_1R = data_1R[:, cntct_mask, :]
    data_2R = data_2R[:, cntct_mask, :]
    contact_info = [contact_info[i] for i in np.argwhere(cntct_mask).flatten().astype(int)]
    
    digit_permute_obj = digit_permutator()
    #np.random.seed(2)
    pidxs = np.random.choice(digit_permute_obj.get_num_perms(), size=2, replace=False)
    print(pidxs)
        
    
    for USE in ['ALL', 'NON_RESP', 'RESP', 'HIGH_RESP']:
        for SPLIT in ['ALL', 'ODD', 'EVEN']:
            
            # if (USE != 'NON_RESP') or (SPLIT != 'ODD'):
            #     continue
            if SPLIT != 'ALL':
                continue
            if USE == 'HIGH_RESP':
                continue
            
            SLCT_CONTACTS_BY_CONTRAST = False
            if not SLCT_CONTACTS_BY_CONTRAST:
                data_1C_, data_2C_, data_1R_, data_2R_, contact_info_, _ = \
                    get_contact_subset(data_1C, data_2C, data_1R, data_2R, contact_info, boundary_sec=boundary_sec, USE=USE, SPLIT=SPLIT)
                print(data_1C_.shape, data_2C_.shape, data_1R_.shape, data_2R_.shape, len(contact_info_))
            #
            if SLCT_CONTACTS_BY_CONTRAST:
                data_1C__ = resample_epoch(data_1C, fs=V_SAMP_PER_SEC, tscale=boundary_sec[:-1], boundary_sec=np.arange(start=boundary_sec[0], stop=boundary_sec[-1]+1e-6, step=1))
                data_2C__ = resample_epoch(data_2C, fs=V_SAMP_PER_SEC, tscale=boundary_sec[:-1], boundary_sec=np.arange(start=boundary_sec[0], stop=boundary_sec[-1]+1e-6, step=1))
                from contrast_analysis import eval_contrast
                mask = eval_contrast(data_1C__, data_2C__, USE=USE)
                data_1C_ = data_1C[:, mask]
                data_2C_ = data_2C[:, mask]
                data_1R_ = data_1R[:, mask]
                data_2R_ = data_2R[:, mask]
                contact_info_ = [contact_info[i] for i in np.argwhere(mask).flatten().astype(int)]
            # data_ctl = data_raw[12:18]
            # _, _, _, _, contact_info, mask = get_contact_subset(data_ctl, data_ctl, data_ctl, data_ctl, contact_info, boundary_sec=boundary_sec, USE=USE, SPLIT=SPLIT)
            # data_1C = data_raw[0:6, mask]
            # data_2C = data_raw[6:12, mask]
            #

            for event in ['CNTDWN', 'RECALL']:

                if event == 'CNTDWN':
                    data_1_, data_2_ = data_1C_, data_2C_
                if event == 'RECALL':
                    data_1_, data_2_ = data_1R_, data_2R_
                output_folder = '{}_USE_{}_SPLIT_{}'.format(event, USE, SPLIT)
                print('\n\n\nworking on', output_folder)
                # output_folder = 'default'
                psth_by_cntct = (data_1_.mean(axis=0) + data_2_.mean(axis=0)) / 2
                psth_all = psth_by_cntct.mean(axis=0)
                psth_all_sem = psth_by_cntct.std(axis=0) / np.sqrt(psth_by_cntct.shape[0])
                fig, ax = plt.subplots(1, 1)
                # ax.bar((boundary_sec[:-1] + boundary_sec[1:]) / 2, psth_all, width=1/V_SAMP_PER_SEC)
                # ax.bar((boundary_sec[:-1] + boundary_sec[1:]) / 2, 2 * psth_all_sem, bottom=psth_all - psth_all_sem, width=0.5/V_SAMP_PER_SEC, color='k')
                ax.plot((boundary_sec[:-1] + boundary_sec[1:]) / 2, np.log(psth_all))
                ax.set_ylim((-0.1, 0.3))
                ax.grid(True)
                ax.set_title('PSTH   ({} contacts)'.format(psth_by_cntct.shape[0]))
                # #
                # psth_by_cntct = data_1_[:6].mean(axis=0)
                # psth_all_1 = psth_by_cntct.mean(axis=0)
                # psth_all_sem_2 = psth_by_cntct.std(axis=0) / np.sqrt(psth_by_cntct.shape[0])
                # psth_by_cntct = data_2_[:6].mean(axis=0)
                # psth_all_2 = psth_by_cntct.mean(axis=0)
                # psth_all_sem_2 = psth_by_cntct.std(axis=0) / np.sqrt(psth_by_cntct.shape[0])
                # fig, ax = plt.subplots(1, 1)
                # ax.plot((boundary_sec[:-1] + boundary_sec[1:]) / 2, np.log(psth_all_1), label='epochs 1-6')
                # ax.plot((boundary_sec[:-1] + boundary_sec[1:]) / 2, np.log(psth_all_2), label='epochs 7-12')
                # ax.set_ylim((-0.1, 0.3))
                # ax.grid(True)
                # ax.legend()
                # ax.set_title('PSTH   ({} contacts), activity avaluated on epochs 1-6'.format(psth_by_cntct.shape[0]))
                # #
                # ax.set_ylim((0.5, 1.5))
                #fig.savefig(os.path.join(os.path.expanduser('~'), 'figs', 'PSTH.pdf'))
                mysavefig(name='PSTH', subfolder=output_folder, fig=fig)
                mysavedata(subfolder=output_folder, name='PSTH', data=dict({'boundary_sec': boundary_sec, 'psth': psth_all, 'psth_sem': psth_all_sem}))
                
                

                fig = show_region_distribution(contact_info_, title='{} contacts , delta=T = {} hrs to {} hrs'.format(len(contact_info_), MIN_TGAP, MAX_TGAP))
                #fig.savefig(os.path.join(os.path.expanduser('~'), 'figs', 'region_distribution.pdf'))
                mysavefig(name='region_distribution', subfolder=output_folder, fig=fig)
                with open(os.path.join(os.path.expanduser('~'), 'figs', output_folder, 'contact_data'), 'wb') as fd:
                    pickle.dump({'contact_info': contact_info_}, fd)
                
                
                input_tscale = boundary_sec[:-1]
                #boundary_sec = np.arange(start=boundary_sec[0], stop=boundary_sec[-1], step=1)
                data_1_ = resample_epoch(data_1_, fs=V_SAMP_PER_SEC, tscale=input_tscale, boundary_sec=np.arange(start=boundary_sec[0], stop=boundary_sec[-1]+1e-6, step=1/V_SAMP_PER_SEC_RDM))
                data_2_ = resample_epoch(data_2_, fs=V_SAMP_PER_SEC, tscale=input_tscale, boundary_sec=np.arange(start=boundary_sec[0], stop=boundary_sec[-1]+1e-6, step=1/V_SAMP_PER_SEC_RDM))
                # make_amp_hists(np.concatenate((data_1_, data_2_), axis=0))
                # assert False
                # #
                # from fc_tools import *
                # fig = show_fc_map(generate_global_fc_map(data_1_[:, 10000:11000:5]))
                # mysavefig(name='functional connectivity', subfolder=output_folder, fig=fig)
                #
                
                # # remove average per epoch
                # avg = data_1_[:, :, :].mean(axis=-1)
                # data_1_ -= np.array([avg for i in range(data_1_.shape[2])]).transpose((1, 2, 0))
                # avg = data_2_[:, :, :].mean(axis=-1)
                # data_2_ -= np.array([avg for i in range(data_2_.shape[2])]).transpose((1, 2, 0))
                
                # data_1_ = scramble_digits(data_1_, seed=-2, permute_epochs=True)
                # data_2_ = scramble_digits(data_2_, seed=-2, add_permute=True)
                
                print(np.round(data_1_[:, 0, :], decimals=3))
                data_1_ = digit_permute_obj(data_1_, permid=pidxs[0])
                print(np.round(data_1_[:, 0, :], decimals=3))
                print('\n\n')
                print(np.round(data_2_[:, 0, :], decimals=3))
                data_2_ = digit_permute_obj(data_2_, permid=pidxs[1])
                print(np.round(data_2_[:, 0, :], decimals=3))
                #
                contact_info_ = contact_info_[:]
                #
                print('size of data:', data_1_.shape, data_2_.shape)
                do_rdm_analisys(data_1_, data_2_, epoch_count=6, output_folder=output_folder, v_samp_per_sec=V_SAMP_PER_SEC_RDM, contact_info=contact_info_)
                plt.close()
                #assert False
                
                    
                
        
        



