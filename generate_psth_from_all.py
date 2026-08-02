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
import argparse
import sys





def split_session_to_fake_sessions(contact_list, segment_size=3, pair_idx=1):

    for i_contact, contact in enumerate(contact_list):
        contact['second'] = contact['first'][int(pair_idx*segment_size):int((pair_idx+1)*segment_size)]
        contact['first'] = contact['first'][:segment_size]

    return contact_list

    
    
    




def get_contact_subset(data_1C, data_2C, data_3C, data_1R, data_2R, data_3R, 
                       contact_info, boundary_sec, USE='ALL', SPLIT='ALL', hr_type='short'):

    # now find "responsive" contacts
    #if USE != 'ALL':
    psth_by_cntct = (data_1C.mean(axis=0) + data_2C.mean(axis=0) + data_3C.mean(axis=0)) / 3
    #rr = np.median(psth_by_cntct[:, 1:], axis=-1) / psth_by_cntct[:, 0]
    if hr_type == 'short':
        mask_baseline = ((boundary_sec >= -0.5) * (boundary_sec < 0))[:-1]
        mask_response = ((boundary_sec >= 0.25) * (boundary_sec < 0.75))[:-1]
    if hr_type == 'long':
        mask_baseline = ((boundary_sec >= -0.5) * (boundary_sec < 0))[:-1]
        mask_response = ((boundary_sec >= 0) * (boundary_sec < 5))[:-1]
    if hr_type == 'max':
        mask_baseline = ((boundary_sec >= -0.5) * (boundary_sec < 0))[:-1]
        mask_response = ((boundary_sec >= 0) * (boundary_sec < 10))[:-1]
    rr = psth_by_cntct[:, mask_response].mean(axis=-1) / psth_by_cntct[:, mask_baseline].mean(axis=-1)
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
        #thd = np.quantile(rr, 0.975)
        thd = max(np.sort(rr)[::-1][min(rr.size - 1, max(25, int(rr.size / 3)))], 1.1)
        if hr_type == 'max':
            thd = max(np.sort(rr)[::-1][min(rr.size - 1, max(25, int(rr.size / 3)))], 1.05)
        use_mask = rr > thd
     #
    data_1C_ = data_1C[:, use_mask]
    data_2C_ = data_2C[:, use_mask]
    data_3C_ = data_2C[:, use_mask]
    data_1R_ = data_1R[:, use_mask]
    data_2R_ = data_2R[:, use_mask]
    data_3R_ = data_2R[:, use_mask]
    contact_info_ = [contact_info[i] for i in np.argwhere(use_mask).flatten().astype(int)]
    
    # split
    if SPLIT == 'ODD':
        split_mask = ((-1) ** np.arange(len(contact_info_))) > 0
    if SPLIT == 'EVEN':
        split_mask = ((-1) ** np.arange(len(contact_info_))) < 0
    if SPLIT != 'ALL':
        data_1C_ = data_1C_[:, split_mask]
        data_2C_ = data_2C_[:, split_mask]
        data_3C_ = data_3C_[:, split_mask]
        data_1R_ = data_1R_[:, split_mask]
        data_2R_ = data_2R_[:, split_mask]
        data_3R_ = data_3R_[:, split_mask]
        contact_info_ = [contact_info_[i] for i in np.argwhere(split_mask).flatten().astype(int)]
    
    # re-generate mask
    mask = np.zeros(use_mask.shape, dtype=bool)
    use_idxes = np.argwhere(use_mask).flatten() if SPLIT=='ALL' else np.argwhere(use_mask).flatten()[split_mask]
    mask[use_idxes] = True
     
    return data_1C_, data_2C_, data_3C_, data_1R_, data_2R_, data_3R_, contact_info_, mask


ALL_REGIONS = ['bankssts', 'caudalanteriorcingulate', 'caudalmiddlefrontal', 'cuneus', 'entorhinal', 'frontalpole', 'fusiform', 
               'inferiorparietal', 'inferiortemporal', 'insula', 'isthmuscingulate', 'lateraloccipital', 'lateralorbitofrontal',
               'lingual', 'medialorbitofrontal', 'middletemporal', 'mix', 'paracentral', 'parahippocampal', 'parsopercularis',
               'parsorbitalis', 'parstriangularis', 'pericalcarine', 'postcentral', 'posteriorcingulate', 'precentral', 'precuneus',
               'rostralanteriorcingulate', 'rostralmiddlefrontal', 'superiorfrontal', 'superiorparietal', 'superiortemporal',
               'supramarginal', 'temporalpole', 'transversetemporal']

RESPONSIVE_REGIONS = ['cuneus', 'pericalcarine', 'postcentral', 'precentral', 'lingual', 'superiorparietal', 
                      'inferiortemporal', 'middletemporal', 'fusiform', 'lateraloccipital']

EXTENSION_REGIONS = ['bankssts', 'caudalmiddlefrontal', 'inferiorparietal', 'insula', 'lateralorbitofrontal',
                     'parsopercularis', 'parstriangularis', 'precuneus', 'rostralmiddlefrontal', 'superiorfrontal',
                     'superiortemporal']

    

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')




if __name__ == '__main__':
    
    print('\n\n***************\n\nstarting\n\n*********************\n\n')
    
# 1. Set up the argument parser
    parser = argparse.ArgumentParser(description="Run session analysis with custom parameters.")
    
    # 2. Define the parameters with their current default values
    parser.add_argument('--NUM_SESSIONS', type=int, default=1, choices=[1, 2, 3],
                        help='Number of sessions (1, 2, or 3)')
    parser.add_argument('--SELECT_BY_EPOCHS', type=str, default='all', choices=['all', 'odd', 'even', 'first', 'second'],
                        help='Epoch selection strategy')
    parser.add_argument('--CALC_BY_EPOCHS', type=str, default='all', choices=['all', 'odd', 'even', 'first', 'second'],
                        help='Epoch calculation strategy')
    parser.add_argument('--X_CNCT_CNTCTS', type=str2bool, default=False,
                        help='Cross-connect contacts (True/False)')
    parser.add_argument('--X_CNCT_EPOCHS', type=str2bool, default=True,
                        help='Cross-connect epochs (True/False)')
    parser.add_argument('--HR_SELECT_TYPE', type=str, default='short', choices=['short', 'max', 'consistancy'],
                        help='creterinon for selecting responsive contacts')

    # 3. Parse the arguments from the command line
    args = parser.parse_parser() if hasattr(parser, 'parse_parser') else parser.parse_args()

    # 4. Map them to your existing variable names
    NUM_SESSIONS = args.NUM_SESSIONS
    SELECT_BY_EPOCHS = args.SELECT_BY_EPOCHS
    CALC_BY_EPOCHS = args.CALC_BY_EPOCHS
    X_CNCT_CNTCTS = args.X_CNCT_CNTCTS
    X_CNCT_EPOCHS = args.X_CNCT_EPOCHS
    HR_SELECT_TYPE = args.HR_SELECT_TYPE

    # --- Your remaining static constants stay the same ---
    print('\n\n***************\n\nstarting\n\n*********************\n\n')
    
    V_SAMP_PER_SEC = 10
    V_SAMP_PER_SEC_RDM = 1
    MIN_TGAP, MAX_TGAP = 24, 480
    CONTACT_SPLIT = None 
    RAW_EPOCH_AVG = 1
    SELECT_BY_REGION = True
    SEM_BY_CONTACT = True

    # Keep your structural assertion safety check
    assert NUM_SESSIONS in [1, 2, 3]

    # Print parsed parameters to verify they look correct in logs
    print(f"Running with: NUM_SESSIONS={NUM_SESSIONS}, SELECT_BY_EPOCHS='{SELECT_BY_EPOCHS}', "
          f"CALC_BY_EPOCHS='{CALC_BY_EPOCHS}', X_CNCT_CNTCTS={X_CNCT_CNTCTS}, X_CNCT_EPOCHS={X_CNCT_EPOCHS}\n")
    
    
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

    list_1C, subject_int_id = data_availability_obj.get_all_epoch_files_and_contacts(proc_type='gamma_c_60_160', event_list=['CNTDWN'], min_num_epochs=8)
    

    # list_1C, list_2C, list_3C = data_availability_obj.get_suitable_epoch_files_and_contacts(min_timegap_hrs=MIN_TGAP, max_timegap_hrs=MAX_TGAP,
    #                                                                                         proc_type='gamma_c_60_160', event_list=['CNTDWN'], 
    #                                                                                         num_epochs=EPOCHS_TO_READ, enforce_first=True, 
    #                                                                                         single_session=WITHIN_SESSION_PROCESS, third_session=THIRD_SESSION)
    # subject_int_id = subject_int_id[:len(list_1C)]
    # if NUM_SESSIONS < 3:
    #     list_3C = list_2C # !!! PATCH !!!
    # (list_1C, list_2C, list_3C) = data_availability_obj.intersect_epoch_files_and_contact_lists([list_1C, list_2C, list_3C])

    # # if PROCESS_RECALL: 
    # #     list_1R, list_2R = data_availability_obj.get_suitable_epoch_files_and_contacts(min_timegap_hrs=MIN_TGAP, max_timegap_hrs=MAX_TGAP,
    # #                                                                                         proc_type='gamma_c_60_160', event_list=['RECALL'], 
    # #                                                                                         num_epochs=EPOCHS_TO_READ, enforce_first=True, single_session=WITHIN_SESSION_PROCESS)
    # #     (list_1C, list_2C, list_1R, list_2R) = data_availability_obj.intersect_epoch_files_and_contact_lists([list_1C, list_2C, list_1R, list_2R])
    

    print('A')
        
    #list_1C, list_2C, list_1R, list_2R = list_1C[:1], list_2C[:1], list_1R[:1], list_2R[:1]
        
    contact_info = data_availability_obj.get_contact_info(list_1C)
    print('B')


    list_1C_full, contact_info_full = copy.copy(list_1C), copy.copy(contact_info)
    erased_folder_list = []
    slct_masks = dict()
    psth_set = dict({'ALL': dict(), 'HIGH_RESP': dict()})
    for base_region in RESPONSIVE_REGIONS:# + EXTENSION_REGIONS:#RESPONSIVE_REGIONS[8:9]:# + EXTENSION_REGIONS:#
        
        list_1C, contact_info = copy.copy(list_1C_full), copy.copy(contact_info_full)
        print('\n', base_region)

        #
        if SELECT_BY_REGION:
            region_list = [base_region + '-R', base_region + '-L']#['superiorparietal-R', 'superiorparietal-L']#early_list + mid_list
            _, contact_info = select_channels_by_regions(contact_info=contact_info, region_list=region_list)
            contact_info_imp = contact_info
        else:
            contact_info_imp = contact_info
        list_1C, _ = data_availability_obj.intersect_contact_list_and_contact_info(contact_list=list_1C, contact_info=contact_info_imp)
        #
        
        boundary_sec = np.arange(start=-5, stop=12+1e-6+3, step=0.02)#1/V_SAMP_PER_SEC)

        # read data
        if len(list_1C) == 0:
            print('XXXXX   NO CONTACTS FOR ', base_region)
            continue

        data_1C, cntct_mask, epoch_count = read_all_epoch_files_by_list(list_1C, norm_per_epoch=True,
                                                    boundary_sec=boundary_sec, random_shift=False, norm_baseline=[-5, 15])#[-0.5, -0.05])#

        # data_1C, cntct_mask = read_all_epoch_files_by_list(list_1C, first_epoch=0, last_epoch=EPOCHS_TO_READ, norm_per_epoch=True,
        #                                             boundary_sec=boundary_sec, random_shift=False, norm_baseline=[-5, 15])#[-0.5, -0.05])#
        print('C')


        data_1C = data_1C[:, cntct_mask, :]
        contact_info = [contact_info[i] for i in np.argwhere(cntct_mask).flatten().astype(int)]
        
        epoch_count = np.array([epoch_count[i] for i in np.argwhere(cntct_mask).flatten().astype(int)]).astype(int)
        data_1C_all_ave = adaptive_epoch_ave(data_1C, epoch_count, tgt_num_epochs=4)
        data_1C_odd_ep = data_1C[1::2]
        data_1C_even_ep = data_1C[::2]
        data_1C_odd_ep_ave = adaptive_epoch_ave(data_1C_odd_ep, (epoch_count / 2).astype(int), tgt_num_epochs=4)
        data_1C_even_ep_ave = adaptive_epoch_ave(data_1C_even_ep, (epoch_count / 2).astype(int), tgt_num_epochs=4)

            
        
        for USE in ['ALL', 'NON_RESP', 'RESP', 'HIGH_RESP']:
            for SPLIT in ['ALL', 'ODD', 'EVEN']:

                # erase all old folders
                for event in ['CNTDWN', 'RECALL']:
                    folder_path = os.path.join(os.path.expanduser('~'), 'figs', '{}_USE_{}_SPLIT_{}'.format(event, USE, SPLIT))
                    if not folder_path in erased_folder_list:
                        shutil.rmtree(folder_path, ignore_errors=True)
                        erased_folder_list.append(folder_path)
                
                # if (USE != 'NON_RESP') or (SPLIT != 'ODD'):
                #     continue
                if SPLIT != 'ALL':
                    continue
                if USE not in ['ALL', 'HIGH_RESP']:
                    continue
                
                if True:
                    # # data_1C_, data_2C_, data_1R_, data_2R_, contact_info_, _ = \
                    # #     get_contact_subset(data_1C, data_2C, data_1R, data_2R, contact_info, boundary_sec=boundary_sec, USE=USE, SPLIT=SPLIT)
                    # #
                    # epoch_subset = np.arange(data_1C.shape[0])
                    # epoch_subset = epoch_subset[1::2] if SELECT_BY_EPOCHS == 'odd' else epoch_subset
                    # epoch_subset = epoch_subset[::2] if SELECT_BY_EPOCHS == 'even' else epoch_subset
                    # epoch_subset = epoch_subset[:int(data_1C.shape[0] / 2)] if SELECT_BY_EPOCHS == 'first' else epoch_subset
                    # epoch_subset = epoch_subset[int(data_1C.shape[0] / 2):] if SELECT_BY_EPOCHS == 'second' else epoch_subset
                    data_1C_sel = data_1C
                    data_1C_sel = data_1C_odd_ep if SELECT_BY_EPOCHS == 'odd' else data_1C_sel
                    data_1C_sel = data_1C_even_ep if SELECT_BY_EPOCHS == 'even' else data_1C_sel
                    data_1C_sel_ave = data_1C_all_ave
                    data_1C_sel_ave = data_1C_odd_ep_ave if SELECT_BY_EPOCHS == 'odd' else data_1C_sel_ave
                    data_1C_sel_ave = data_1C_even_ep_ave if SELECT_BY_EPOCHS == 'even' else data_1C_sel_ave
                    data_1C_clc = data_1C
                    data_1C_clc = data_1C_odd_ep if SELECT_BY_EPOCHS == 'even' else data_1C_sel
                    data_1C_clc = data_1C_even_ep if SELECT_BY_EPOCHS == 'odd' else data_1C_sel
                    data_1C_clc_ave = data_1C_all_ave
                    data_1C_clc_ave = data_1C_odd_ep_ave if SELECT_BY_EPOCHS == 'even' else data_1C_sel_ave
                    data_1C_clc_ave = data_1C_even_ep_ave if SELECT_BY_EPOCHS == 'odd' else data_1C_sel_ave
                    epoch_subset = np.arange(4)
                    # #
                    _, _, _, _, _,  _, contact_info_, slct_mask = \
                        get_contact_subset(data_1C_sel_ave[epoch_subset], data_1C_sel_ave[epoch_subset], data_1C_sel_ave[epoch_subset], 
                                           data_1C_sel_ave[epoch_subset], data_1C_sel_ave[epoch_subset], data_1C_sel_ave[epoch_subset], 
                                           contact_info, boundary_sec=boundary_sec, USE=USE, SPLIT=SPLIT, hr_type=HR_SELECT_TYPE)
                    # #
                    # epoch_subset = np.arange(data_1C.shape[0])
                    # epoch_subset = epoch_subset[1::2] if CALC_BY_EPOCHS == 'odd' else epoch_subset
                    # epoch_subset = epoch_subset[::2] if CALC_BY_EPOCHS == 'even' else epoch_subset
                    # epoch_subset = epoch_subset[:int(data_1C.shape[0] / 2)] if CALC_BY_EPOCHS == 'first' else epoch_subset
                    # epoch_subset = epoch_subset[int(data_1C.shape[0] / 2):] if CALC_BY_EPOCHS == 'second' else epoch_subset
                    # #
                    data_1C_ = data_1C[epoch_subset][:, slct_mask]
                    #
                    data_1C_sel_ = data_1C_sel[epoch_subset][:, slct_mask]
                    data_1C_sel_ave_ = data_1C_sel_ave[epoch_subset][:, slct_mask]
                    data_1C_clc_ = data_1C_clc[epoch_subset][:, slct_mask]
                    data_1C_clc_ave_ = data_1C_clc_ave[epoch_subset][:, slct_mask]
                    # data_2C_ = data_2C[epoch_subset][:, slct_mask]
                    # data_3C_ = data_3C[epoch_subset][:, slct_mask]
                    # print(data_1C_.shape, data_2C_.shape, data_3C_.shape, len(contact_info_))
                    slct_masks[base_region] = slct_mask
                #
                
                if (USE != 'ALL') and (len(contact_info_) < 1):
                    continue
                

                for event in ['CNTDWN']:#, 'RECALL']:

                    if event == 'CNTDWN':
                        data_1_ = data_1C_
                    # if event == 'RECALL':
                    #     if PROCESS_RECALL:
                    #         data_1_ = data_1R_
                    #     else:
                    #         continue
                        
                    output_folder = os.path.join('{} sessions'.format(NUM_SESSIONS), 
                                                'selectby_{}_calc_{}'.format(SELECT_BY_EPOCHS, CALC_BY_EPOCHS),
                                                 '{}_USE_{}_SPLIT_{}'.format(event, USE, SPLIT))
                    print('\n\n\nworking on', output_folder)
                    #

                    # SEM_BY_EPOCH = not SEM_BY_CONTACT
                    # if SEM_BY_CONTACT:
                    #     psth_by_cntct_1 = data_1_.mean(axis=0)
                    #     psth_all_1 = psth_by_cntct_1.mean(axis=0)
                    #     psth_all_sem_1 = psth_by_cntct_1.std(axis=0) / np.sqrt(psth_by_cntct_1.shape[0])
                    # if SEM_BY_EPOCH:
                    #     psth_by_epoch_1 = data_1_.mean(axis=1)
                    #     psth_all_1 = psth_by_epoch_1.mean(axis=0)
                    #     psth_all_sem_1 = psth_by_epoch_1.std(axis=0) / np.sqrt(psth_by_epoch_1.shape[0])
                    # #
                    for data_use_ave, data_use_name in zip((data_1C_sel_ave_, data_1C_clc_ave), (SELECT_BY_EPOCHS, CALC_BY_EPOCHS)):
                        #psth_by_cntct_1 = data_1C_sel_ave_.mean(axis=0)
                        psth_by_cntct_1 = data_use_ave.mean(axis=0)
                        psth_all_1 = psth_by_cntct_1.mean(axis=0)
                        psth_all_sem_1 = psth_by_cntct_1.std(axis=0) / np.sqrt(psth_by_cntct_1.shape[0])
                        #
                        fig, ax = plt.subplots(1, 1)
                        # ax.bar((boundary_sec[:-1] + boundary_sec[1:]) / 2, psth_all, width=1/V_SAMP_PER_SEC)
                        # ax.bar((boundary_sec[:-1] + boundary_sec[1:]) / 2, 2 * psth_all_sem, bottom=psth_all - psth_all_sem, width=0.5/V_SAMP_PER_SEC, color='k')
                        line1, = ax.plot((boundary_sec[:-1] + boundary_sec[1:]) / 2, np.log(psth_all_1), label='sess 1')
                        ax.fill_between((boundary_sec[:-1] + boundary_sec[1:]) / 2,
                                        np.log(np.maximum(psth_all_1 - psth_all_sem_1, 1e-6)),
                                        np.log(np.maximum(psth_all_1 + psth_all_sem_1, 1e-6)), color=line1.get_color(), alpha=0.2)
                        psth_set[USE][base_region] = [[psth_all_1, psth_all_sem_1]]

                        ax.set_ylim((-0.1, 0.3))
                        ax.grid(True)
                        ax.legend()
                        ax.set_title('PSTH   ({}  ,  {} contacts  , epoch subset = {})'.format(base_region, data_1_.shape[1], data_use_name))
                        mysavefig(name='PSTH  ({})   subset {}'.format(base_region, data_use_name), subfolder=output_folder, fig=fig)
                        # mysavedata(subfolder=output_folder, name='PSTH', data=dict({'boundary_sec': boundary_sec,
                        #                                                             'psth_1': psth_all_1, 'psth_sem_1': psth_all_sem_1,
                        #                                                             #'psth_2': psth_all_2, 'psth_sem_2': psth_all_sem_2,
                        #                                                             'slct_masks': slct_masks, 'psth_set': psth_set}))
                        #
                        # save contact list and info
                        region_list_1C, _ = data_availability_obj.intersect_contact_list_and_contact_info(contact_list=list_1C, contact_info=contact_info_)
                        lists = [region_list_1C]
                        if data_use_name == SELECT_BY_EPOCHS:
                            mysavedata(subfolder=output_folder, fname='contacts', name=base_region, 
                                        data=dict({'contact_lists': lists, 'contact_info': contact_info_}))

                    
                    

                    fig = show_region_distribution(contact_info_, title='{} contacts , delta=T = {} hrs to {} hrs'.format(len(contact_info_), MIN_TGAP, MAX_TGAP))
                    #fig.savefig(os.path.join(os.path.expanduser('~'), 'figs', 'region_distribution.pdf'))
                    mysavefig(name='region_distribution', subfolder=output_folder, fig=fig)
                    with open(os.path.join(os.path.expanduser('~'), 'figs', output_folder, 'contact_data'), 'wb') as fd:
                        pickle.dump({'contact_info': contact_info_}, fd)
                    

                    # now create RDM
                    # parameters:
                    FIRST_CHUNK = 0#-5
                    LASK_CHUNK = 10#15
                    CHUNK_SIZE = 0.1#1#
                    CHUNK_PRUNE = 1
                    CHUNK_PHASE = 0
                    assert CHUNK_PHASE < CHUNK_PRUNE 
                    #
                    from rdm_tools_new import resample_epoch, calc_rdm
                    import seaborn as sns
                    rdm_boundary_sec = np.arange(start=FIRST_CHUNK, stop=LASK_CHUNK, step=CHUNK_SIZE)
                    # data_1C__ = resample_epoch(data_1C_, fs=None, tscale=boundary_sec[:-1], boundary_sec=rdm_boundary_sec)
                    # data_1C__ = data_1C__[:, :, CHUNK_PHASE::CHUNK_PRUNE]
                    #
                    data_1C__ = resample_epoch(data_1C_clc_ave_, fs=None, tscale=boundary_sec[:-1], boundary_sec=rdm_boundary_sec)
                    data_1C__ = data_1C__[:, :, CHUNK_PHASE::CHUNK_PRUNE]
                    

                    fig_rdm_whole, ax_rdm_whole = plt.subplots(1, 1)
                    rdm_whole = calc_rdm(data_1C__.mean(axis=0)[np.newaxis, : :], data_1C__.shape[-1], 0, 1, corr_mode='p')
                    sns.heatmap(rdm_whole, ax=ax_rdm_whole, vmin=-1, vmax=1, xticklabels=False, yticklabels=False)
                    matrix_size = rdm_whole.shape[0]
                    tick_positions = np.arange(0, matrix_size, 5)
                    tick_labels = np.round(np.linspace(0, 0.1*(matrix_size-1), num=matrix_size), decimals=1)[::5]
                    ax_rdm_whole.set_xticks(tick_positions)
                    ax_rdm_whole.set_xticklabels(tick_labels, rotation=0)
                    ax_rdm_whole.set_yticks(tick_positions)
                    ax_rdm_whole.set_yticklabels(tick_labels, rotation=0)
                    ax_rdm_whole.set_title('{} (subset {})'.format(base_region, CALC_BY_EPOCHS))
                    ax_rdm_whole.set_xlabel('time inside countdown')
                    ax_rdm_whole.set_ylabel('time inside countdown')
                    mysavefig(name=f'whole rdm ({base_region})', subfolder=output_folder, fig=fig_rdm_whole)


                    # # rdm0 = calc_rdm(data_1C__[0:2], data_1C__.shape[-1], 0, 1, corr_mode='p')
                    # # rdm1 = calc_rdm(data_1C__[1:3], data_1C__.shape[-1], 0, 1, corr_mode='p')
                    # # rdm2 = calc_rdm(data_1C__[2:4], data_1C__.shape[-1], 0, 1, corr_mode='p')
                    # rdms = [calc_rdm(data_1C__[i:i+2], data_1C__.shape[-1], 0, 1, corr_mode='p') for i in range(3)]
                    
                    # # R0_ = relative_codes(rdm0_, first=0, remove_diag=True, normalize=False)
                    # # R1_ = relative_codes(rdm1_, first=0, remove_diag=True, normalize=False)
                    # import matplotlib.ticker as ticker
                    # fig, ax = plt.subplots(1, 3, figsize=(18, 6))
                    # rdm_boundary_sec = np.round(rdm_boundary_sec, decimals=1)
                    # for i_ax in range(3):
                    #     sns.heatmap(np.round(rdms[i_ax], decimals=2), ax=ax[i_ax], cbar=False, vmin=-1, vmax=1, annot=False, 
                    #                 xticklabels=rdm_boundary_sec[:-1], yticklabels=rdm_boundary_sec[:-1], square=False)
                    #     ax[i_ax].xaxis.set_major_locator(ticker.IndexLocator(base=10, offset=0.5))
                    #     ax[i_ax].yaxis.set_major_locator(ticker.IndexLocator(base=10, offset=0.5))
                    #     #
                    # fig.suptitle('rdms, chunk_size={},  {}'.format(CHUNK_SIZE, base_region))
                    # mysavefig(name=f'rdms ({base_region})', subfolder=output_folder, fig=fig)
                    # #
                    # data_1C__[0] = data_1C__.mean(axis=0)
                    # data_1C__ = data_1C__[:1]
                    # rdm = calc_rdm(data_1C__, data_1C__.shape[-1], 0, 1, corr_mode='p')
                    # fig, ax = plt.subplots(1, 1)
                    # sns.heatmap(rdm, ax=ax, cbar=True, vmin=-1, vmax=1, annot=False, 
                    #             xticklabels=rdm_boundary_sec[:-1], yticklabels=rdm_boundary_sec[:-1], square=True)
                    # fig.suptitle('rdm, chunk_size={},  {}'.format(CHUNK_SIZE, base_region))
                    # ax.xaxis.set_major_locator(ticker.IndexLocator(base=10, offset=0.5))
                    # ax.yaxis.set_major_locator(ticker.IndexLocator(base=10, offset=0.5))
                    # mysavefig(name=f'rdm ({base_region})', subfolder=output_folder, fig=fig)
                    #
                    # partial second averaging
                    #
                    WINSIZE = 1#4#
                    PERIOD = 1#4#2#
                    WINSHIFT = 1#3#
                    fig_rdm, ax_rdm = plt.subplots(2, 4, figsize=(16, 8))
                    ax_rdm = np.atleast_1d(ax_rdm)
                    [ax.axis(False) for ax in ax_rdm.flatten()]
                    fig_rdm.suptitle(base_region)
                    fig_rdmx, ax_rdmx = plt.subplots(2, 4, figsize=(16, 8))
                    ax_rdmx = np.atleast_1d(ax_rdmx)
                    [ax.axis(False) for ax in ax_rdmx.flatten()]
                    ave_epoch_list, rdm_list, rdmx_list = [], [], []
                    fig_rdmx.suptitle(base_region)
                    i_ax = 0
                    for t_start in np.arange(start=rdm_boundary_sec[0], stop=rdm_boundary_sec[-1] - WINSIZE, step=WINSHIFT):
                        try:
                            tmask = ((rdm_boundary_sec[:-1] >= t_start) * (rdm_boundary_sec[:-1] < t_start + PERIOD))
                            ave_epoch = data_1C__[:, :, tmask]
                            t1 = t_start + PERIOD 
                            for i in range(1, int(WINSIZE / PERIOD)):
                                t0 = t_start + i * PERIOD
                                t1 = t0 + PERIOD
                                tmask = ((rdm_boundary_sec[:-1] >= t0) * (rdm_boundary_sec[:-1] < t1))
                                #print(t0, t1, tmask.sum())
                                ave_epoch += data_1C__[:, :, tmask]
                            ave_epoch_list.append(ave_epoch)
                            rdm_list.append(calc_rdm(ave_epoch[-1][np.newaxis, : :], ave_epoch[-1].shape[-1], 0, 1, corr_mode='p'))
                            sns.heatmap(rdm_list[-1], ax=ax_rdm.flatten()[i_ax], vmin=-1, vmax=1)
                            ax_rdm.flatten()[i_ax].set_title('{:4.1f} -- {:4.1f}'.format(t_start, t1))
                            # if len(ave_epoch_list) >= 3:
                            #     pair_ave = np.concatenate((ave_epoch_list[-3], ave_epoch_list[-1]), axis=0)
                            #     rdmx_list.append(calc_rdm(pair_ave, ave_epoch_list[-1].shape[-1], 0, 1, corr_mode='p'))
                            #     sns.heatmap(rdmx_list[-1], ax=ax_rdmx.flatten()[i_ax], vmin=-1, vmax=1)
                            #     ax_rdmx.flatten()[i_ax].set_title('{:4.1f} -- {:4.1f}'.format(t_start - 2 * WINSHIFT, t1))
                            i_ax += 1
                        except:
                            pass
                    #mysavefig(name=f'mult rdm ({base_region})', subfolder=output_folder, fig=fig_rdm)
                    #mysavefig(name=f'mult rdmx ({base_region})', subfolder=output_folder, fig=fig_rdmx)
                    fig_ave_rdm, ax = plt.subplots(1, 1)
                    ave_rdm = np.array(rdm_list).mean(axis=0)
                    sns.heatmap(ave_rdm, ax=ax, vmin=-1, vmax=1)
                    ax.set_title('{} (subset {})'.format(base_region, CALC_BY_EPOCHS))
                    ax.set_xlabel('digit')
                    ax.set_ylabel('digit')
                    mysavefig(name=f'ave rdm ({base_region})', subfolder=output_folder, fig=fig_ave_rdm)
                    #
                    fig_comb, ax_comb = plt.subplots(2, 5, figsize=(16, 5))
                    rdm_comb_list = []
                    for i in range(10):
                        x = data_1C__.mean(axis=0)[:, i::10][np.newaxis, : :]
                        rdm_comb_list.append(calc_rdm(x, x.shape[-1], 0, 1, corr_mode='p'))
                        sns.heatmap(rdm_comb_list[-1], ax=ax_comb.flatten()[i], vmin=-1, vmax=1, cbar=False)
                        ax_comb.flatten()[i].set_title('part {}'.format(i+1))
                    mysavefig(name=f'comb rdms ({base_region})', subfolder=output_folder, fig=fig_comb)
                    ave_comb_rdm = np.zeros((10, 10))
                    ndiv = 0
                    for i in range(10):
                        if rdm_comb_list[i].shape[0] == 10:
                            ave_comb_rdm += rdm_comb_list[i]
                            ndiv += 1
                    ave_comb_rdm /= ndiv
                    fig_comb_ave, ax_comb_ave = plt.subplots(1, 1)
                    sns.heatmap(ave_comb_rdm, ax=ax_comb_ave, vmin=-1, vmax=1,
                                xticklabels=np.round(np.linspace(0, 0.9, num=10), decimals=1),
                                yticklabels=np.round(np.linspace(0, 0.9, num=10), decimals=1))
                    ax.set_title('{} (subset {})'.format(base_region, CALC_BY_EPOCHS))
                    ax.set_xlabel('"phase" (sec.)')
                    ax.set_ylabel('"phase" (sec.)')
                    mysavefig(name=f'comb rdms ave ({base_region})', subfolder=output_folder, fig=fig_comb_ave)


 
 