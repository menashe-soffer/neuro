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

    
    
    




def get_contact_subset(data_1C, data_2C, data_3C, data_1R, data_2R, data_3R, 
                       contact_info, boundary_sec, USE='ALL', SPLIT='ALL', hr_type='short'):

    # now find "responsive" contacts
    #if USE != 'ALL':
    psth_by_cntct = (data_1C.mean(axis=0) + data_2C.mean(axis=0) + data_3C.mean(axis=0)) / 2
    #rr = np.median(psth_by_cntct[:, 1:], axis=-1) / psth_by_cntct[:, 0]
    if hr_type == 'short':
        mask_baseline = ((boundary_sec >= -0.5) * (boundary_sec < 0))[:-1]
        mask_response = ((boundary_sec >= 0.25) * (boundary_sec < 0.75))[:-1]
    if hr_type == 'long':
        mask_baseline = ((boundary_sec >= -0.5) * (boundary_sec < 0))[:-1]
        mask_response = ((boundary_sec >= 0) * (boundary_sec < 5))[:-1]
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

ax[i_row, i_col].plot(data['HIGH_RESP']['even_odd']['PSTH']['psth_set'][region][0][0])
ALL_REGIONS = ['bankssts', 'caudalanteriorcingulate', 'caudalmiddlefrontal', 'cuneus', 'entorhinal', 'frontalpole', 'fusiform', 
               'inferiorparietal', 'inferiortemporal', 'insula', 'isthmuscingulate', 'lateraloccipital', 'lateralorbitofrontal',
               'lingual', 'medialorbitofrontal', 'middletemporal', 'mix', 'paracentral', 'parahippocampal', 'parsopercularis',
               'parsorbitalis', 'parstriangularis', 'pericalcarine', 'postcentral', 'posteriorcingulate', 'precentral', 'precuneus',
               'rostralanteriorcingulate', 'rostralmiddlefrontal', 'superiorfrontal', 'superiorparietal', 'superiortemporal',
               'supramarginal', 'temporalpole', 'transversetemporal']

RESPONSIVE_REGIONS = ['cuneus', 'pericalcarine', 'postcentral', 'precentral', 'lingual', 'superiorparietal', 
                      'inferiortemporal', 'middletemporal', 'fusiform', 'lateraloccipital']

EXTENSION_REGIONS = ['bankssts', 'caudalmiddlefrontal', 'inferiorparietal', 'insula', 'lateralorbitofrontal', 'lingual',
                     'parsopercularis', 'parstriangularis', 'precuneus', 'rostralmiddlefrontal', 'superiorfrontal',
                     'superiorparietal', 'superiortemporal', 'superiortemporal']

    

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
    SELECT_BY_REGION = True
    #
    X_CNCT_CNTCTS = True
    X_CNCT_EPOCHS = False
    #
    SEM_BY_CONTACT = True

    NUM_SESSIONS = 1
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


    list_1C_full, list_2C_full, list_3C_full, contact_info_full = copy.copy(list_1C), copy.copy(list_2C), copy.copy(list_3C), copy.copy(contact_info)
    erased_folder_list = []
    slct_masks = dict()
    psth_set = dict()
    for base_region in RESPONSIVE_REGIONS + EXTENSION_REGIONS:
        
        list_1C, list_2C, list_3C, contact_info = copy.copy(list_1C_full), copy.copy(list_2C_full), copy.copy(list_3C_full), copy.copy(contact_info_full)
        print('\n', base_region)

        #
        if SELECT_BY_REGION:
            responsive_list = ['cuneus', 'pericalcarine', 'postcentral', 'precentral', 'lingual',
                            'superiorparietal', 'inferiortemporal', 'middletemporal', 'fusiform', 'lateraloccipital']
            # early_list = ['pericalcarine-R', 'cuneus-R', 'lingual-R', 'lateraloccipital-R', 'pericalcarine-L', 'cuneus-L', 'lingual-L', 'lateraloccipital-L']
            # mid_list = ['fusiform-R', 'inferiortemporal-R', 'parahippocampal-R', 'fusiform-L', 'inferiortemporal-L', 'parahippocampal-L']
            # late_list = ['precuneus-R', 'superiorparietal-R', 'precuneus-L', 'superiorparietal-L']
            #base_region = 'fusiform'
            region_list = [base_region + '-R', base_region + '-L']#['superiorparietal-R', 'superiorparietal-L']#early_list + mid_list
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
        if len(list_1C) == 0:
            print('XXXXX   NO CONTACTS FOR ', base_region)
            continue

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
            data_2C = np.copy(data_1C)
            data_3C = np.copy(data_1C)
        if NUM_SESSIONS >= 2:
            data_2C, cntct_mask_2 = read_epoch_files_by_list(list_2C, first_epoch=0, last_epoch=EPOCHS_TO_READ, norm_per_epoch=True,
                                                            boundary_sec=boundary_sec, random_shift=False, norm_baseline=[5, 15])#[-0.5, -0.05])#
            cntct_mask = cntct_mask * cntct_mask_2
            data_3C = np.copy(data_2C)
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
        

        if X_CNCT_CNTCTS:
            if NUM_SESSIONS == 2:
                data_1C = np.concatenate((data_1C, data_2C), axis=1)
                data_2C, data_3C = data_1C, data_1C
                contact_info = contact_info + contact_info
            if NUM_SESSIONS == 3:
                data_1C = np.concatenate((data_1C, data_2C, data_3C), axis=1)
                data_2C, data_3C = data_1C, data_1C
                contact_info = contact_info + contact_info + contact_info
        if X_CNCT_EPOCHS:
            if NUM_SESSIONS == 2:
                data_1C = np.concatenate((data_1C, data_2C), axis=0)
                data_2C, data_3C = data_1C, data_1C
            if NUM_SESSIONS == 3:
                data_1C = np.concatenate((data_1C, data_2C, data_3C), axis=0)
                data_2C, data_3C = data_1C, data_1C
            
        
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
                    # data_1C_, data_2C_, data_1R_, data_2R_, contact_info_, _ = \
                    #     get_contact_subset(data_1C, data_2C, data_1R, data_2R, contact_info, boundary_sec=boundary_sec, USE=USE, SPLIT=SPLIT)
                    _, _, _, _, _,  _, contact_info_, slct_mask = \
                        get_contact_subset(data_1C, data_2C, data_3C, data_1C, data_2C, data_3C, 
                                           contact_info, boundary_sec=boundary_sec, USE=USE, SPLIT=SPLIT)
                    data_1C_ = data_1C[:, slct_mask]
                    data_2C_ = data_2C[:, slct_mask]
                    data_3C_ = data_3C[:, slct_mask]
                    print(data_1C_.shape, data_2C_.shape, data_3C_.shape, len(contact_info_))
                    slct_masks[base_region] = slct_mask
                #
                
                if (USE != 'ALL') and (len(contact_info_) < 10):
                    continue
                

                for event in ['CNTDWN', 'RECALL']:

                    if event == 'CNTDWN':
                        data_1_, data_2_, data_3_ = data_1C_, data_2C_, data_3C_
                    if event == 'RECALL':
                        if PROCESS_RECALL:
                            data_1_, data_2_ = data_1R_, data_2R_
                        else:
                            continue
                        
                    output_folder = '{}_USE_{}_SPLIT_{}'.format(event, USE, SPLIT)
                    print('\n\n\nworking on', output_folder)
                    #

                    SEM_BY_EPOCH = not SEM_BY_CONTACT
                    if SEM_BY_CONTACT:
                        psth_by_cntct_1 = data_1_.mean(axis=0)
                        psth_all_1 = psth_by_cntct_1.mean(axis=0)
                        psth_all_sem_1 = psth_by_cntct_1.std(axis=0) / np.sqrt(psth_by_cntct_1.shape[0])
                        psth_by_cntct_2 = data_2_.mean(axis=0)
                        psth_all_2 = psth_by_cntct_2.mean(axis=0)
                        psth_all_sem_2 = psth_by_cntct_2.std(axis=0) / np.sqrt(psth_by_cntct_2.shape[0])
                        psth_by_cntct_3 = data_3_.mean(axis=0)
                        psth_all_3 = psth_by_cntct_3.mean(axis=0)
                        psth_all_sem_3 = psth_by_cntct_3.std(axis=0) / np.sqrt(psth_by_cntct_2.shape[0])
                    if SEM_BY_EPOCH:
                        psth_by_epoch_1 = data_1_.mean(axis=1)
                        psth_all_1 = psth_by_epoch_1.mean(axis=0)
                        psth_all_sem_1 = psth_by_epoch_1.std(axis=0) / np.sqrt(psth_by_epoch_1.shape[0])
                        psth_by_epoch_2 = data_2_.mean(axis=1)
                        psth_all_2 = psth_by_epoch_2.mean(axis=0)
                        psth_all_sem_2 = psth_by_epoch_2.std(axis=0) / np.sqrt(psth_by_epoch_2.shape[0])
                        psth_by_epoch_3 = data_3_.mean(axis=1)
                        psth_all_3 = psth_by_epoch_3.mean(axis=0)
                        psth_all_sem_3 = psth_by_epoch_3.std(axis=0) / np.sqrt(psth_by_epoch_2.shape[0])
                    #
                    fig, ax = plt.subplots(1, 1)
                    # ax.bar((boundary_sec[:-1] + boundary_sec[1:]) / 2, psth_all, width=1/V_SAMP_PER_SEC)
                    # ax.bar((boundary_sec[:-1] + boundary_sec[1:]) / 2, 2 * psth_all_sem, bottom=psth_all - psth_all_sem, width=0.5/V_SAMP_PER_SEC, color='k')
                    line1, = ax.plot((boundary_sec[:-1] + boundary_sec[1:]) / 2, np.log(psth_all_1), label='sess 1')
                    ax.fill_between((boundary_sec[:-1] + boundary_sec[1:]) / 2,
                                    np.log(np.maximum(psth_all_1 - psth_all_sem_1, 1e-6)),
                                    np.log(np.maximum(psth_all_1 + psth_all_sem_1, 1e-6)), color=line1.get_color(), alpha=0.2)
                    psth_set[base_region] = [[psth_all_1, psth_all_sem_1]]
                    if NUM_SESSIONS >= 2:
                        line2, = ax.plot((boundary_sec[:-1] + boundary_sec[1:]) / 2, np.log(psth_all_2), label='sess 2')
                        ax.fill_between((boundary_sec[:-1] + boundary_sec[1:]) / 2,
                                        np.log(np.maximum(psth_all_2 - psth_all_sem_2, 1e-6)),
                                        np.log(np.maximum(psth_all_2 + psth_all_sem_2, 1e-6)), color=line2.get_color(), alpha=0.2) 
                        psth_set[base_region].append([psth_all_2, psth_all_sem_2])               
                    if NUM_SESSIONS >= 3:
                        line3, = ax.plot((boundary_sec[:-1] + boundary_sec[1:]) / 2, np.log(psth_all_3), label='sess 3')
                        ax.fill_between((boundary_sec[:-1] + boundary_sec[1:]) / 2,
                                        np.log(np.maximum(psth_all_3 - psth_all_sem_3, 1e-6)),
                                        np.log(np.maximum(psth_all_3 + psth_all_sem_3, 1e-6)), color=line3.get_color(), alpha=0.2)    
                        psth_set[base_region].append([psth_all_3, psth_all_sem_3])            
                    ax.set_ylim((-0.1, 0.3))
                    ax.grid(True)
                    ax.legend()
                    ax.set_title('PSTH   ({}  ,  {} contacts  ,   {} epochs)'.format(base_region, data_1_.shape[1], data_1_.shape[0]))
                    mysavefig(name='PSTH  ({})'.format(base_region), subfolder=output_folder, fig=fig)
                    mysavedata(subfolder=output_folder, name='PSTH', data=dict({'boundary_sec': boundary_sec,
                                                                                'psth_1': psth_all_1, 'psth_sem_1': psth_all_sem_1,
                                                                                'psth_2': psth_all_2, 'psth_sem_2': psth_all_sem_2,
                                                                                'slct_masks': slct_masks, 'psth_set': psth_set}))
                    
                    

                    fig = show_region_distribution(contact_info_, title='{} contacts , delta=T = {} hrs to {} hrs'.format(len(contact_info_), MIN_TGAP, MAX_TGAP))
                    #fig.savefig(os.path.join(os.path.expanduser('~'), 'figs', 'region_distribution.pdf'))
                    mysavefig(name='region_distribution', subfolder=output_folder, fig=fig)
                    with open(os.path.join(os.path.expanduser('~'), 'figs', output_folder, 'contact_data'), 'wb') as fd:
                        pickle.dump({'contact_info': contact_info_}, fd)
                    

 