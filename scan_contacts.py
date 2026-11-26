import numpy as np
import pickle
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['MNE_3D_BACKEND'] = 'pyvista'
import pyvista
pyvista.OFF_SCREEN = True


from rdm_tools_new import *
#from permute_digits import digit_permutator
from paths_and_constants import *
from channel_selection_new import select_channels_by_regions#, get_sublist_by_importance
from select_by_periodicity import contact_subset_by_periodicity
import tqdm
import shutil

import mne
from mne.viz import Brain
from nilearn import datasets, plotting

#from itertools import combinations
import argparse
import sys





# def split_session_to_fake_sessions(contact_list, segment_size=3, pair_idx=1):

#     for i_contact, contact in enumerate(contact_list):
#         contact['second'] = contact['first'][int(pair_idx*segment_size):int((pair_idx+1)*segment_size)]
#         contact['first'] = contact['first'][:segment_size]

#     return contact_list

    
    
    




def get_contact_subset(data_1C, data_2C, data_3C, data_1R, data_2R, data_3R, 
                       contact_info, boundary_sec, USE='ALL', SPLIT='ALL'):#, hr_type='short'):

    # # now find "responsive" contacts
    # #if USE != 'ALL':
    # psth_by_cntct = (data_1C.mean(axis=0) + data_2C.mean(axis=0) + data_3C.mean(axis=0)) / 3
    # #rr = np.median(psth_by_cntct[:, 1:], axis=-1) / psth_by_cntct[:, 0]
    # if hr_type == 'short':
    #     mask_baseline = ((boundary_sec >= -0.5) * (boundary_sec < 0))[:-1]
    #     mask_response = ((boundary_sec >= 0.25) * (boundary_sec < 0.75))[:-1]
    # if hr_type == 'long':
    #     mask_baseline = ((boundary_sec >= -0.5) * (boundary_sec < 0))[:-1]
    #     mask_response = ((boundary_sec >= 0) * (boundary_sec < 5))[:-1]
    # if hr_type == 'max':
    #     mask_baseline = ((boundary_sec >= -0.5) * (boundary_sec < 0))[:-1]
    #     mask_response = ((boundary_sec >= 0) * (boundary_sec < 10))[:-1]
    # rr = psth_by_cntct[:, mask_response].mean(axis=-1) / psth_by_cntct[:, mask_baseline].mean(axis=-1)
    if USE == 'ALL':
        use_mask =np.ones(rr.shape, dtype='bool')
    # # if USE == 'NON_RESP':
    # #     rr = np.maximum(rr, 1/rr)
    # #     thd = np.quantile(rr, 1/3)
    # #     use_mask = rr < thd
    # if USE == 'RESP':
    #     thd = np.quantile(rr, 2/3)
    #     thd = np.sort(rr)[::-1][min(rr.size - 1, 200)]
    #     use_mask = rr > thd
    # if USE == 'HIGH_RESP':
    #     #thd = np.quantile(rr, 0.975)
    #     thd = max(np.sort(rr)[::-1][min(rr.size - 1, max(25, int(rr.size / 3)))], 1.1)
    #     if hr_type == 'max':
    #         thd = max(np.sort(rr)[::-1][min(rr.size - 1, max(25, int(rr.size / 3)))], 1.05)
    #     use_mask = rr > thd
    #     print('>>>>> PICKED', use_mask.sum(), 'CONTACTS FOR', USE)
    # if USE == 'NON_RESP':
    #     thd = max(np.sort(rr)[::-1][min(rr.size - 1, max(25, int(rr.size / 3)))], 1.1)
    #     if hr_type == 'max':
    #         thd = max(np.sort(rr)[::-1][min(rr.size - 1, max(25, int(rr.size / 3)))], 1.05)
    #     use_mask = rr <= thd
    #     print('>>>>> PICKED', use_mask.sum(), 'CONTACTS FOR', USE)
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


# def contact_subset_by_consistancy(data_sel_, boundary_sec_, contact_info_, USE):

#     data_sel = resample_epoch(data_sel_, fs=None, tscale=boundary_sec_[:-1], boundary_sec=np.arange(start=0, stop=9.9, step=0.1))
#     nepochs, ncntcts, nsmpls = data_sel.shape
#     c = np.zeros(ncntcts)
#     p = np.zeros(ncntcts)
#     num_epochs = (data_sel.mean(axis=-1) > 1e-12).sum(axis=0)
#     for i_cntct in range(ncntcts):
#         epoch_set = data_sel[:num_epochs[i_cntct], i_cntct]
#         epoch_mean = epoch_set.mean(axis=0)
#         epoch_std = epoch_set.std(axis=0)
#         epoch_sem = epoch_std / np.sqrt(num_epochs[i_cntct])
#         c[i_cntct] = np.linalg.norm(epoch_sem) / np.linalg.norm(epoch_mean)
#         # now calculate correlation with average epoch
#         for i_epoch in range(num_epochs[i_cntct]):
#             p[i_cntct] += pierson(epoch_mean, epoch_set[i_epoch])
#         p[i_cntct] /= num_epochs[i_cntct]
    
#     use_mask = p > 0.45
#     use_mask = np.logical_not(use_mask) if USE == 'NON_RESP' else use_mask
#     use_mask = np.ones_like(use_mask).astype(bool) if USE == 'ALL' else use_mask
#     contact_info_ = [contact_info_[i] for i in np.argwhere(use_mask).flatten().astype(int)]

#     return contact_info, use_mask




ALL_REGIONS = ['bankssts', 'caudalanteriorcingulate', 'caudalmiddlefrontal', 'cuneus', 'entorhinal', 'frontalpole', 'fusiform', 
               'inferiorparietal', 'inferiortemporal', 'insula', 'isthmuscingulate', 'lateraloccipital', 'lateralorbitofrontal',
               'lingual', 'medialorbitofrontal', 'middletemporal', 'mix', 'paracentral', 'parahippocampal', 'parsopercularis',
               'parsorbitalis', 'parstriangularis', 'pericalcarine', 'postcentral', 'posteriorcingulate', 'precentral', 'precuneus',
               'rostralanteriorcingulate', 'rostralmiddlefrontal', 'superiorfrontal', 'superiorparietal', 'superiortemporal',
               'supramarginal', 'temporalpole', 'transversetemporal'][:4]

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



import re
import matplotlib.pyplot as plt
import numpy as np
from nilearn import datasets, plotting
from nilearn.surface import load_surf_mesh

# Standard Desikan-Killiany (aparc) base name -> Destrieux (aparc.a2009s) label(s)
APARC_TO_DESTRIEUX = {
    "bankssts": ["S_temporal_sup", "G_temp_sup-Lateral"],
    "caudalanteriorcingulate": ["G_and_S_cingul-Mid-Ant", "G_and_S_cingul-Ant"],
    "caudalmiddlefrontal": ["G_front_middle", "S_front_sup"],
    "cuneus": ["G_cuneus"],
    "entorhinal": ["G_oc-temp_med-Parahip", "Pole_temporal"],
    "frontalpole": ["G_front_sup", "G_and_S_frontomargin"],
    "fusiform": ["G_oc-temp_lat-fusifor"],
    "inferiorparietal": ["G_pariet_inf-Angular", "G_pariet_inf-Supramar"],
    "inferiortemporal": ["G_temporal_inf"],
    "insula": ["G_Ins_lg_and_S_cent_ins", "G_insular_short"],
    "isthmuscingulate": ["G_cingul-Post-ventral", "G_cingul-Post-dorsal"],
    "lateraloccipital": ["G_occipital_middle", "S_oc_middle_and_Lunatus"],
    "lateralorbitofrontal": ["G_orbital", "G_front_inf-Orbital"],
    "lingual": ["G_oc-temp_med-Lingual"],
    "medialorbitofrontal": ["G_rectus", "G_orbital"],
    "middletemporal": ["G_temporal_middle"],
    "paracentral": ["G_and_S_paracentral"],
    "parahippocampal": ["G_oc-temp_med-Parahip"],
    "parsopercularis": ["G_front_inf-Opercular"],
    "parsorbitalis": ["G_front_inf-Orbital"],
    "parstriangularis": ["G_front_inf-Triangul"],
    "pericalcarine": ["S_calcarine"],
    "postcentral": ["G_postcentral"],
    "posteriorcingulate": ["G_cingul-Post-dorsal", "G_and_S_cingul-Mid-Post"],
    "precentral": ["G_precentral"],
    "precuneus": ["G_precuneus"],
    "rostralanteriorcingulate": ["G_and_S_cingul-Ant"],
    "rostralmiddlefrontal": ["G_front_middle", "G_and_S_frontomargin"],
    "superiorfrontal": ["G_front_sup"],
    "superiorparietal": ["G_parietal_sup"],
    "superiortemporal": ["G_temp_sup-Lateral", "G_temp_sup-Plan_tempo"],
    "supramarginal": ["G_pariet_inf-Supramar"],
    "temporalpole": ["Pole_temporal"],
    "transversetemporal": ["G_temp_sup-G_T_transv"],
}


def _parse_region_and_hemi(raw_name: str):
    """Extract (base_name, hemi) where hemi is 'left', 'right', or None."""
    s = str(raw_name).strip()
    hemi = None
    m = re.search(r"[-_](L|R|LH|RH)$", s, flags=re.IGNORECASE)
    if m:
        tag = m.group(1).lower()
        hemi = "left" if tag.startswith("l") else "right"
        s = s[: m.start()]
    return s.strip(), hemi


def _clean_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def create_region_legend_movie(
    regions,
    stat_map_left=None,
    global_vmax=None,
    out_path="left_hemi_regions.png",):

    if stat_map_left is not None:
        global_vmax = stat_map_left.max() if global_vmax is None else global_vmax
    
    fsaverage = datasets.fetch_surf_fsaverage("fsaverage5")
    destrieux = datasets.fetch_atlas_surf_destrieux()

    coords, _ = load_surf_mesh(fsaverage["infl_left"])
    roi_map = destrieux["map_left"]
    raw_labels = [
        lbl.decode("utf-8") if isinstance(lbl, bytes) else str(lbl)
        for lbl in destrieux["labels"]
    ]

    # Build lookup from cleaned Destrieux label -> region index
    atlas_lookup = {}
    for i, atlas_lbl in enumerate(raw_labels):
        if atlas_lbl in {"Unknown", "Medial_wall"} or not np.any(roi_map == i):
            continue
        atlas_lookup[_clean_key(atlas_lbl)] = (i, atlas_lbl)

    selected_idx = []
    selected_names = []
    unmatched = []

    for req in regions:
        req_str = str(req)
        base_name, hemi = _parse_region_and_hemi(req_str)

        # Keep left-hemisphere (or unspecified hemisphere) regions for the left-hemisphere plot
        if hemi == "right":
            continue

        base_key = _clean_key(base_name)
        candidate_destrieux = APARC_TO_DESTRIEUX.get(base_key, [])

        # Flexible fallback: if not in explicit table, try matching tokens/substrings
        if not candidate_destrieux and base_key:
            for c_key, (_, c_raw) in atlas_lookup.items():
                if base_key in c_key or c_key in base_key:
                    candidate_destrieux.append(c_raw)

        matched_any = False
        for d_name in candidate_destrieux:
            entry = atlas_lookup.get(_clean_key(d_name))
            if entry is not None:
                idx, _ = entry
                if idx not in selected_idx:
                    selected_idx.append(idx)
                    selected_names.append(req_str)
                matched_any = True

        if not matched_any:
            unmatched.append(req_str)

    if unmatched:
        print(f"Unmatched regions ({len(unmatched)}): {unmatched}")

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(8, 8))

    if stat_map_left is not None:
        plotting.plot_surf_stat_map(
            surf_mesh=fsaverage["infl_left"],
            stat_map=stat_map_left,
            hemi="left",
            view="lateral",
            vmax=global_vmax,
            axes=ax,
            colorbar=True,
        )
    else:
        plotting.plot_surf_roi(
            surf_mesh=fsaverage["infl_left"],
            roi_map=np.zeros(coords.shape[0], dtype=int),
            hemi="left",
            view="lateral",
            axes=ax,
            bg_map=fsaverage["sulc_left"],
            cmap="gray",
            alpha=0.0,
        )

    if selected_idx:
        plotting.plot_surf_contours(
            surf_mesh=fsaverage["infl_left"],
            roi_map=roi_map,
            levels=selected_idx,
            labels=selected_names,
            colors=["k"] * len(selected_idx),
            axes=ax,
            legend=False,
        )

        for idx, disp_name in zip(selected_idx, selected_names):
            verts = np.where(roi_map == idx)[0]
            cx, cy, cz = coords[verts].mean(axis=0)
            ax.text(
                cx, cy, cz,
                disp_name,
                fontsize=7,
                color="black",
                weight="bold",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
            )

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


    


def visualize_index_per_region(regions, values, fig_base_name='brain map'):       


    # 1. load/download fsaverage
    fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
    subjects_dir = fs_dir.parent
    subject = 'fsaverage'

    # load Labels for atlas Desikan-Killiany (aparc) for both hemispheres
    labels = mne.read_labels_from_annot(subject, parc='aparc', subjects_dir=subjects_dir)

    # brain = Brain(subject, hemi='both', surf='inflated', subjects_dir=subjects_dir)

    # for i_region, region in enumerate(regions):
    #     for label in labels:
    #         base_name = label.name.rsplit('-', 1)[0]
    #         if region == base_name:
    #             #print(region, label)
    #             brain.add_label(label, color=None, values=values[i_region], colormap='jet')

    # brain.show()
    # brain.save_movie(os.path.join(FIG_FOLDER, 'brain_rotation.mp4'), time_dilation=20, fps=30)

    n_verts = 163842
    stat_map_lh = np.zeros(n_verts)
    stat_map_rh = np.zeros(n_verts)

    for i_region, region in enumerate(regions):
        for label in labels:
            base_name = label.name#.rsplit('-', 1)[0]
            if region == base_name.replace('-rh', '-R').replace('-lh', '-L'):
                val = values[i_region]
                if label.hemi == 'lh':
                    stat_map_lh[label.vertices] = val
                elif label.hemi == 'rh':
                    stat_map_rh[label.vertices] = val

    # INCLUDE_ANNOTATION = True
    # if INCLUDE_ANNOTATION:
    #     # Fetch the Destrieux surface atlas (provides region names and vertex maps)
    #     destrieux = datasets.fetch_atlas_surf_destrieux()

    #     # Region names list (strings) and vertex maps for each hemisphere
    #     region_names = destrieux['labels']
    #     destrieux_left = destrieux['map_left']
    #     destrieux_right = destrieux['map_right']

    fsaverage = datasets.fetch_surf_fsaverage('fsaverage')

    vmin = min(stat_map_rh.min(), stat_map_lh.min())
    vmax = min(stat_map_rh.max(), stat_map_lh.max())

    # left hemisphere
    fig_lh = plotting.plot_surf_stat_map(
        fsaverage.infl_left, stat_map_lh, hemi='left', view='lateral',
        bg_map=fsaverage.sulc_left, colorbar=True, cmap='jet', vmin=vmin, vmax=vmax, cbar_tick_format='%.1f',
        title=fig_base_name + ' Left Hemisphere'#, figure=axes[0]
    )
    fig_lh.savefig(os.path.join(FIG_FOLDER, '{} lh.png'.format(fig_base_name)), dpi=300)
    plt.close()

    # right hemisphere
    fig_rh = plotting.plot_surf_stat_map(
        fsaverage.infl_right, stat_map_rh, hemi='right', view='lateral',
        bg_map=fsaverage.sulc_right, colorbar=True, cmap='jet', vmin=vmin, vmax=vmax, cbar_tick_format='%.1f',
        title=fig_base_name + ' Right Hemisphere'#, figure=axes[1]
    )
    fig_rh.savefig(os.path.join(FIG_FOLDER, '{} rh.png'.format(fig_base_name)), dpi=300)
    plt.close()

    # plt.tight_layout()
    # plt.savefig(os.path.join(FIG_FOLDER, 'brain map.png'), dpi=300)
    # plt.close()





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
    parser.add_argument('--HR_SELECT_TYPE', type=str, default='short', choices=['short', 'max', 'consistancy', 'periodicity'],
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
    # epoch_subsets = [[i*RAW_EPOCH_AVG, (i+1)*RAW_EPOCH_AVG-1] for i in range(int(EPOCHS_TO_READ / RAW_EPOCH_AVG))]
    # epoch_subsets = ['e{}-e{}'.format(i1, i2) for (i1, i2) in epoch_subsets]
    
    

    # prepare contact list
    # stage 1: find suitable contacts

    list_1C, subject_int_id = data_availability_obj.get_all_epoch_files_and_contacts(proc_type='gamma_c_60_160', event_list=['CNTDWN'], min_num_epochs=8)
    

    

    print('A')
        
    #list_1C, list_2C, list_1R, list_2R = list_1C[:1], list_2C[:1], list_1R[:1], list_2R[:1]
        
    contact_info = data_availability_obj.get_contact_info(list_1C)
    print('B')


    list_1C_full, contact_info_full = copy.copy(list_1C), copy.copy(contact_info)
    erased_folder_list = []
    slct_masks = dict()
    psth_set = dict({'ALL': dict(), 'HIGH_RESP': dict(), 'NON_RESP': dict()})

    regions, pwr_01, pwr_1, val_period, val_onset, periodicity_frac, onset_frac, cntct_cnt = [], [], [], [], [], [], [], []


    # create_region_legend_movie(np.array([[r + '-R', r + '-L'] for r in ALL_REGIONS]).flatten())
    # assert False

    for base_region in [str(r) for r in np.array([[r + '-R', r + '-L'] for r in ALL_REGIONS]).flatten()]:#RESPONSIVE_REGIONS + EXTENSION_REGIONS:#RESPONSIVE_REGIONS[8:9]:# + EXTENSION_REGIONS:#
        
        list_1C, contact_info = copy.copy(list_1C_full), copy.copy(contact_info_full)
        print('\n', base_region)

        #
        if SELECT_BY_REGION:
            region_list = [base_region]# + '-R', base_region + '-L']#['superiorparietal-R', 'superiorparietal-L']#early_list + mid_list
            _, contact_info = select_channels_by_regions(contact_info=contact_info, region_list=region_list)
            contact_info_imp = contact_info
        else:
            contact_info_imp = contact_info
        list_1C, _ = data_availability_obj.intersect_contact_list_and_contact_info(contact_list=list_1C, contact_info=contact_info_imp)
        #
        
        boundary_sec = np.arange(start=-5, stop=12+1e-6+3, step=0.01)#1/V_SAMP_PER_SEC)

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
        
        #epoch_count = np.array([epoch_count[i] for i in np.argwhere(cntct_mask).flatten().astype(int)]).astype(int)
        data_1C_per_contact_ave = adaptive_epoch_ave(data_1C, epoch_count, tgt_num_epochs=1)#4)
        epoch_count = epoch_count[cntct_mask]

        ave_for_regions = np.array([data_1C_per_contact_ave[0, i_cntct] * epoch_count[i_cntct] for i_cntct in range(epoch_count.size)]).sum(axis=0) / epoch_count.sum()
        ave_for_regions_full = np.copy(ave_for_regions)
        tmask = ((boundary_sec >= 0) * (boundary_sec < 10))
        tscale = boundary_sec[tmask]
        tmask = tmask[:ave_for_regions.size]
        ave_for_regions = ave_for_regions[tmask]
        #ave_for_regions = np.cos(tscale * 2 * np.pi)


        tfm = np.fft.fft(ave_for_regions)
        tfm = np.real(tfm * np.conj(tfm))[:int(tmask.size / 2)]
        fscale = np.linspace(start=0, stop=1 / np.diff(tscale[:2]), num=tfm.size + 1)[:-1]
        tfm[0] = 0
        tfm = tfm / np.linalg.norm(tfm)
        idx01 = np.argwhere(np.abs(fscale - 0.1).squeeze() < 1e-6).squeeze()
        idx1 = np.argwhere(np.abs(fscale - 1).squeeze() < 1e-6).squeeze()

        # now repeat for isolted contacts
        for i_cntct in range(data_1C_per_contact_ave.shape[1]):
            tfm = np.fft.fft(data_1C_per_contact_ave[0, i_cntct, tmask])
            tfm = np.real(tfm * np.conj(tfm))[:int(tmask.size / 2)]
            tfm[0] = 0
            tfm = tfm / np.linalg.norm(tfm)
            fig, ax = plt.subplots(3, 1, figsize=(9, 9))
            ax[0].plot(tscale, data_1C_per_contact_ave[0, i_cntct, tmask].squeeze())
            ax[0].set_ylabel('PSTH (linear)')
            ax[1].plot(tscale, np.log(data_1C_per_contact_ave[0, i_cntct, tmask].squeeze() + 1e-9).squeeze())
            ax[1].set_ylim((-0.3, 0.3))
            ax[1].set_ylabel('PSTH (log)')
            ax[2].plot(fscale, tfm)
            ax[2].scatter([fscale[idx01], fscale[idx1]], [tfm[idx01], tfm[idx1]], marker='*', c='r')
            ax[2].set_xlim((0, 2.25))
            ax[2].set_ylim((0, 1))#2 * max(tfm[idx01], tfm[idx1])))
            ax[2].set_ylabel('FFT (power)')
            #
            _, dtct, dtct_start, dtct_start_val, dtct_periodicity_val = \
            contact_subset_by_periodicity(data_1C_per_contact_ave[0:1, i_cntct:i_cntct + 1], boundary_sec, 
                                          [contact_info[i_cntct]], 'ALL', gen_debug=False, show_only_positiove=True)
            #
            fig.suptitle(contact_info[i_cntct]['name'] + '   onset: {} period: {}'.format(dtct_start, dtct))
            [ax_.grid(True) for ax_ in ax]

            if (tfm[idx01] > 0.65) or dtct_start:
                mysavefig(name='profile {}   onset-{} period-{}'.format(contact_info[i_cntct]['name'], dtct_start, dtct), subfolder=os.path.join('contact_profiling', base_region), fig=fig)
            plt.close('all')



        # using the old function
        from select_by_periodicity import contact_subset_by_periodicity

        _, _, _, dtct_start_val, dtct_periodicity_val = \
            contact_subset_by_periodicity(ave_for_regions_full[np.newaxis, np.newaxis, :], boundary_sec, 
                                          [contact_info[0]], 'ALL', gen_debug=True, show_only_positiove=True)

        _, dtct, dtct_start, dtct_start_val, dtct_periodicity_val = \
            contact_subset_by_periodicity(data_1C_per_contact_ave, boundary_sec, 
                                          contact_info, 'ALL', gen_debug=True, show_only_positiove=True)
        #print(dtct.sum(), dtct.mean(), dtct_start.sum(), dtct_start.mean())

        if not 'global_onset_dtcts' in locals():
            global_onset_dtcts = 0
        global_onset_dtcts += dtct_start.sum()
        print('>>> ', global_onset_dtcts, ' global onset detections')

        regions.append(base_region)
        pwr_01.append(tfm[idx01])
        pwr_1.append(tfm[idx1])
        val_period.append(dtct_periodicity_val)
        val_onset.append(dtct_start_val)
        periodicity_frac.append(dtct.mean())
        onset_frac.append(dtct_start.mean())
        cntct_cnt.append(len(contact_info))

    
        fig, ax = plt.subplots(3, 1, figsize=(9, 9))
        ax[0].plot(tscale, ave_for_regions)
        ax[0].set_ylabel('PSTH (linear)')
        ax[1].plot(tscale, np.log(ave_for_regions + 1e-9))
        ax[1].set_ylim((-0.3, 0.1))
        ax[1].set_ylabel('PSTH (log)')
        ax[2].plot(fscale, tfm)
        ax[2].scatter([fscale[idx01], fscale[idx1]], [tfm[idx01], tfm[idx1]], marker='*', c='r')
        ax[2].set_xlim((0, 2.25))
        ax[2].set_ylim((0, 1))#2 * max(tfm[idx01], tfm[idx1])))
        ax[2].set_ylabel('FFT (power)')
        fig.suptitle(base_region)

        [ax_.grid(True) for ax_ in ax]

        mysavefig(name='profile {}'.format(base_region), subfolder='contact_profiling', fig=fig)
        plt.close('all')

        assert len(contact_info) == data_1C.shape[1]
        for i_cntct in range(len(contact_info)):
            info_ = contact_info[i_cntct]
            data_ = data_1C[:, i_cntct]
            data_ave = data_.mean()

    pwr_01 = np.array(pwr_01)
    pwr_1 = np.array(pwr_1)
    ord_01 = np.argsort(pwr_01)
    ord_1 = np.argsort(pwr_1)

    print('\nfor 0.1 Hz:\n')
    for i in ord_01:
        print(regions[i], ' :\t', pwr_01[i])
    print('\nfor 1 Hz:\n')
    for i in ord_1:
        #print(regions[i], ' :\t', pwr_1[i])
        print('{} :\t 0.1Hz:  {:4.2f}\t 1Hz: {:4.2f}'.format(regions[i], pwr_01[i], pwr_1[i]))

    # Fetch the Destrieux surface atlas (provides region names and vertex maps)
    destrieux = datasets.fetch_atlas_surf_destrieux()

    # Region names list (strings) and vertex maps for each hemisphere
    region_names = destrieux['labels']
    destrieux_left = destrieux['map_left']
    destrieux_right = destrieux['map_right']

    create_region_legend_movie(np.array([[r + '-R', r + '-L'] for r in ALL_REGIONS]).flatten(), stat_map_left=pwr_1)
    visualize_index_per_region(regions=regions, values=pwr_1, fig_base_name='power 1Hz plain')
    visualize_index_per_region(regions=regions, values=pwr_01, fig_base_name='power 0.1Hz plain')
    # visualize_index_per_region(regions=regions, values=val_period, fig_base_name='periodicity indicator')
    # visualize_index_per_region(regions=regions, values=val_onset, fig_base_name='onset indicator')
    visualize_index_per_region(regions=regions, values=periodicity_frac, fig_base_name='periodicity_frac')
    visualize_index_per_region(regions=regions, values=onset_frac, fig_base_name='onset_frac')
    visualize_index_per_region(regions=regions, values=cntct_cnt, fig_base_name='contact_count')


    