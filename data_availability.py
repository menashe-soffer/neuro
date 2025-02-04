import copy
import os

import mne
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import datetime

from path_utils import get_subject_list, get_paths
from event_reader import event_reader
from my_montage_reader import my_montage_reader
from my_mne_wrapper import my_mne_wrapper


def make_data_availability_list(base_folder, region_list, hemisphere_sel):

    subject_list = get_subject_list(base_folder=base_folder)
    data = dict()

    for i_subject, subject in enumerate(subject_list):

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
            subject_data['electrodes'] = electrode_list
            total_contacts = np.sum([len(electrode_list[region]) for region in electrode_list])

            # add session info
            subject_data['numsessions'] = len(paths)
            subject_data['sessions'] = dict()
            for i_sess, path in enumerate(paths):
                sess_name = path['events'][path['events'].find('ses-'):][:5]
                #
                event_reader_obj = event_reader(path['events'])
                num_countdowns = len(event_reader_obj.get_countdowns())
                #
                mne_data = mne.io.read_raw_edf(path['signals'], preload=False)
                subject_data['bipolar_names'] = mne_data.ch_names
                #
                subject_data['sessions'][sess_name] = {'date': mne_data.info['meas_date'], 'num countdowns': num_countdowns}

            if total_contacts > 0:
                data[subject] = subject_data
                print(subject, '  ({} / {})'.format(i_subject + 1, len(subject_list)))
            else:
                print('no contacts in ROI for', subject)

        except:
            print('failed for', subject)

    return data
    # # generate excel file
    # df = pd.DataFrame.from_dict(data).T
    # # "flatening" the contact counts - failed to do it so I made it flat in the first place
    # writer = pd.ExcelWriter('E:/ds004789-download/data_availability_for_protocol_1.xlsx', engine='xlsxwriter')
    # df.to_excel(writer, sheet_name='.'.join([r[::2][:4] for r in region_list])[:31])
    # writer.close()

def process_raw_data(data):

    subjects_list = list(data.keys())

    for subject in subjects_list:
        subject_data = data[subject]

        # map bipolar contacts to regions
        # lets create dictionary by electrode name
        electrodes = dict()
        bipolar_contacts = dict()
        for region in list(subject_data['electrodes'].keys()):
            for electrode in subject_data['electrodes'][region]:
                electrodes[electrode['name']] = {'region': region, 'coords': electrode['coords']}
            bipolar_contacts[region] = [] # we will fill that in next loop
        subject_data['electrodes by name'] = copy.deepcopy(electrodes)
        # maps
        bipolar_contacts['mixed'], bipolar_contacts['unresoved'] = [], []
        for name in subject_data['bipolar_names']:
            name1, name2 = name.split('-')
            try:
                el1, el2 = electrodes[name1], electrodes[name2]
            except:
                bipolar_contacts['unresoved'].append(name)
            not_boundary = el1['region'] == el2['region']
            if not_boundary:
                bipolar_contacts[el1['region']].append({'name1': name1, 'name2': name2, 'coords1': el1['coords'], 'coords2': el2['coords']})
            else:
                bipolar_contacts['mixed'].append([el1, el2])
        subject_data['bipolar contacts'] = copy.deepcopy(bipolar_contacts)


        # add time between sessions
        # TBD
        sess_names = list(subject_data['sessions'].keys())
        base_date_str = subject_data['sessions'][sess_names[0]]['date']
        base_timestamp = datetime.datetime.timestamp(base_date_str)
        #print('')
        for name in sess_names:
            subject_data['sessions'][name]['timestamp'] = datetime.datetime.timestamp(subject_data['sessions'][name]['date'])
            subject_data['sessions'][name]['relative timestamp'] = subject_data['sessions'][name]['timestamp'] - base_timestamp
            #print(subject, name, subject_data['sessions'][name]['date'])

    return data


def make_availity_list_by_rules(data, region_list, min_sessions=3, countdown_range=[26, 26], min_total_bipolar=1):

    output = dict()
    for subject in list(data.keys()):
        print(subject)
        subject_data = data[subject]
        session_list, bipolar_contact_list = dict(), dict()
        num_bipolar_contacts = 0
        for region in region_list:
            n = len(subject_data['bipolar contacts'][region])
            if n >= 1:
                num_bipolar_contacts += n
                bipolar_contact_list[region] = subject_data['bipolar contacts'][region]

        for session_name in subject_data['sessions']:
            session = subject_data['sessions'][session_name]
            n = session['num_countdowns']
            if (n >= countdown_range[0]) and (n <= countdown_range[-1]):
                session_list[session_name] = session

        if (num_bipolar_contacts >= min_total_bipolar) and (len(session_list) >= min_sessions):
            output[subject] = {'bipolar contacts': bipolar_contact_list, 'sessions': session_list}

    return output


READ_FILES = False

base_folder = 'E:/ds004789-download'

# area of interest definition
all_regions = ['bankssts', 'caudalanteriorcingulate', 'caudalmiddlefrontal', 'cuneus', 'entorhinal', 'frontalpole', 'fusiform',
               'inferiorparietal', 'inferiortemporal', 'insula', 'isthmuscingulate', 'lateraloccipital', 'lateralorbitofrontal',
               'lingual', 'medialorbitofrontal', 'middletemporal', 'nan', 'paracentral', 'parahippocampal', 'parsopercularis',
               'parsorbitalis', 'parstriangularis', 'pericalcarine', 'postcentral', 'posteriorcingulate', 'precentral', 'precuneus',
               'rostralanteriorcingulate', 'rostralmiddlefrontal', 'superiorfrontal', 'superiorparietal', 'superiortemporal',
               'supramarginal', 'temporalpole', 'transversetemporal']
all_regions_hemisphere_sel = ['LR' for region in all_regions]
# create table of available data
if READ_FILES:
    subjects_list = get_subject_list(base_folder=base_folder)
    raw_availability_data = make_data_availability_list(base_folder=base_folder, region_list=all_regions, hemisphere_sel=all_regions_hemisphere_sel)
    with open(os.path.join(base_folder, 'raw_availability_data'), 'wb') as fd:
        pickle.dump(raw_availability_data, fd)
    #assert False

# process the files
with open(os.path.join(base_folder, 'raw_availability_data'), 'rb') as fd:
    raw_availability_data = pickle.load(fd)
availability_data = process_raw_data(raw_availability_data)


region_list = ['fusiform-R', 'inferiortemporal-R', 'lateraloccipital-R', 'lingual-R', 'fusiform-L', 'inferiortemporal-L', 'lateraloccipital-L', 'lingual-L']
#hemisphere_sel = ['LR', 'LR', 'LR', 'both']
admit_list = make_availity_list_by_rules(availability_data, region_list)
print(admit_list.keys())


# # temporary statistics
# gap_after_countdown, countdown_to_word = [], []
# next_list, second_next_list = [], []
# for subject in list(admit_list.keys()):
#
#     subject_data = admit_list[subject]
#     admit_sessions = list(subject_data['sessions'].keys())
#     paths = get_paths(base_folder=base_folder, subject=subject, mode='bipolar', sess_slct=[int(a[-1]) for a in admit_sessions])
#     for path in paths:
#         event_filename = path['events']
#         session_name = event_filename[event_filename.find('ses-'):][:5]
#
#         event_reader_obj = event_reader(path['events'])
#         event_types = event_reader_obj.df.trial_type.values
#         times = event_reader_obj.df.onset.values
#         #
#         idxs = np.argwhere(event_types == 'COUNTDOWN_END').squeeze()
#         gap_after_countdown = gap_after_countdown + list(times[idxs+1] - times[idxs])
#         for idx in idxs:
#             #print(idx)
#             next_word_idx = idx + 1 + np.argwhere(event_types[idx + 1:] == 'WORD').squeeze()
#             next_practice_word_idx = idx + 1 + np.argwhere(event_types[idx + 1:] == 'PRACTICE_WORD').squeeze()
#             next_word_idx = (np.concatenate((next_word_idx, next_practice_word_idx))).min()
#             countdown_to_word.append(times[next_word_idx] - times[idx])
#             next_list.append(event_types[idx + 1] + ', ' + event_types[idx+2])
#             #second_next_list.append(event_types[idx + 2])
#         print(subject, session_name, len(gap_after_countdown), len(countdown_to_word))
#
# next_pair_types = np.unique(next_list)
# for next_event_pair in next_pair_types:
#     print(next_event_pair, '   ({})'.format(np.sum([next_event_pair == e for e in next_list])))
#
# fig, ax = plt.subplots(1, 2, figsize=(8, 4))
# h, bins = np.histogram(countdown_to_word, bins=100)
# bins = (bins[:-1] + bins[1:]) / 2
# ax[0].bar(bins, h, width=0.1)
# h, bins = np.histogram(gap_after_countdown, bins=100)
# bins = (bins[:-1] + bins[1:]) / 2
# ax[1].bar(bins, h, width=0.1)
# plt.show()
