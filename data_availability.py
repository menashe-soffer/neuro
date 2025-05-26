import copy
import glob
import os

import mne
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import datetime
import tqdm

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
    # generate excel file
    df = pd.DataFrame.from_dict(data).T
    # "flatening" the contact counts - failed to do it so I made it flat in the first place
    writer = pd.ExcelWriter('E:/ds004789-download/data_availability_for_protocol_1.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='.'.join([r[::2][:4] for r in region_list])[:31])
    writer.close()

def process_raw_data(data, ovrd_times=None):

    subjects_list = list(data.keys())

    miss_cnt, hit_cnt, deltas_list = 0, 0, []
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
        sess_names = list(subject_data['sessions'].keys())
        #print('')
        for i_sess, name in enumerate(sess_names):
            if ovrd_times is not None:
                locate = (ovrd_times['subject'].values == subject) * (ovrd_times['session'].values == int(name[-1])) * [ovrd_times['experiment'].values == 'FR1']
                i = np.argwhere(locate.squeeze()).squeeze()
                # subject_data['sessions'][name]['date'] = ovrd_times.iloc[i]['datetime']
                if i.size != 1:
                    miss_cnt += 1
                    print(subject, name, i.size)
                    for i1 in np.argwhere((ovrd_times['subject'].values == subject) * (ovrd_times['session'].values == int(name[-1]))).flatten():
                        print('\t', i1, ovrd_times.iloc[i1]['experiment'])
                else:
                    hit_cnt += 1
                    subject_data['sessions'][name]['date'] = datetime.datetime.strptime(ovrd_times.iloc[i]['datetime'], '%Y-%m-%d %H:%M:%S')
                    print(subject, name, i.size, '--- OK', subject_data['sessions'][name]['date'])
                #subject_data['sessions'][name]['date'] = datetime.datetime.fromtimestamp(ovrd_times.iloc[i]['mstime'] / 1000)# TBD fix timezone
            if i_sess == 0:
                base_date_str = subject_data['sessions'][sess_names[0]]['date']
                base_timestamp = datetime.datetime.timestamp(base_date_str)
            subject_data['sessions'][name]['timestamp'] = datetime.datetime.timestamp(subject_data['sessions'][name]['date'])
            subject_data['sessions'][name]['relative timestamp'] = subject_data['sessions'][name]['timestamp'] - base_timestamp
            #print(subject, name, subject_data['sessions'][name]['date'])
            if subject_data['sessions'][name]['relative timestamp'] != 0:
                print(subject_data['sessions'][name]['relative timestamp'] / 3600)
                deltas_list.append(subject_data['sessions'][name]['relative timestamp'])

    print('miss: {}   hit:  {}'.format(miss_cnt, hit_cnt))
    deltas_list = np.array(deltas_list) / 3600
    deltas_list = deltas_list[(np.abs(deltas_list) < 1000) * (np.abs(deltas_list) >= 1)]
    h = np.histogram(np.array(deltas_list), bins=np.linspace(start=-100, stop=1000, num=1101))
    plt.bar(h[1][1:], h[0])
    plt.grid(True)
    plt.xlabel('hours from sess-0')
    plt.show()

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
            n = session['num countdowns']
            if (n >= countdown_range[0]) and (n <= countdown_range[-1]):
                session_list[session_name] = session

        if (num_bipolar_contacts >= min_total_bipolar) and (len(session_list) >= min_sessions):
            output[subject] = {'bipolar contacts': bipolar_contact_list, 'sessions': session_list}

    return output


class data_availability:

    base_folder = 'E:/ds004789-download'
    processed_folder = 'E:/dr-processed'

    def __init__(self, base_folder='E:/ds004789-download', fixed_data_filename='fixed_availability_data'):

        with open(os.path.join(base_folder, fixed_data_filename), 'rb') as fd:
            self.data = pickle.load(fd)

    def __bipolar_contact_additional_data(self, contact_name, contact_details):

        names = contact_name.split('-')
        ext_data = []
        for name in names:
            try:
                ext_data.append(contact_details[name])
                ext_data[-1]['name'] = name
            except:
                ext_data.append({'region': '', 'coords': None, 'name': name})

        return ext_data




    def get_contacts_for_2_session_gap(self, min_timegap_hrs, max_timegap_hrs, event_type=None, sub_event_type=None, epoch_subset=None):

        # first make flattened session list
        suitable_session_pairs = []
        for subject in self.data:
            subject_data = self.data[subject]
            keys = list(subject_data['sessions'].keys())
            num_sessions = len(keys)
            timegap_matrix_hrs = np.zeros((num_sessions, num_sessions))
            session_dates =  [subject_data['sessions'][sess]['date'] for sess in keys]
            for i1 in range(num_sessions-1):
                for i2 in range(i1+1, num_sessions):
                    try:
                        timegap_matrix_hrs[i1, i2] = (session_dates[i2] - session_dates[i1]).total_seconds() / 3600
                    except:
                        print('fail', subject, keys[i1], keys[i2], session_dates[i2], session_dates[i1])
            #
            suitability_mat = (timegap_matrix_hrs > min_timegap_hrs) * (timegap_matrix_hrs < max_timegap_hrs)
            for i1 in range(num_sessions-1):
                for i2 in range(i1+1, num_sessions):
                    if suitability_mat[i1, i2]:
                        pair = {'subject': subject, 'first': keys[i1], 'second': keys[i2],
                                'delta_hrs': timegap_matrix_hrs[i1, i2], 'contacts': subject_data['bipolar_names'], 'contacts_details': subject_data['electrodes by name']}
                        suitable_session_pairs.append(pair)
                        suitability_mat[i1, :], suitability_mat[:, i1], suitability_mat[i2, :], suitability_mat[:, i2] = False, False, False, False

        # intersect with available evoked files
        if event_type is not None:

            #pattern = os.path.join(self.processed_folder, 'sub-R*', 'ses-*', event_type, '*' + sub_event_type + '-ieeg-evoked-ave.fif')
            if epoch_subset is None:
                pattern = os.path.join(self.processed_folder, 'sub-R*', 'ses-*', event_type, '*bipolar_-' + sub_event_type + '-ieeg-evoked-ave.fif')
            else:
                pattern = os.path.join(self.processed_folder, 'sub-R*', 'ses-*', event_type, '*bipolar_{}--'.format(epoch_subset) + sub_event_type + '-ieeg-evoked-ave.fif')
            evoked_list = glob.glob(pattern)
            revised_pair_list = []
            for pair in suitable_session_pairs:
                subject, first, second = pair['subject'], pair['first'], pair['second']
                # look for the files
                evoked_1, evoked_2 = None, None
                for fname in evoked_list:
                    if (fname.find(subject) > -1):
                        evoked_1 = fname if fname.find(first) > -1 else evoked_1
                        evoked_2 = fname if fname.find(second) > -1 else evoked_2
                if (evoked_1 is not None) and (evoked_2 is not None):
                    pair['first'] = evoked_1
                    pair['second'] = evoked_2
                    revised_pair_list.append(pair)
            suitable_session_pairs = revised_pair_list


        # flatten
        suitable_contacts = []
        for pair in suitable_session_pairs:
            for contact in pair['contacts']:
                contact_data = {'subject': pair['subject'], 'name': contact, 'first': pair['first'], 'second': pair['second'], 'delta_hrs': pair['delta_hrs']}
                contact_data['location'] = self.__bipolar_contact_additional_data(contact, pair['contacts_details'])
                suitable_contacts.append(contact_data)

        return suitable_session_pairs, suitable_contacts






    def get_get_contacts_for_2_session_gap_epoch_splits(self, min_timegap_hrs, max_timegap_hrs, event_type=None, sub_event_type=None, epoch_subsets=None):

        _, contact_list = self.get_contacts_for_2_session_gap(min_timegap_hrs=min_timegap_hrs, max_timegap_hrs=max_timegap_hrs,
                                                              event_type=event_type, sub_event_type=sub_event_type, epoch_subset=epoch_subsets[0])

        for contact in contact_list:
            contact['first'], contact['second'] = [contact['first']], [contact['second']]

        for i_sbst, second_epoch_subset in enumerate(epoch_subsets[1:]):

            _, contact_list2 = self.get_contacts_for_2_session_gap(min_timegap_hrs=min_timegap_hrs, max_timegap_hrs=max_timegap_hrs,
                                                                   event_type=event_type, sub_event_type=sub_event_type, epoch_subset=second_epoch_subset)
            # TBD find intersect of two lists
            combined_list = []
            rplc_pattrn_1, rplc_pattern2 = str(epoch_subsets[0]), str(second_epoch_subset)
            for i_cntct, contact in tqdm.tqdm(enumerate(contact_list)):
                # check if the same contact apears in the second list
                exist2 = False
                for i_cntct2, contact2 in enumerate(contact_list2):
                    if contact['subject'] == contact2['subject']:
                        if contact['name'] == contact2['name']:
                            ok1 = contact['first'][0].replace(rplc_pattrn_1, rplc_pattern2) == contact2['first']
                            ok2 = contact['second'][0].replace(rplc_pattrn_1, rplc_pattern2) == contact2['second']
                            if ok1 and ok2:
                                contact['first'].append(contact2['first'])
                                contact['second'].append(contact2['second'])
                                combined_list.append(contact)
                                break
            contact_list = combined_list

        return contact_list



    # def get_avail_evoked_ave_files(self, major_event, sub_event):
    #
    #     pattern = os.path.join(self.processed_folder, 'sub-R*', 'ses-*', major_event, '*' + sub_event + '-ieeg-evoked-ave.fif')
    #
    #     return glob.glob(pattern)







if __name__ == '__main__':

    # data_availability_obj = data_availability()
    # _, suitable_contacts = data_availability_obj.get_contacts_for_2_session_gap(min_timegap_hrs=72, max_timegap_hrs=96)
    # print(len(suitable_contacts))

    READ_FILES = False

    base_folder = 'E:/ds004789-download'

    start_times = pd.read_csv(os.path.join(base_folder, 'start_times.csv'))

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
    availability_data = process_raw_data(raw_availability_data, ovrd_times=start_times)

    #
    with open(os.path.join(base_folder, 'fixed_availability_data'), 'wb') as fd:
        pickle.dump(availability_data, fd)
    #

    region_list = ['fusiform-R', 'inferiortemporal-R', 'lateraloccipital-R', 'lingual-R', 'fusiform-L', 'inferiortemporal-L', 'lateraloccipital-L', 'lingual-L']
    #hemisphere_sel = ['LR', 'LR', 'LR', 'both']
    admit_list = make_availity_list_by_rules(availability_data, region_list, min_sessions=3, countdown_range=[26, 26], )
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


    # # word repeat statistics
    # word_per_subject_list = dict()
    # for subject in list(admit_list.keys()):
    #     subject_data = admit_list[subject]
    #     admit_sessions = list(subject_data['sessions'].keys())
    #     paths = get_paths(base_folder=base_folder, subject=subject, mode='bipolar',
    #                       sess_slct=[int(a[-1]) for a in admit_sessions])
    #     words_list = []
    #     for path in paths:
    #         print(path['events'])
    #         event_reader_obj = event_reader(path['events'])
    #         event_types = event_reader_obj.df.trial_type.values
    #         event_names = event_reader_obj.df.item_name
    #         words = event_names[event_types == 'WORD'].values
    #         words_list.append(words)
    #     subject_words = set(words_list[0]).intersection(set(words_list[1])).intersection(set(words_list[2]))
    #     word_per_subject_list[subject] = copy.deepcopy(subject_words)
    #
    # n_subjects = len(admit_list)
    # admit_subjects = list(admit_list)
    # hit_mat = np.zeros((n_subjects, n_subjects))
    # for i_sub1 in range(n_subjects - 1):
    #     list1 = word_per_subject_list[admit_subjects[i_sub1]]
    #     if i_sub1 == 0:
    #         common_list = copy.deepcopy(list1)
    #     else:
    #         if i_sub1 != 3:
    #             common_list = common_list.intersection(list1)
    #     for i_sub2 in range(i_sub1, n_subjects):
    #         list2 = word_per_subject_list[admit_subjects[i_sub2]]
    #         hit_mat[i_sub1, i_sub2] = len(list1.intersection(list2))
    # print('here')
