import numpy as np
import pandas as pd


class event_reader:

    def __init__(self, fname):

        assert fname[-4:] == '.tsv'
        self.df = pd.read_csv(fname, delimiter='\t')

        # onsets = event_df['onset']
        # durations = self.df['duration']
        # satrt_samples = self.df['sample']
        # types = self.df['trial_type']
        # neames = self.df['onset']
        # word_ids = self.df['serialpos']
        # list_ids = self.df['list']
        # tests = self.df['test']  # WHAT THAT IS?
        # responce_times = self.df['response_time']  # WHAT THAT IS?
        # answers = self.df['answer']  # WHAT THAT IS?


    def align_to_sampling_rate(self, old_sfreq=0, new_sfreq=0):

        # save_option = pd.options.mode.copy_on_write
        # pd.options.mode.copy_on_write = True
        for i_line in range(self.df.shape[0]):
            self.df.at[i_line, 'sample'] = int(self.df.iloc[i_line]['sample'] * (new_sfreq / old_sfreq))
        # pd.options.mode.copy_on_write = save_option


    def get_countdowns(self):

        countdowns = []

        countdown_start_idxs = np.argwhere(self.df['trial_type'] == 'COUNTDOWN_START').squeeze()
        countdown_end_idxs = np.argwhere(self.df['trial_type'] == 'COUNTDOWN_END').squeeze()
        for i_cntdn, (start_idx, stop_idx) in enumerate(zip(countdown_start_idxs, countdown_end_idxs)):
            event = dict()
            line = self.df.iloc[start_idx]
            event['onset'] = line['onset']
            event['onset sample'] = line['sample']
            event['interim events'] = None
            for i1 in range(start_idx + 1, stop_idx):
                line = self.df.iloc[i1]
                event_line = '{} \t time {:6.1}  sample {}'.format(line['trial_type'], line['onset'], line['sample'])
                event['interim events'] = [event_line] if event['interim events'] is None else event['interim events'] + [event_line]
            line = self.df.iloc[stop_idx]
            event['end'] = line['onset']
            event['end sample'] = line['sample']
            countdowns.append(event)

        return countdowns


    def get_word_events(self):

        #prectice_event_ids = np.argwhere(self.df['trial_type'] == 'PRACTICE_WORD').flatten()

        def read_word_event(line):

            return({'word': line['item_name'], 'list': line['list'], 'serial': line['serialpos'], 'onset': line['onset'], 'onset sample': line['sample'], 'duration': line['duration']})

        word_event_ids = np.argwhere(self.df['trial_type'] == 'WORD').flatten()
        word_recall_ids =  np.argwhere(self.df['trial_type'] == 'REC_WORD').flatten()

        lists = dict()
        # read all presented words
        for idx in word_event_ids:
            event = read_word_event(self.df.iloc[idx])
            list_id = event['list']
            if not list_id in list(lists.keys()):
                lists[list_id] = {'words': [], 'recall_words': [], 'events': []}
            lists[list_id]['words'].append(event['word'])
            lists[list_id]['events'].append(event)

        # read all recalled words
        for idx in word_recall_ids:
            event = read_word_event(self.df.iloc[idx])
            list_id = event['list']
            assert list_id in list(lists.keys())
            lists[list_id]['recall_words'].append(event['word'])

        # go over lists of presented words and mark each as recalled/not recaled
        for list_id in lists:
            lists[list_id]['recall_mask'] = np.zeros(len(lists[list_id]['words']), dtype=bool)
            for i, word in enumerate(lists[list_id]['words']):
                is_recall = word in lists[list_id]['recall_words']
                lists[list_id]['recall_mask'][i] = is_recall
                lists[list_id]['events'][i]['recall'] = is_recall


        # return lists

        # making a "flat" list (DO I NEED THIS???????)
        event_list = []
        for list_id in lists:
            for event in lists[list_id]['events']:
                event_list.append(event)

        return event_list



    def get_statistics(self):

        cnt_CNTDWN = (self.df['trial_type'] == 'COUNTDOWN_START').values.sum()
        cnt_WORD = (self.df['trial_type'] == 'WORD').values.sum()
        stat = {'total events': self.df.shape[0], 'last onset': self.df['onset'].values[-1], 'cntdwns': cnt_CNTDWN, 'words': cnt_WORD}

        return stat




        # word_dictionary = self.df['item_name'][prectice_event_ids].tolist()
        # word_dictionary = dict()
        # for i, line_id in enumerate(prectice_event_ids):
        #     line = self.df.iloc[line_id]
        #     word = line['item_name']
        #     event = {'onset': line['onset'], 'duration': line['duration'], 'onset sample': line['sample']}
        #     word_dictionary[word] = dict({'list': line['list'], 'serial': i, 'event': event, 'repeats': []})
        #
        # for line_id in word_event_ids:
        #     line = self.df.iloc[line_id]
        #     word = line['item_name']
        #     event = {'onset': line['onset'], 'duration': line['duration'], 'onset sample': line['sample']}
        #     if word in list(word_dictionary.keys()):
        #         word_dictionary[word]['lists'].append({'list': line['list'], 'serial': line['serial'], 'event': event})
        #     else:
        #         pass # TBD: wrong word repeats
        #
        # return word_dictionary

