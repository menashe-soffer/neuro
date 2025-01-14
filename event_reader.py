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

