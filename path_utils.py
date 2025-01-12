import numpy as np
import os
import glob



def get_paths(base_folder, subject, sess_slct=None, mode='monopolar'):

    assert mode in ['monopolar', 'bipolar']
    paths = []
    sessions = glob.glob(os.path.join(base_folder, subject, 'ses-*'))
    sessions = [os.path.basename(s) for s in sessions]
    for sess in sessions:
        if (sess_slct is None) or (int(sess[-1]) in sess_slct):
            base_name = '{}_{}_task-FR1'.format(subject, sess)
            electrode_name = base_name + '_space-MNI152NLin6ASym_electrodes.tsv'
            electrode_name = os.path.join(base_folder, subject, sess, 'ieeg', electrode_name)
            # if not os.path.isfile(electrode_name):
            #     electrode_name = base_name + '_space-Talairach_electrodes.tsv'
            #     electrode_name = os.path.join(base_folder, subject, sess, 'ieeg', electrode_name)
            event_name = os.path.join(base_folder, subject, sess, 'beh', base_name + '_beh.tsv')
            #signal_name = os.path.join(base_folder, subject, sess, 'ieeg', base_name + '_acq-monopolar_ieeg.edf')
            signal_name = os.path.join(base_folder, subject, sess, 'ieeg', base_name + '_acq-{}_ieeg.edf'.format(mode))
            if os.path.isfile(electrode_name) and os.path.isfile(event_name) and os.path.isfile(signal_name):
                paths.append(dict({'electrodes': electrode_name, 'events': event_name, 'signals': signal_name}))

    return paths


def get_subject_list(base_folder):

    subjects = glob.glob(os.path.join(base_folder, 'sub-*'))
    subjects = [os.path.basename(s) for s in subjects]

    return subjects

