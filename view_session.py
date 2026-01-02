import numpy as np
import matplotlib.pyplot as plt
import os
import mne
import argparse 
import pickle

from paths_and_constants import *
import path_utils
from my_mne_wrapper import my_mne_wrapper


if __name__ == '__main__':
    
    # Parse command-line arguments for partitioning
    parser = argparse.ArgumentParser(description='Process data folders in parallel.')
    parser.add_argument('--subject', type=str, default='sub-R1001P', help='the sybject')
    parser.add_argument('--session', type=int, default=0, help='ssssion number.')
    
    args = parser.parse_args()
    
    paths = path_utils.get_paths(args.subject, [args.session], mode='bipolar')
    # read the .edf file
    mne_wrapper = my_mne_wrapper()
    mne_wrapper.read_edf_file(paths[0]['signals'])
    mne_wrapper.preprocess()
    assert mne_wrapper.get_mne().info['sfreq'] == 500

    # read the annotations
    annotation_fname = paths[0]['signals'].replace('ieeg/', '').replace('.edf', '.annot').replace(BASE_FOLDER, PROC_FOLDER)
    with open(annotation_fname, 'rb') as fd:
        annotations = pickle.load(fd)
    mne_wrapper.get_mne().info['bads'] = annotations['bads']
    mne_wrapper.get_mne().set_annotations(annotations['annotations'])
    #
    mne_wrapper.get_mne().plot()
    plt.show()
    
    
    
