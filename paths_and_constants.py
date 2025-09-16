#
import os
import socket


hostname = socket.gethostname()

if 'wexac' in hostname:
    BASE_FOLDER = os.path.join('/home/labs/malach/sofferme', 'ds004789-download')
    PROC_FOLDER = os.path.join('/home/labs/malach/sofferme', 'dr-processed')
    LOG_FOLDER = os.path.join('/home/labs/malach/sofferme', 'logs')
    TEMP_FOLDER = os.path.join('/home/labs/malach/sofferme', 'dr-temp')
    HOME_DIR = os.path.expanduser('~')
    IS_CLUSTER = True
else:
    BASE_FOLDER = 'E:/ds004789-download'
    PROC_FOLDER = 'E:/dr-processed'
    LOG_FOLDER = 'E:/logs'
    TEMP_FOLDER = 'E:/dr-temp'
    HOME_DIR = 'E:'
    IS_CLUSTER = False

EVENT_TYPES = {'CNTDWN': 10, 'DIGIT': 11, 'LIST': 20, 'WORD': 21, 'ORIENT': 30, 'RANDOM': 40, 'RECALL': 50, 'DSTRCT': 60, 'REST': 70,
               10: 'CNTDWN', 11: 'DIGIT', 20: 'LIST', 21: 'WORD', 30: 'ORIENT', 40: 'RANDOM', 50: 'RECALL', 60: 'DSTRCT', 70: 'REST'}


