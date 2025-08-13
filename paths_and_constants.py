#
import os
import socket

hostname = socket.gethostname()

if 'wexac' in hostname:
    BASE_FOLDER = os.path.join('/home/labs/malach/sofferme', 'ds004789-download')
    PROC_FOLDER = os.path.join('/home/labs/malach/sofferme', 'dr-processed')
else:
    BASE_FOLDER = 'E:/ds004789-download'
    PROC_FOLDER = 'E:/dr-processed'

EVENT_TYPES = {'CNTDWN': 10, 'DIGIT': 11, 'LIST': 20, 'WORD': 21, 'ORIENT': 30, 'RANDOM': 40, 'RECALL': 50, 'DSTRCT': 60, 'REST': 70,
               10: 'CNTDWN', 11: 'DIGIT', 20: 'LIST', 21: 'WORD', 30: 'ORIENT', 40: 'RANDOM', 50: 'RECALL', 60: 'DSTRCT', 70: 'REST'}


