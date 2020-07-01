import csv
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def _sst2(path):
    sent = []
    ys = []
    with open(path, 'r') as f:
        for line in f:
            #print(line)
            y, s = line.rstrip('\n').split(' ', 1)
            sent.append(s)
            ys.append(y)
    ys_ = np.array(ys, dtype=int)
    return sent, ys_


def SST2(data_dir):
    teX1, _ = _sst2(os.path.join(data_dir, 'stsa.binary.test'))
    tr_sent, tr_ys = _sst2(os.path.join(data_dir, 'stsa.binary.train'))
    va_sent, va_ys = _sst2(os.path.join(data_dir, 'stsa.binary.dev'))

    return (tr_sent, tr_ys), (va_sent, va_ys), (teX1,)

