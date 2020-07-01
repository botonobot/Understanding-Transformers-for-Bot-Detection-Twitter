import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from datasets import _sst2


def news(data_dir, pred_path, log_path):
    preds = pd.read_csv(pred_path, delimiter='\t')['prediction'].values.tolist()
    # _, labels = _news(os.path.join(data_dir, 'newsgroup_test.csv'))
    _, labels = _ding_date(os.path.join(data_dir, 'total_test'), os.path.join(data_dir, 'total_test_label'))
    # _, labels = _ding_date(os.path.join(data_dir, 'new_test.csv'), os.path.join(data_dir, 'new_test_label.csv'))
    test_accuracy = accuracy_score(labels, preds) * 100.
    logs = [json.loads(line) for line in open(log_path)][1:]
    best_validation_index = np.argmax([log['va_acc'] for log in logs])
    valid_accuracy = logs[best_validation_index]['va_acc']
    print('Newsgroup Valid Accuracy: %.2f' % (valid_accuracy))
    print('Newsgroup Test Accuracy:  %.2f' % (test_accuracy))


# news('data/stocknet_basic/','submission/stocknet_basic.csv','log/stocknet_basic.jsonl')


def SST_analyze(data_dir, pred_path, log_path):
    preds = pd.read_csv(pred_path, delimiter='\t')['prediction'].values.tolist()
    _, labels = _sst2(os.path.join(data_dir, 'stsa.binary.test'))
    test_accuracy = accuracy_score(labels, preds) * 100.
    logs = [json.loads(line) for line in open(log_path)][1:]
    best_validation_index = np.argmax([log['va_acc'] for log in logs])
    valid_accuracy = logs[best_validation_index]['va_acc']
    print('SST-2 Valid Accuracy: %.2f' % (valid_accuracy))
    print('SST-2 Test Accuracy:  %.2f' % (test_accuracy))
