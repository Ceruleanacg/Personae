# coding=utf-8

import matplotlib.pyplot as plt
import numpy as np
import json
import math
import os

from helper.args_parser import codes
from checkpoints import CHECKPOINTS_DIR

with open(os.path.join(CHECKPOINTS_DIR, 'SL', 'NaiveLSTM', 'model_label.json')) as fp:
    label = np.array(json.load(fp))

with open(os.path.join(CHECKPOINTS_DIR, 'SL', 'NaiveLSTM', 'model_y.json')) as fp:
    y_naive_lstm = np.array(json.load(fp))

with open(os.path.join(CHECKPOINTS_DIR, 'SL', 'DualAttnRNN', 'model_y.json')) as fp:
    y_dual_attn_rnn = np.array(json.load(fp))


row, col = int(math.ceil(len(codes) / 2)), int(math.ceil(len(codes) / 2))
plt.figure(figsize=(20, 15))
for index, code in enumerate(codes):
    plt.subplot(row * 100 + col * 10 + (index + 1))
    plt.title(code)
    plt.plot(label[:, index], label="Real")
    plt.plot(y_naive_lstm[:, index], label="Naive-LSTM")
    plt.plot(y_dual_attn_rnn[:, index], label="DualAttnRNN")
    plt.legend(loc='upper left')
plt.show(dpi=200)


