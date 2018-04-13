# coding=utf-8

import matplotlib.pyplot as plt
import json
import os

from checkpoints import CHECKPOINTS_DIR

with open(os.path.join(CHECKPOINTS_DIR, 'RL', 'DDPG', 'model_baseline_profits.json')) as fp:
    profits_baseline = json.load(fp)

with open(os.path.join(CHECKPOINTS_DIR, 'RL', 'DDPG', 'model_history_profits.json')) as fp:
    profits_DDPG = json.load(fp)

with open(os.path.join(CHECKPOINTS_DIR, 'RL', 'PolicyGradient', 'model_history_profits.json')) as fp:
    profits_PG = json.load(fp)


plt.figure(figsize=(20, 15))
plt.subplot(111)
plt.title("Profits - Baseline")
plt.plot(profits_baseline, label='Baseline')
plt.plot(profits_DDPG, label='DDPG')
plt.plot(profits_PG, label='PG')
plt.legend(loc='upper left')
plt.show(dpi=200)
# plt.savefig(save_path, dpi=200)
