# coding=utf-8

import matplotlib.pyplot as plt
import json
import os

from checkpoints import CHECKPOINTS_DIR


def load_profits(market='stock'):

    with open(os.path.join(CHECKPOINTS_DIR, 'RL', 'DDPG', market, 'model_baseline_profits.json')) as fp:
        p_baseline = json.load(fp)

    with open(os.path.join(CHECKPOINTS_DIR, 'RL', 'DDPG', market, 'model_history_profits.json')) as fp:
        p_ddpg = json.load(fp)

    with open(os.path.join(CHECKPOINTS_DIR, 'RL', 'DoubleDQN', market, 'model_history_profits.json')) as fp:
        p_double_dqn = json.load(fp)

    with open(os.path.join(CHECKPOINTS_DIR, 'RL', 'DuelingDQN', market, 'model_history_profits.json')) as fp:
        p_dueling_dqn = json.load(fp)

    with open(os.path.join(CHECKPOINTS_DIR, 'RL', 'PolicyGradient', market, 'model_history_profits.json')) as fp:
        p_policy_gradient = json.load(fp)

    return p_baseline, p_ddpg, p_double_dqn, p_dueling_dqn, p_policy_gradient


profits_baseline, profits_DDPG, profits_DoubleDQN, profits_DuelingDQN, profits_PG = load_profits('stock')

plt.figure(figsize=(20, 15))
plt.subplot(111)
plt.title("Profits - Baseline")
plt.plot(profits_baseline, label='Baseline')
plt.plot(profits_DDPG, label='DDPG')
plt.plot(profits_DoubleDQN, label='Double-DQN')
plt.plot(profits_DuelingDQN, label='Dueling-DQN')
plt.plot(profits_PG, label='PG')
plt.legend(loc='upper left')
plt.show(dpi=200)
# plt.savefig(save_path, dpi=200)
