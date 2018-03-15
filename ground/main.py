# coding=utf-8

import tensorflow as tf
import numpy as np

from agent.DDPG import Algorithm
from base.env.finance import Market, MarketStatus


if __name__ == '__main__':

    codes = ["600036", "601998"]
    market = Market(codes)

    agent = Algorithm(tf.Session(), market.trader.action_space, market.data_dim)

    for episode in range(300):
        market.trader.log_asset(episode)
        s = market.reset()
        while True:
            a = agent.predict_action(s)
            a_indices = np.where(a > 1 / 3, 1, np.where(a < - 1 / 3, -1, 0)).astype(np.int32)[0].tolist()
            s_next, r, status, info = market.forward(a_indices)
            agent.save_transition(s, a, r, s_next)
            agent.train_if_need()
            s = s_next
            if status == MarketStatus.NotRunning:
                break
