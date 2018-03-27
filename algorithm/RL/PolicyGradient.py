# coding=utf-8

import tensorflow as tf
import numpy as np

import logging
import os

from algorithm import config
from base.env.finance import Market
from base.nn.tf.model import BaseRLTFModel
from checkpoints import CHECKPOINTS_DIR
from helper.args_parser import model_launcher_parser


class Algorithm(BaseRLTFModel):

    def __init__(self, session, env, a_space, s_space, **options):
        super(Algorithm, self).__init__(session, env, a_space, s_space, **options)

        self.loss = .0

        self.a_buffer = []
        self.r_buffer = []
        self.s_buffer = []

        self._init_input()
        self._init_nn()
        self._init_op()
        self._init_saver()

    def _init_input(self):
        self.a = tf.placeholder(tf.float32, [None, self.a_space])
        self.s = tf.placeholder(tf.float32, [None, self.s_space], 'state')
        self.r = tf.placeholder(tf.float32, [None, ], 'reward')
        self.s_next = tf.placeholder(tf.float32, [None, self.s_space], 'state_next')

    def _init_nn(self):
        # Initialize predict actor and critic.
        w_init, b_init = tf.random_normal_initializer(.0, .3), tf.constant_initializer(0.1)

        with tf.variable_scope('nn'):

            first_dense = tf.layers.dense(self.s,
                                          50,
                                          tf.nn.relu,
                                          kernel_initializer=w_init,
                                          bias_initializer=b_init)

            second_dense = tf.layers.dense(first_dense,
                                           50,
                                           tf.nn.relu,
                                           kernel_initializer=w_init,
                                           bias_initializer=b_init)

            action_prob = tf.layers.dense(second_dense,
                                          self.a_space,
                                          tf.nn.sigmoid,
                                          kernel_initializer=w_init,
                                          bias_initializer=b_init)

            self.a_prob = action_prob

    def _init_op(self):
        with tf.variable_scope('loss'):
            negative_cross_entropy = -tf.reduce_sum(tf.log(self.a_prob) * self.a)
            self.loss_fn = tf.reduce_mean(negative_cross_entropy * self.r)
        with tf.variable_scope('train'):
            self.train_op = tf.train.RMSPropOptimizer(self.learning_rate * 2).minimize(self.loss_fn)
        self.session.run(tf.global_variables_initializer())

    def run(self):
        if self.mode != 'train':
            self.restore()
        else:
            for episode in range(self.episodes):
                self.log_loss(episode)
                s = self.env.reset(self.mode)
                while True:
                    a = self.predict(s)
                    s_next, r, status, info = self.env.forward(a)
                    a = np.array(a).reshape((1, -1))
                    self.save_transition(s, a, r, s_next)
                    s = s_next
                    if status == self.env.Done:
                        self.train()
                        self.env.trader.log_asset(episode)
                        break
                if self.enable_saver and episode % 10 == 0:
                    self.save(episode)

    def train(self):
        _, self.loss = self.session.run([self.train_op, self.loss_fn], {
            self.s: np.array(self.s_buffer),
            self.a: np.array(self.a_buffer),
            self.r: np.array(self.r_buffer)
        })

        self.s_buffer = []
        self.a_buffer = []
        self.r_buffer = []

    def predict(self, s):
        a = self.session.run(self.a_prob, {self.s: s})
        a = self.get_a_indices(a)
        return a

    def save_transition(self, s, a, r, s_next):
        self.s_buffer.append(s.reshape((-1, )))
        self.a_buffer.append(a.reshape((-1, )))
        self.r_buffer.append(r)

    def log_loss(self, episode):
        logging.warning("Episode: {0} | Actor Loss: {1:.2f}".format(episode, self.loss))


def main(args):
    env = Market(args.codes)
    algorithm = Algorithm(tf.Session(config=config), env, env.trader.action_space, env.data_dim, **{
        "mode": args.mode,
        # "mode": "test",
        "episodes": 200,
        "log_level": args.log_level,
        # "log_level": logging.INFO,
        "save_path": os.path.join(CHECKPOINTS_DIR, "RL", "PolicyGradient", "model"),
        "enable_saver": True,
    })
    algorithm.run()
    algorithm.eval_and_plot()


if __name__ == '__main__':
    main(model_launcher_parser.parse_args())


