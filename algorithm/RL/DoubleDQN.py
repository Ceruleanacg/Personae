# coding=utf-8

import tensorflow as tf
import numpy as np
import os

from algorithm import config
from checkpoints import CHECKPOINTS_DIR
from base.env.stock_market import Market
from base.nn.tf.model import BaseRLTFModel
from helper.data_logger import algorithm_logger
from helper.args_parser import model_launcher_parser


class Algorithm(BaseRLTFModel):

    def __init__(self, session, env, a_space, s_space, **options):
        super(Algorithm, self).__init__(session, env, a_space, s_space, **options)

        self.buffer = np.zeros((self.buffer_size, self.s_space + 1 + 1 + self.s_space))
        self.buffer_length = 0

        self.update_q_target_step = 200
        self.critic_loss = 0

        self._init_input()
        self._init_nn()
        self._init_op()
        self._init_saver()
        self._init_summary_writer()

    def _init_input(self, *args):
        self.s = tf.placeholder(tf.float32, [None, self.s_space])
        self.s_next = tf.placeholder(tf.float32, [None, self.s_space])
        self.q_next = tf.placeholder(tf.float32, [None, self.a_space])

    def _init_nn(self, *args):
        self.q_eval = self.__build_critic_nn(self.s, 'q_eval')
        self.q_target = self.__build_critic_nn(self.s_next, 'q_target')

    def _init_op(self):
        self.loss = tf.reduce_mean(tf.squared_difference(self.q_next, self.q_eval))
        self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_eval')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_target')
        self.update_q_target_op = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        self.session.run(tf.global_variables_initializer())

    def train(self):
        # 1. If buffer length is less than buffer size, return.
        if self.buffer_length < self.buffer_size:
            return
        # 2. Update Q-Target if need.
        if self.total_step % self.update_q_target_step == 0:
            self.session.run(self.update_q_target_op)

        # 3. Get transition batch.
        s, a, r, s_next = self.get_transition_batch()

        # 4. Calculate q_eval_next.
        q_eval_next = self.session.run(self.q_eval, {self.s: s_next})

        # 5. Get action indices and make batch indices.
        a_indices = np.argmax(q_eval_next, axis=1)
        b_indices = np.arange(self.batch_size, dtype=np.int)

        # 6. Calculate q_target_next selected by actions.
        q_target_next = self.session.run(self.q_target, {self.s_next: s_next})
        q_target_next_with_a = q_target_next[b_indices, a_indices]

        # 7. Calculate labels.
        q_eval = self.session.run(self.q_eval, {self.s: s})
        q_next = q_eval.copy()
        q_next[b_indices, a.astype(np.int)] = r + self.gamma * q_target_next_with_a

        # 8. Calculate loss.
        _, self.critic_loss = self.session.run([self.train_op, self.loss], {self.s: s, self.q_next: q_next})

        # 9. Increase total step.
        self.total_step += 1

    def predict(self, s):
        q = self.session.run(self.q_eval, {self.s: s})
        a = np.argmax(q)
        return self.get_stock_code_and_action(a, use_greedy=True, use_prob=True if self.mode == 'train' else False)

    def save_transition(self, s, a, r, s_next):
        transition = np.hstack((s, [[a]], [[r]], s_next))
        self.buffer[self.buffer_length % self.buffer_size, :] = transition
        self.buffer_length += 1

    def get_transition_batch(self):
        indices = np.random.choice(self.buffer_size, size=self.batch_size)
        batch = self.buffer[indices, :]
        s = batch[:, :self.s_space]
        a = batch[:, self.s_space: self.s_space + 1]
        r = batch[:, -self.s_space - 1: -self.s_space]
        s_next = batch[:, -self.s_space:]
        return s, a, r, s_next

    def run(self):
        if self.mode != 'train':
            self.restore()
        else:
            for episode in range(self.episodes):
                self.log_loss(episode)
                s = self.env.reset(self.mode)
                while True:
                    c, a, a_index = self.predict(s)
                    s_next, r, status, info = self.env.forward_v2(c, a)
                    self.save_transition(s, a_index, r, s_next)
                    self.train()
                    s = s_next
                    if status == self.env.Done:
                        self.env.trader.log_asset(episode)
                        break
                if self.enable_saver and episode % 10 == 0:
                    self.save(episode)

    def log_loss(self, episode):
        algorithm_logger.warning("Episode: {0} | Critic Loss: {1:.2f}".format(episode, self.critic_loss))

    def __build_critic_nn(self, s, scope):
        w_init, b_init = tf.random_normal_initializer(.0, .3), tf.constant_initializer(.1)
        with tf.variable_scope(scope):
            s_first_dense = tf.layers.dense(s,
                                            32,
                                            activation=tf.nn.relu,
                                            kernel_initializer=w_init,
                                            bias_initializer=b_init)

            s_second_dense = tf.layers.dense(s_first_dense,
                                             32,
                                             tf.nn.relu,
                                             kernel_initializer=w_init,
                                             bias_initializer=b_init)

            q = tf.layers.dense(s_second_dense,
                                self.a_space,
                                kernel_initializer=w_init,
                                bias_initializer=b_init)

            return q


def main(args):
    env = Market(args.codes)
    algorithm = Algorithm(tf.Session(config=config), env, env.trader.action_space, env.data_dim, **{
        # "mode": args.mode,
        "mode": "test",
        "episodes": args.episode,
        "save_path": os.path.join(CHECKPOINTS_DIR, "RL", "DoubleDQN", "model"),
        "summary_path": os.path.join(CHECKPOINTS_DIR, "RL", "DoubleDQN", "summary"),
        "enable_saver": True,
        "enable_summary_writer": True
    })
    algorithm.run()
    algorithm.eval_v2()
    algorithm.plot()


if __name__ == '__main__':
    main(model_launcher_parser.parse_args())
