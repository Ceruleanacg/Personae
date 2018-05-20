# coding=utf-8

import tensorflow as tf
import logging
import os

from algorithm import config
from base.env.market import Market
from checkpoints import CHECKPOINTS_DIR
from base.algorithm.model import BaseSLTFModel
from sklearn.preprocessing import MinMaxScaler
from helper.args_parser import model_launcher_parser


class Algorithm(BaseSLTFModel):
    def __init__(self, session, env, seq_length, x_space, y_space, **options):
        super(Algorithm, self).__init__(session, env, **options)

        self.seq_length, self.x_space, self.y_space = seq_length, x_space, y_space

        try:
            self.hidden_size = options['hidden_size']
        except KeyError:
            self.hidden_size = 1

        self._init_input()
        self._init_nn()
        self._init_op()
        self._init_saver()
        self._init_summary_writer()

    def _init_input(self):
        self.rnn_x = tf.placeholder(tf.float32, [None, self.seq_length, self.x_space])
        self.cnn_x = tf.placeholder(tf.float32, [None, self.seq_length, self.x_space, 1])
        self.label = tf.placeholder(tf.float32, [None, self.y_space])

    def _init_nn(self):
        self.rnn = self.add_rnn(1, self.hidden_size)
        self.rnn_output, _ = tf.nn.dynamic_rnn(self.rnn, self.rnn_x, dtype=tf.float32)
        self.rnn_output = self.rnn_output[:, -1]
        # self.cnn_x_input is a [-1, 5, 20, 1] tensor, after cnn, the shape will be [-1, 5, 20, 5].
        self.cnn = self.add_cnn(self.cnn_x, filters=2, kernel_size=[2, 2], pooling_size=[2, 2])
        self.cnn_output = tf.reshape(self.cnn, [-1, self.seq_length * self.x_space * 2])
        self.y_concat = tf.concat([self.rnn_output, self.cnn_output], axis=1)
        self.y_dense = self.add_fc(self.y_concat, 16)
        self.y = self.add_fc(self.y_dense, self.y_space)

    def _init_op(self):
        with tf.variable_scope('loss'):
            self.loss = tf.losses.mean_squared_error(self.y, self.label)
        with tf.variable_scope('train'):
            self.global_step = tf.Variable(0, trainable=False)
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = self.optimizer.minimize(self.loss)
        self.session.run(tf.global_variables_initializer())

    def train(self):
        for step in range(self.train_steps):
            batch_x, batch_y = self.env.get_batch_data(self.batch_size)
            x_rnn, x_cnn = batch_x, batch_x.reshape((-1, self.seq_length, self.x_space, 1))
            _, loss = self.session.run([self.train_op, self.loss], feed_dict={self.rnn_x: x_rnn,
                                                                              self.cnn_x: x_cnn,
                                                                              self.label: batch_y})
            if (step + 1) % 1000 == 0:
                logging.warning("Step: {0} | Loss: {1:.7f}".format(step + 1, loss))
            if step > 0 and (step + 1) % self.save_step == 0:
                if self.enable_saver:
                    self.save(step)

    def predict(self, x):
        return self.session.run(self.y, feed_dict={self.rnn_x: x,
                                                   self.cnn_x: x.reshape(-1, self.seq_length, self.x_space, 1)})


def main(args):

    # mode = args.mode
    mode = "test"
    # codes = args.codes
    codes = ["600036"]
    # codes = ["AU88", "RB88", "CU88", "AL88"]
    market = args.market
    train_steps = args.train_steps
    training_data_ratio = 0.98
    # training_data_ratio = args.training_data_ratio

    env = Market(codes, start_date="2012-01-01", end_date="2018-01-01", **{
        "market": market,
        "use_sequence": True,
        "scaler": MinMaxScaler,
        "mix_index_state": True,
        "training_data_ratio": training_data_ratio,
    })

    model_name = os.path.basename(__file__).split('.')[0]

    algorithm = Algorithm(tf.Session(config=config), env, env.seq_length, env.data_dim, env.code_count, **{
        "mode": mode,
        "hidden_size": 5,
        "enable_saver": True,
        "train_steps": train_steps,
        "enable_summary_writer": True,
        "save_path": os.path.join(CHECKPOINTS_DIR, "SL", model_name, market, "model"),
        "summary_path": os.path.join(CHECKPOINTS_DIR, "SL", model_name, market, "summary"),
    })

    algorithm.run()
    algorithm.eval_and_plot()


if __name__ == '__main__':
    main(model_launcher_parser.parse_args())
