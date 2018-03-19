# coding=utf-8

import tensorflow as tf
import logging

from base.SL.nn import BaseTFModel


class Algorithm(BaseTFModel):
    def __init__(self, session,  hidden_size, seq_length, x_space, **options):
        super(Algorithm, self).__init__(session, **options)
        self._init_input(seq_length, x_space)
        self._init_nn(hidden_size, x_space)
        self._init_op()

    def _init_input(self, *args):
        seq_length, x_space = args[0], args[1]
        self.x = tf.placeholder(tf.float32, [None, seq_length, x_space])
        self.label = tf.placeholder(tf.float32, [None, 1])

    def _init_nn(self, *args):
        hidden_size, x_dim = args[0], args[1]
        # First Attn
        with tf.variable_scope("1st_encoder"):
            self.f_encoder_rnn = self.add_rnn(1, hidden_size)
            self.f_encoder_outputs, _ = tf.nn.dynamic_rnn(self.f_encoder_rnn, self.x, dtype=tf.float32)
            self.f_attn_inputs = tf.nn.softmax(self.f_encoder_outputs)
            self.f_attn_outputs = self.add_fc(self.f_attn_inputs, hidden_size, tf.tanh)
        with tf.variable_scope("1st_decoder"):
            self.f_decoder_input = tf.matmul(self.f_encoder_outputs, self.f_attn_outputs, transpose_a=True)
            self.f_decoder_rnn = self.add_rnn(1, hidden_size)
            self.f_decoder_outputs, _ = tf.nn.dynamic_rnn(self.f_decoder_rnn, self.f_decoder_input, dtype=tf.float32)
        # Second Attn
        with tf.variable_scope("2nd_encoder"):
            self.s_attn_input = tf.nn.softmax(self.f_decoder_outputs)
            self.s_attn_outputs = self.add_fc(self.s_attn_input, hidden_size, tf.tanh)
        with tf.variable_scope("2nd_decoder"):
            self.s_decoder_input = tf.matmul(self.f_decoder_outputs, self.s_attn_outputs, transpose_a=True)
            self.s_decoder_rnn = self.add_rnn(2, hidden_size)
            self.f_decoder_outputs, _ = tf.nn.dynamic_rnn(self.s_decoder_rnn, self.s_decoder_input, dtype=tf.float32)
            self.y = self.add_fc(self.f_decoder_outputs[:, -1], 1, tf.tanh)

    def _init_op(self):
        self.loss = tf.losses.mean_squared_error(self.y, self.label)
        self.global_step = tf.Variable(0, trainable=False)
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)
        self.session.run(tf.global_variables_initializer())

    def train(self, batch_x, batch_y):
        for step in range(self.train_steps):
            _, loss = self.session.run([self.train_op, self.loss], feed_dict={self.x: batch_x, self.label: batch_y})
            if (step + 1) % 1000 == 0:
                logging.warning("Step: {0} | Loss: {1:.2f}".format(step + 1, loss))
            if step > 0 and (step + 1) % self.save_step == 0:
                if self.enable_saver:
                    self.save(step)


if __name__ == '__main__':
    pass
