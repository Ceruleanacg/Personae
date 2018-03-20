# coding=utf-8

import tensorflow as tf
import logging

from abc import abstractmethod
from tensorflow.contrib import rnn


class BaseTFModel(object):

    def __init__(self, session, **options):

        # Initialize session.
        self.session = session

        # Initialize parameters.
        self.x, self.label, self.y, self.loss = None, None, None, None

        try:
            self.learning_rate = options["learning_rate"]
        except KeyError:
            self.learning_rate = 0.0003

        try:
            self.train_steps = options["train_steps"]
        except KeyError:
            self.train_steps = 30000

        try:
            self.enable_saver = options["enable_saver"]
        except KeyError:
            self.enable_saver = False

        try:
            self.save_step = options["save_step"]
        except KeyError:
            self.save_step = 1000

        try:
            self.save_path = options["save_path"]
        except KeyError:
            self.save_path = None

        self._init_saver()

    def _init_saver(self):
        if self.enable_saver:
            self.saver = tf.train.Saver()

    def predict(self, x):
        return self.session.run(self.y, feed_dict={self.x: x})

    def evaluate(self, x, label):
        loss, y = self.session.run([self.loss, self.y], feed_dict={self.x: x, self.label: label})
        logging.warning("Evaluation loss is %f" % loss)
        return y

    def save(self, step):
        self.saver.save(self.session, self.save_path)
        logging.warning("Step: {} | Saver reach checkpoint.".format(step))

    def restore(self):
        self.saver.restore(self.session, self.save_path)

    @staticmethod
    def add_rnn(layer_count, hidden_size, cell=rnn.BasicLSTMCell, activation=tf.tanh):
        cells = [cell(hidden_size, activation=activation) for _ in range(layer_count)]
        return rnn.MultiRNNCell(cells)

    @staticmethod
    def add_cnn(x_input, filters, kernel_size, pooling_size):
        convoluted_tensor = tf.layers.conv2d(x_input, filters, kernel_size, padding='SAME', activation=tf.nn.relu)
        return tf.layers.max_pooling2d(convoluted_tensor, pooling_size, strides=[1, 1])

    @staticmethod
    def add_fc(x, units, activation=None):
        return tf.layers.dense(x, units, activation=activation)

    @abstractmethod
    def _init_input(self, *args):
        pass

    @abstractmethod
    def _init_nn(self, *args):
        pass

    @abstractmethod
    def _init_op(self):
        pass
