# coding=utf-8

import logging

from abc import abstractmethod


class BasePTModel(object):

    def __init__(self, env, **options):

        self.env = env

        try:
            self.learning_rate = options['learning_rate']
        except KeyError:
            self.learning_rate = 0.001

        try:
            self.batch_size = options['batch_size']
        except KeyError:
            self.batch_size = 32

        try:
            self.save_path = options["save_path"]
        except KeyError:
            self.save_path = None

        try:
            self.mode = options['mode']
        except KeyError:
            self.mode = 'train'

        try:
            logging.basicConfig(level=options['log_level'])
        except KeyError:
            logging.basicConfig(level=logging.WARNING)

    def restore(self):
        pass


class BaseRLPTModel(BasePTModel):

    def __init__(self, env, a_space, s_space, **options):
        super(BasePTModel, self).__init__()

        self.env = env

        self.a_space, self.s_space = a_space, s_space

        try:
            self.gamma = options['gamma']
        except KeyError:
            self.gamma = 0.9

        try:
            self.tau = options['tau']
        except KeyError:
            self.tau = 0.01

        try:
            self.buffer_size = options['buffer_size']
        except KeyError:
            self.buffer_size = 2000

    @abstractmethod
    def _init_input(self, *args):
        pass

    @abstractmethod
    def _init_nn(self, *args):
        pass

    @abstractmethod
    def _init_op(self):
        pass