# coding=utf-8

import numpy as np
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

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self, a):
        pass

    def restore(self):
        pass


class BaseRLPTModel(BasePTModel):

    def __init__(self, env, a_space, s_space, **options):
        super(BaseRLPTModel, self).__init__(env, **options)

        self.env = env

        self.a_space, self.s_space = a_space, s_space

        try:
            self.episodes = options['episodes']
        except KeyError:
            self.episodes = 30

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

        try:
            self.mode = options['mode']
        except KeyError:
            self.mode = 'train'

    def run(self):
        for episode in range(self.episodes):
            self.log_loss(episode)
            s = self.env.reset()
            while True:
                a = self.predict(s)
                a = self.get_a_indices(a)
                s_next, r, status, info = self.env.forward(a)
                a = np.array(a).reshape((1, -1))
                self.save_transition(s, a, r, s_next)
                self.train()
                s = s_next
                if status == self.env.Done:
                    self.env.trader.log_asset(episode)
                    break

    @abstractmethod
    def _init_input(self, *args):
        pass

    @abstractmethod
    def _init_nn(self, *args):
        pass

    @abstractmethod
    def _init_op(self):
        pass

    @abstractmethod
    def save_transition(self, s, a, r, s_n):
        pass

    @abstractmethod
    def log_loss(self, episode):
        pass

    @staticmethod
    def get_a_indices(a):
        a = np.where(a > 1 / 3, 1, np.where(a < - 1 / 3, -1, 0)).astype(np.int32)[0].tolist()
        return a
