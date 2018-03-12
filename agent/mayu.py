# coding=utf-8

import tensorflow as tf


class Algorithm(object):

    def __init__(self, session, action_space, state_space, learning_rate=0.002, gamma=0.9, tau=0.01,  **options):

        # Initialize session.
        self.session = session

        # Initialize parameters.
        self.state_space, self.action_space = state_space, action_space
        self.learning_rate, self.gamma, self.tau = learning_rate, gamma, tau

        # Initialize options.
        try:
            self.batch_size = options['batch_size']
        except KeyError:
            self.batch_size = 16

        try:
            self.buffer_size = options['buffer_size']
        except KeyError:
            self.buffer_size = 200
