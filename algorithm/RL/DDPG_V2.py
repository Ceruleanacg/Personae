# coding=utf-8

import torch.nn.functional as func
import numpy as np
import torch
import gym

from torch.autograd import Variable
from torch import FloatTensor


class Algorithm(object):

    def __init__(self, a_space, s_space, **options):

        # Initialize evn parameters.
        self.a_space, self.s_space = a_space, s_space

        try:
            self.learning_rate = options['learning_rate']
        except KeyError:
            self.learning_rate = 0.001

        try:
            self.gamma = options['gamma']
        except KeyError:
            self.gamma = 0.9

        try:
            self.tau = options['tau']
        except KeyError:
            self.tau = 0.01

        try:
            self.batch_size = options['batch_size']
        except KeyError:
            self.batch_size = 32

        try:
            self.buffer_size = options['buffer_size']
        except KeyError:
            self.buffer_size = 2000

        self.buffer = np.array([])
        self.buffer_length = 0

        self._init_nn()
        self._init_op()

    def _init_nn(self):
        self.actor_e = ActorNetwork(self.s_space, self.a_space)
        self.actor_t = ActorNetwork(self.s_space, self.a_space)
        self.critic_e = CriticNetwork(self.s_space, self.a_space)
        self.critic_t = CriticNetwork(self.s_space, self.a_space)

    def _init_op(self):
        self.optimizer_a = torch.optim.RMSprop(self.actor_e.parameters(), self.learning_rate)
        self.optimizer_c = torch.optim.RMSprop(self.critic_e.parameters(), self.learning_rate * 2)
        self.loss_c = torch.nn.MSELoss()

    def predict_action(self, s):
        a_prob = self.actor_e.forward(Variable(FloatTensor(s)))
        return a_prob.data.numpy()

    def save_transition(self, s, a, r, s_n):
        buff = [[s, [a], [r], s_n]]
        self.buffer = np.append(self.buffer, buff, axis=0)
        self.buffer_length += 1

    def get_transition_batch(self):
        batch = self.buffer[np.random.choice(self.buffer_size, size=self.batch_size)]
        s, a, r, s_n = batch[0], batch[1], np.array(batch[2]), batch[3]
        return Variable(FloatTensor(s)), Variable(FloatTensor(a)), Variable(FloatTensor(r)), Variable(FloatTensor(s_n))

    def train(self):
        if self.buffer_length < self.buffer_size:
            return
        self.soft_update_nn()
        s, a, r, s_n = self.get_transition_batch()
        q_e = self.critic_e(s, a)
        q_t = r + self.gamma * self.critic_t(s_n, a)
        self._train_a(q_e)
        self._train_c(q_e, q_t)

    def _train_a(self, q_eval):
        loss_a = -torch.mean(q_eval)
        self.optimizer_a.zero_grad()
        loss_a.backward()
        self.optimizer_a.step()

    def _train_c(self, q_eval, q_target):
        loss_c = self.loss_c(q_eval, q_target)
        self.optimizer_c.zero_grad()
        loss_c.backward()
        self.optimizer_c.step()

    def soft_update_nn(self):
        self._soft_update_nn(self.actor_t, self.actor_e)
        self._soft_update_nn(self.critic_t, self.critic_e)

    def _soft_update_nn(self, nn_t, nn_e):
        for p_t, p_e in zip(nn_t.parameters(), nn_e.parameters()):
            p_t.data.copy_(p_t.data * (1.0 - self.tau) + p_e.data * self.tau)


class ActorNetwork(torch.nn.Module):

    def __init__(self, s_space, a_space):
        super(ActorNetwork, self).__init__()
        self.first_dense = torch.nn.Linear(s_space, 50)
        self.second_dense = torch.nn.Linear(50, a_space)

    def forward(self, s):
        phi_s = func.relu(self.first_dense(s))
        prb_a = func.tanh(self.second_dense(phi_s))
        return prb_a


class CriticNetwork(torch.nn.Module):
    def __init__(self, s_space, a_space):
        super(CriticNetwork, self).__init__()
        self.s_dense = torch.nn.Linear(s_space, 50)
        self.a_dense = torch.nn.Linear(a_space, 50)
        self.q_dense = torch.nn.Linear(50, 1)

    def forward(self, s, a):
        phi_s = self.s_dense(s)
        phi_a = self.a_dense(a)
        pre_q = func.relu(phi_s + phi_a)
        q_value = self.q_dense(pre_q)
        return q_value


def run():
    env = gym.make('CartPole-v0')
    env = env.unwrapped
    a_space = env.action_space.n
    s_space = env.observation_space.shape[0]

    RL = Algorithm(a_space, s_space)

    for i_episode in range(400):
        s = env.reset()
        ep_r = 0
        while True:
            if i_episode > 200:
                env.render()

            a_prob = RL.predict_action(s)
            a_index = np.argmax(a_prob)
            s_, r, done, info = env.step(a_index)

            # modify the reward
            x, x_dot, theta, theta_dot = s_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            r = r1 + r2

            a = np.empty(a_space)
            a[a_index] = 1
            RL.save_transition(s, a, r, s_)

            ep_r += r
            RL.train()
            if done:
                print('Ep: ', i_episode,
                      '| Ep_r: ', round(ep_r, 2))
                break
            s = s_


if __name__ == '__main__':
    run()