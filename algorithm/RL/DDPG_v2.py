# coding=utf-8

import torch.nn.functional as func
import numpy as np
import torch

from torch.autograd import Variable
from torch import FloatTensor

from base.env.stock_market import Market
from base.nn.pt.model import BaseRLPTModel
from helper.args_parser import model_launcher_parser


class Algorithm(BaseRLPTModel):

    def __init__(self, env, a_space, s_space, **options):
        super(Algorithm, self).__init__(env, a_space, s_space, **options)

        # Initialize buffer.
        self.buffer = np.zeros((self.buffer_size, self.s_space * 2 + self.a_space + 1))
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

    def predict(self, s):
        a_prob = self.actor_e.forward(Variable(FloatTensor(s)))
        return a_prob.data.numpy()

    def save_transition(self, s, a, r, s_next):
        transition = np.hstack((s, a, [[r]], s_next))
        self.buffer[self.buffer_length % self.buffer_size, :] = transition
        self.buffer_length += 1

    def get_transition_batch(self):
        indices = np.random.choice(self.buffer_size, size=self.batch_size)
        batch = self.buffer[indices, :]
        s = batch[:, :self.s_space]
        a = batch[:, self.s_space: self.s_space + self.a_space]
        r = batch[:, -self.s_space - 1: -self.s_space]
        s_next = batch[:, -self.s_space:]
        return s, a, r, s_next

    def run(self):
        for episode in range(self.episodes):
            self.log_loss(episode)
            s = self.env.reset()
            while True:
                a = self.predict(s)
                a = self.get_a_indices(a)
                s_next, r, status, info = self.env.forward_v1(a)
                a = np.array(a).reshape((1, -1))
                self.save_transition(s, a, r, s_next)
                self.train()
                s = s_next
                if status == self.env.Done:
                    self.env.trader.log_asset(episode)
                    break

    def train(self):
        if self.buffer_length < self.buffer_size:
            return
        # Soft update target actor and target critic.
        self.soft_update_nn()
        # Get sample batch.
        s, a, r, s_n = self.get_transition_batch()
        # Calculate Q-eval.
        q_e = self.critic_e(Variable(FloatTensor(s)), Variable(FloatTensor(a)))
        # Calculate Q-target.
        a_t = self.actor_t(Variable(FloatTensor(s_n), volatile=True))
        q_t = self.critic_t(Variable(FloatTensor(s_n), volatile=True), a_t)
        q_t = Variable(FloatTensor(r), volatile=True) + self.gamma * q_t
        self._train_c(q_e, q_t)
        self._train_a(s)

    def _train_a(self, s):
        self.optimizer_a.zero_grad()
        loss_a = -self.critic_e(Variable(FloatTensor(s)), self.actor_e(Variable(FloatTensor(s)))).mean()
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
        prb_a = func.sigmoid(self.second_dense(phi_s))
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


def main(args):
    env = Market(args.codes)
    algorithm = Algorithm(env, env.trader.action_space, env.data_dim, **{
        # "mode": args.mode,
        # "mode": "test",
        "episodes": 10,
    })
    algorithm.run()


if __name__ == '__main__':
    main(model_launcher_parser.parse_args())

