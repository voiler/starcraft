import torch as th
from torch.nn import functional as F
from torch import optim
import torch.nn as nn
import numpy as np
from policy import FullyConvPolicy


class DQN(object):
    def __init__(self, unit_type_emb_dim=4, lr=5e-4, policy=FullyConvPolicy):
        self.eval_net, self.target_net = policy(unit_type_emb_dim), policy(unit_type_emb_dim)
        if th.cuda.is_available():
            self.eval_net = self.eval_net.cuda()
            self.target_net = self.target_net.cuda()
        self.learn_step_counter = 0  # for target updating
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=lr)
        self.loss_func = nn.SmoothL1Loss()

    def choose_action(self, x):
        x = Variable(th.unsqueeze(th.FloatTensor(x), 0))
        if th.cuda.is_available():
            x = x.cuda()
        # input only one sample
        if np.random.uniform() < EPSILON:  # greedy
            actions_value = self.eval_net.forward(x)
            if self.cuda:
                action = th.max(actions_value, 1)[1].cuda().data.cpu().numpy()
                action = action[0]
            else:
                action = th.max(actions_value, 1)[1].data.numpy()
                action = action[0]
        else:  # random
            action = np.random.randint(0, self.num_actions)
            action = action
        return action

    def load_state_dict(self, state_dict):
        self.eval_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)

    def save_state_dict(self):
        return self.target_net.state_dict()

    def update_target(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def learn(self, obses_t, actions, rewards, obses_tp1, gamma, batch_size):
        b_s = Variable(th.FloatTensor(obses_t))
        b_a = Variable(th.LongTensor(actions.astype(int)))
        b_r = Variable(th.FloatTensor(rewards))
        b_s_ = Variable(th.FloatTensor(obses_tp1))
        if th.cuda.is_available():
            b_s, b_a, b_r, b_s_ = b_s.cuda(), b_a.cuda(), b_r.cuda(), b_s_.cuda()
        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()  # detach from graph, don't backpropagate
        q_target = b_r + gamma * q_next.max(1)[0].view(batch_size, 1)  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if th.cuda.is_available():
            q_eval, q_target = q_eval.cpu(), q_target.cpu()
        q_eval, q_target = q_eval.data.numpy(), q_target.data.numpy()
        return q_eval - q_target
