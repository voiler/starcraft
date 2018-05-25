from options import FLAGS
from tensorboardX import SummaryWriter
import collections
import os
import numpy as np
import torch as th
from torch.nn import functional as F
from torch import optim
from policy import FullyConvPolicy
from common.utils import select_from_each_row, ravel_index_pairs

SelectedLogProbs = collections.namedtuple("SelectedLogProbs", ["action_id", "spatial", "total"])


class Agent:

    def __init__(self,
                 summary_path: str,
                 spatial_dim: int,
                 clip_epsilon=0.2,
                 unit_type_emb_dim=4,
                 loss_value_weight=1.0,
                 entropy_weight_spatial=1e-6,
                 entropy_weight_action_id=1e-5,
                 optimiser="adam",
                 optimiser_pars: dict = None,
                 policy=FullyConvPolicy
                 ):

        assert optimiser in ["adam", "rmsprop"]
        self.spatial_dim = spatial_dim
        self.loss_value_weight = loss_value_weight
        self.entropy_weight_spatial = entropy_weight_spatial
        self.entropy_weight_action_id = entropy_weight_action_id
        self.summary_path = summary_path
        os.makedirs(summary_path, exist_ok=True)
        self.train_step = 0
        self.clip_epsilon = clip_epsilon
        self.policy = policy(unit_type_emb_dim)
        if th.cuda.is_available():
            self.policy = self.policy.cuda()
        self.writer = SummaryWriter(summary_path)
        opt_class = optim.Adam if optimiser == "adam" else optim.RMSprop
        if optimiser_pars is None:
            pars = {
                "adam": {
                    "params": self.policy.parameters(),
                    "lr": 1e-4,
                    "eps": 5e-7
                },
                "rmsprop": {
                    "params": self.policy.parameters(),
                    "lr": 2e-4
                }
            }[optimiser]
        else:
            pars = optimiser_pars
        self.distribution = th.distributions.Categorical
        self.optimiser = opt_class(**pars)
        self.old_policy = policy(unit_type_emb_dim)
        self.old_policy.eval()
        if th.cuda.is_available():
            self.old_policy.cuda()
        self.update_policy()

    def pre_process(self, obs):
        if th.cuda.is_available():
            return {k: th.from_numpy(v.copy()).cuda() for k, v in obs.items()}
        else:
            return {k: th.from_numpy(v.copy()) for k, v in
                    obs.items()}  # avoid strides of a given numpy array are negative

    def _get_select_action_probs(self, selected_action_id,
                                 action_id_log_probs,
                                 selected_spatial_action_flat,
                                 spatial_action_log_probs):
        action_id = select_from_each_row(
            action_id_log_probs, selected_action_id
        )
        spatial = select_from_each_row(
            spatial_action_log_probs, selected_spatial_action_flat
        )
        total = spatial + action_id

        return SelectedLogProbs(action_id, spatial, total)

    def choise_action(self, pi, sp_pi):
        m = self.distribution(pi)
        pi_action = m.sample()
        sp_m = self.distribution(sp_pi)
        spatial_action = sp_m.sample()
        action_id, spatial_action = \
            pi_action.numpy(), spatial_action.numpy()
        spatial_action_2d = np.array(
            np.unravel_index(spatial_action, (self.spatial_dim,) * 2)
        ).transpose()
        return action_id, spatial_action_2d

    def step(self, obs):
        obs = self.pre_process(obs)
        action_id, spatial_action, value_estimate = self.policy(**obs)
        value = value_estimate.detach()
        if th.cuda.is_available():
            value = value.cpu()
        value = value.numpy()
        action_id, spatial_action = \
            action_id.detach(), spatial_action.detach()
        if th.cuda.is_available():
            action_id, spatial_action = \
                action_id.cpu(), spatial_action.cpu()
        action_id, spatial_action_2d = self.choise_action(action_id, spatial_action)
        return action_id, spatial_action_2d, value

    def train(self, inputs, obs):
        inputs = self.pre_process(inputs)
        obs = self.pre_process(obs)
        advantage = inputs['advantage']
        is_spatial_action_available = inputs['is_spatial_action_available']
        selected_action_id = inputs['selected_action_id']
        selected_spatial_action = inputs['selected_spatial_action']
        value_target = inputs['value_target']

        action_id, spatial_action_probs, value_estimate = self.policy(**obs)
        selected_spatial_action_flat = ravel_index_pairs(
            selected_spatial_action, self.spatial_dim
        )

        spatial_action_log_probs = (
                th.log(th.clamp(spatial_action_probs, 1e-12, 1.0))
                * th.unsqueeze(is_spatial_action_available, 1)
        )
        action_id_log_probs = th.log(th.clamp(action_id, 1e-12, 1.0))
        selected_log_probs = self._get_select_action_probs(selected_action_id,
                                                           action_id_log_probs,
                                                           selected_spatial_action_flat,
                                                           spatial_action_log_probs)
        eps = th.FloatTensor([1e-10])
        if th.cuda.is_available():
            eps = eps.cuda()
        sum_spatial_action_available = th.max(
            eps[0], th.sum(is_spatial_action_available)
        )
        # non-available actions get log(1e-10) value but that's ok because it's never used

        neg_entropy_spatial = th.sum(
            spatial_action_probs * spatial_action_log_probs
        ) / sum_spatial_action_available
        neg_entropy_action_id = th.mean(th.sum(
            action_id * action_id_log_probs, dim=1
        ))
        action_id_old, spatial_action_probs_old, value_estimate_old = self.old_policy(**obs)
        self.update_policy()
        spatial_action_log_probs_old = (
                th.log(th.clamp(spatial_action_probs_old, 1e-12, 1.0))
                * th.unsqueeze(is_spatial_action_available, 1)
        )
        # non-available actions get log(1e-10) value but that's ok because it's never used
        action_id_log_probs_old = th.log(th.clamp(action_id_old, 1e-12, 1.0))

        selected_log_probs_old = self._get_select_action_probs(selected_action_id,
                                                               action_id_log_probs_old,
                                                               selected_spatial_action_flat,
                                                               spatial_action_log_probs_old
                                                               )
        ratio = th.exp(selected_log_probs.total - selected_log_probs_old.total)
        clipped_ratio = th.clamp(
            ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
        )
        l_clip = th.min(
            ratio * advantage,
            clipped_ratio * advantage
        )
        policy_loss = -th.mean(l_clip)
        value_loss = F.mse_loss(value_target, value_estimate)
        loss = policy_loss + \
               value_loss * self.loss_value_weight + \
               neg_entropy_spatial * self.entropy_weight_spatial + \
               neg_entropy_action_id * self.entropy_weight_action_id
        self.writer.add_scalar('loss/loss', loss, self.train_step)
        self.writer.add_scalar('loss/value_loss', value_loss, self.train_step)
        self.writer.add_scalar('loss/policy_loss', policy_loss, self.train_step)
        self.writer.add_scalar('loss/neg_entropy_spatial', neg_entropy_spatial, self.train_step)
        self.writer.add_scalar('loss/neg_entropy_action_id', neg_entropy_action_id, self.train_step)
        print(
            "train step: {} | value loss: {} | policy loss: {} | neg entropy spatial: {} | neg entropy action: {} | loss: {}".format(
                self.train_step, value_loss, policy_loss, neg_entropy_spatial, neg_entropy_action_id, loss))
        self.optimiser.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(self.policy.parameters(), FLAGS.grad_norm_clip)
        self.optimiser.step()
        self.train_step += 1

    def get_value(self, obs):
        obs = self.pre_process(obs)
        _, __, value = self.old_policy(**obs)
        value = value.detach()
        if th.cuda.is_available():
            value = value.cpu()
        value = value.numpy()
        return value

    def save(self, epoch, episode, path):
        state = {
            'epoch': epoch,
            'episode': episode,
            'policy_state_dict': self.policy.state_dict(),
        }
        th.save(state, path)

    def load(self, path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = th.load(path)
        start_epoch = checkpoint['epoch']
        episode = checkpoint['episode']
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        print("=> loaded checkpoint (epoch {})"
              .format(start_epoch + 1))
        return start_epoch, episode

    def update_policy(self):
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.old_policy.eval()
