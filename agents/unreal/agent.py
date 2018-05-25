import collections
import torch as th
import torch.nn.functional as F
from policy import BaseConvNet, BaseLSTMNet, BaseValueNet, BasePolicyNet
from policy.unreal import PixelChangeNetwork, RewardPredictionNetwork
from common.utils import ravel_index_pairs, select_from_each_row
from options import FLAGS
from common.utils import one_hot_encoding
from pysc2.lib import SCREEN_FEATURES, MINIMAP_FEATURES

SelectedLogProbs = collections.namedtuple("SelectedLogProbs", ["action_id", "spatial", "total"])


class Agent(object):
    def __init__(self, thread_index,
                 use_pixel_change,
                 use_value_replay,
                 use_reward_prediction,
                 pixel_change_lambda,
                 entropy_beta):
        self._thread_index = thread_index
        self._use_pixel_change = use_pixel_change
        self._use_value_replay = use_value_replay
        self._use_reward_prediction = use_reward_prediction
        self._pixel_change_lambda = pixel_change_lambda
        self._entropy_beta = entropy_beta
        self.base_conv_net = BaseConvNet()
        if th.cuda.is_available():
            self.base_conv_net.cuda()
        self.base_lstm_net = BaseLSTMNet()
        if th.cuda.is_available():
            self.base_lstm_net.cuda()
        self.base_value_net = BaseValueNet()
        if th.cuda.is_available():
            self.base_value_net.cuda()
        self.base_policy_net = BasePolicyNet()
        if th.cuda.is_available():
            self.base_policy_net.cuda()
        if use_pixel_change:
            self.pc_net = PixelChangeNetwork()
            if th.cuda.is_available():
                self.pc_net.cuda()
        if use_reward_prediction:
            self.rp_net = RewardPredictionNetwork()
            if th.cuda.is_available():
                self.rp_net.cuda()
        self.reset_state()

    def _pre_process(self, obs):
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

    def _base_loss(self, base_pi, base_a, base_sp_pi, base_sp_a,
                   is_spatial_action_available, base_adv, base_r, base_v):
        # [base A3C]
        # Taken action (input for policy)
        selected_spatial_action_flat = ravel_index_pairs(
            base_sp_a, FLAGS.resolution
        )

        spatial_action_log_probs = (
                th.log(th.clamp(base_sp_pi, 1e-12, 1.0))
                * th.unsqueeze(is_spatial_action_available, 1)
        )
        action_id_log_probs = th.log(th.clamp(base_pi, 1e-12, 1.0))
        selected_log_probs = self._get_select_action_probs(base_a,
                                                           action_id_log_probs,
                                                           selected_spatial_action_flat,
                                                           spatial_action_log_probs)
        sum_spatial_action_available = th.max(
            th.FloatTensor([1e-10])[0], th.sum(is_spatial_action_available)
        )
        # non-available actions get log(1e-10) value but that's ok because it's never used

        neg_entropy_spatial = th.sum(
            base_sp_pi * spatial_action_log_probs
        ) / sum_spatial_action_available
        neg_entropy_action_id = th.mean(th.sum(
            base_pi * action_id_log_probs, dim=1
        ))
        policy_loss = -th.mean(selected_log_probs.total * base_adv)
        value_loss = F.mse_loss(base_r, base_v)
        base_loss = policy_loss + \
                    value_loss * FLAGS.loss_value_weight + \
                    neg_entropy_spatial * FLAGS.entropy_weight_spatial + \
                    neg_entropy_action_id * FLAGS.entropy_weight_action_id
        return base_loss

    def _pc_loss(self, pc_a, pc_q, pc_r):
        # [pixel change]

        # Extract Q for taken action
        pc_qa_ = th.mm(pc_q, pc_a)  # -1 action size 16 16
        pc_qa = th.sum(pc_qa_, dim=1, keepdim=False)
        # (-1, 16, 16)

        # TD target for Q

        pc_loss = self._pixel_change_lambda * ((pc_r - pc_qa) ** 2 / 2.)  # lambda * l2 loss
        return pc_loss

    def _vr_loss(self, vr_r, vr_v):
        # R (input for value)

        # Value loss (output)
        vr_loss = (vr_r - vr_v) ** 2 / 2.  # l2 loss
        return vr_loss

    def _rp_loss(self, rp_c, rp_c_target):
        # reward prediction target. one hot vector

        # Reward prediction loss (output)
        rp_c = th.clamp(rp_c, 1e-20, 1.0)
        rp_loss = -th.sum(rp_c_target * th.log(rp_c))
        return rp_loss

    def loss(self, base_pi, base_a, base_sp_pi, base_sp_a,
             is_spatial_action_available, base_adv, base_r, base_v,
             pc_a, pc_q, pc_r,
             vr_r, vr_v,
             rp_c, rp_c_target):
        loss = self._base_loss(base_pi, base_a, base_sp_pi, base_sp_a,
                               is_spatial_action_available, base_adv, base_r, base_v)

        if self._use_pixel_change:
            pc_loss = self._pc_loss(pc_a, pc_q, pc_r)
            loss = loss + pc_loss

        if self._use_value_replay:
            vr_loss = self._vr_loss(vr_r, vr_v)
            loss = loss + vr_loss

        if self._use_reward_prediction:
            rp_loss = self._rp_loss(rp_c, rp_c_target)
            loss = loss + rp_loss

        return loss

    def reset_state(self):
        if th.cuda.is_available():
            self.base_lstm_state_out = [(
                th.zeros(1, 64, FLAGS.resolution, FLAGS.resolution).cuda(),
                th.zeros(1, 64, FLAGS.resolution, FLAGS.resolution).cuda()), ] * 2
        else:
            self.base_lstm_state_out = [(
                th.zeros(1, 64, FLAGS.resolution, FLAGS.resolution),
                th.zeros(1, 64, FLAGS.resolution, FLAGS.resolution)), ] * 2

    def base_net_forward(self, available_action_ids, minimap_numeric,
                         player_relative_minimap, player_relative_screen,
                         screen_numeric, screen_unit_type, last_action_reward, lstm_state=None):
        player_relative_screen = one_hot_encoding(
            player_relative_screen, SCREEN_FEATURES.player_relative.scale,
            (FLAGS.resolution, FLAGS.resolution))

        player_relative_minimap = one_hot_encoding(
            player_relative_minimap, MINIMAP_FEATURES.player_relative.scale,
            (FLAGS.resolution, FLAGS.resolution))
        map_output = self.base_conv_net(available_action_ids, minimap_numeric,
                                        player_relative_minimap, player_relative_screen,
                                        screen_numeric, screen_unit_type)
        lstm_output, lstm_state = self.base_lstm_net(map_output, last_action_reward, lstm_state)
        fc1, value = self.base_value_net(lstm_output)
        pi_out, sp_pi_out = self.base_net(available_action_ids, lstm_output, fc1)
        return pi_out, sp_pi_out, value, lstm_state

    def run_base_policy_and_value(self, obs, last_action_reward):
        obs = self._pre_process(obs)
        last_action_reward = th.from_numpy(last_action_reward)
        if th.cuda.is_available():
            last_action_reward = last_action_reward.cuda()
        pi_out, sp_pi_out, v_out, self.base_lstm_state_out = self.base_net_forward(
            last_action_reward=last_action_reward,
            lstm_state=self.base_lstm_state_out,
            **obs)
        return pi_out, sp_pi_out, v_out

    def run_base_value(self, obs, last_action_reward):
        last_action_reward = th.from_numpy(last_action_reward)
        if th.cuda.is_available():
            last_action_reward = last_action_reward.cuda()
        ___, __, v_out, _ = self.base_net_forward(last_action_reward=last_action_reward,
                                                  lstm_state=self.base_lstm_state_out,
                                                  **obs)
        return v_out

    def pc_net_forward(self, available_action_ids, minimap_numeric,
                       player_relative_minimap, player_relative_screen,
                       screen_numeric, screen_unit_type, last_action_reward, lstm_state=None):
        map_output = self.base_conv_net(available_action_ids, minimap_numeric,
                                        player_relative_minimap, player_relative_screen,
                                        screen_numeric, screen_unit_type)
        lstm_output, lstm_state = self.base_lstm_net(map_output, last_action_reward, lstm_state)
        pc_q, pc_q_max = self.pc_net(lstm_output)
        return pc_q, pc_q_max

    def run_pc_q_max(self, obs, last_action_reward):
        obs = self._pre_process(obs)
        last_action_reward = th.from_numpy(last_action_reward)
        if th.cuda.is_available():
            last_action_reward = last_action_reward.cuda()
        pc_q, pc_q_max = self.pc_net_forward(last_action_reward=last_action_reward, **obs)
        return pc_q, pc_q_max

    def vr_net_forward(self, available_action_ids, minimap_numeric,
                       player_relative_minimap, player_relative_screen,
                       screen_numeric, screen_unit_type, last_action_reward, lstm_state=None):
        map_output = self.base_conv_net(available_action_ids, minimap_numeric,
                                        player_relative_minimap, player_relative_screen,
                                        screen_numeric, screen_unit_type)
        lstm_output, lstm_state = self.base_lstm_net(map_output, last_action_reward, lstm_state)
        _, value = self.base_value_net(lstm_output)
        return value

    def run_vr_value(self, obs, last_action_reward):
        obs = self._pre_process(obs)
        last_action_reward = th.from_numpy(last_action_reward)
        if th.cuda.is_available():
            last_action_reward = last_action_reward.cuda()
        vr_v_out = self.vr_net_forward(last_action_reward=last_action_reward, **obs)
        return vr_v_out

    def rp_net_forward(self, available_action_ids, minimap_numeric,
                       player_relative_minimap, player_relative_screen,
                       screen_numeric, screen_unit_type):
        map_output = self.base_conv_net(available_action_ids, minimap_numeric,
                                        player_relative_minimap, player_relative_screen,
                                        screen_numeric, screen_unit_type)
        rp = self.rp_net(map_output)
        return rp

    def run_rp_c(self, obs):
        # For display tool
        obs = self._pre_process(obs)
        rp_c_out = self.rp_net_forward(**obs)
        return rp_c_out

    def sync_from(self, net):
        self.base_conv_net.load_state_dict(net.base_conv_net)
        self.base_lstm_net.load_state_dict(net.base_lstm_net)
        self.base_value_net.load_state_dict(net.base_value_net)
        self.base_policy_net.load_state_dict(net.base_net.state_dict())
        if self._use_pixel_change:
            self.pc_net.load_state_dict(net.pc_net.state_dict())
        if self._use_reward_prediction:
            self.rp_net.load_state_dict(net.rp_net.state_dict())

    def sync_to(self, net):
        for lp, gp in zip(self.base_conv_net.parameters(), net.base_conv_net.parameters()):
            gp._grad = lp.grad
        for lp, gp in zip(self.base_lstm_net.parameters(), net.base_lstm_net.parameters()):
            gp._grad = lp.grad
        for lp, gp in zip(self.base_value_net.parameters(), net.base_value_net.parameters()):
            gp._grad = lp.grad
        for lp, gp in zip(self.base_policy_net.parameters(), net.base_net.parameters()):
            gp._grad = lp.grad
        if self._use_pixel_change:
            for lp, gp in zip(self.pc_net.parameters(), net.pc_net.parameters()):
                gp._grad = lp.grad
        if self._use_reward_prediction:
            for lp, gp in zip(self.rp_net.parameters(), net.rp_net.parameters()):
                gp._grad = lp.grad

    def save(self, epoch, path):
        state = {
            'epoch': epoch,
            'base_conv_state_dict': self.base_conv_net.state_dict(),
            'base_lstm_state_dict': self.base_lstm_net.state_dict(),
            'base_value_state_dict': self.base_value_net.state_dict(),
            'base_state_dict': self.base_policy_net.state_dict(),
        }
        if self._use_pixel_change:
            state.update({'pc_state_dict': self.pc_net.state_dict()})
        if self._use_reward_prediction:
            state.update({'rp_state_dict': self.rp_net.state_dict()})
        th.save(state, path)

    def load(self, path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = th.load(path)
        start_epoch = checkpoint['epoch']
        self.base_conv_net.load_state_dict(checkpoint['base_conv_state_dict'])
        self.base_lstm_net.load_state_dict(checkpoint['base_lstm_state_dict'])
        self.base_value_net.load_state_dict(checkpoint['base_value_state_dict'])
        self.base_policy_net.load_state_dict(checkpoint['base_state_dict'])
        if self._use_pixel_change:
            self.pc_net.load_state_dict(checkpoint['pc_state_dict'])
        if self._use_reward_prediction:
            self.rp_net.load_state_dict(checkpoint['rp_state_dict'])
        print("=> loaded checkpoint (epoch {})"
              .format(start_epoch + 1))
        return start_epoch

    def share_memory(self):
        self.base_conv_net.share_memory()
        self.base_lstm_net.share_memory()
        self.base_value_net.share_memory()
        self.base_policy_net.share_memory()
        if self._use_pixel_change:
            self.pc_net.share_memory()
        if self._use_reward_prediction:
            self.rp_net.share_memory()
