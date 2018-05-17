from pysc2.lib import actions
from torch import nn
from torch.nn import functional as F
import torch
from options import FLAGS


class LSTMConvNetwork(nn.Module):
    def __init__(self):
        super(LSTMConvNetwork, self).__init__()
        self.spatial_action_conv = nn.Conv2d(in_channels=64,
                                             out_channels=1,
                                             kernel_size=1,
                                             stride=1
                                             )
        self.action_fc = nn.Linear(in_features=256,
                                   out_features=len(actions.FUNCTIONS)
                                   )

    def forward(self, available_action_ids, lstm_output, fc1):
        spatial_action_logits = self.spatial_action_conv(lstm_output)
        spatial_action_probs = F.softmax(
            spatial_action_logits.view(spatial_action_logits.size(0), -1), dim=1
        )
        action_output = F.softmax(self.action_fc(fc1), dim=1)
        action = action_output * available_action_ids
        action_id_probs = action / torch.sum(action, dim=1, keepdim=True)
        return action_id_probs, spatial_action_probs


class PixelChangeNetwork(nn.Module):
    def __init__(self):
        super(PixelChangeNetwork, self).__init__()
        self.conv_v = nn.Conv2d(in_channels=64,
                                out_channels=1,
                                kernel_size=4,
                                stride=4,
                                padding=0
                                )
        self.conv_a = nn.Conv2d(in_channels=64,
                                out_channels=len(actions.FUNCTIONS),
                                kernel_size=4,
                                stride=4,
                                padding=0)

    def forward(self, lstm_output):
        pc_conv_v = F.relu(self.conv_v(lstm_output))  # -1 1 16 16 #paper中使用反卷积
        pc_conv_a = F.relu(self.conv_a(lstm_output))  # -1 action size 8 8
        conv_a_mean = torch.mean(pc_conv_a, dim=1, keepdim=True)
        pc_q = pc_conv_v + pc_conv_a - conv_a_mean  # -1 action size 8 8
        pc_q_max = torch.max(pc_q, dim=1, keepdim=False)  # -1 8 8
        return pc_q, pc_q_max


class RewardPredictionNetwork(nn.Module):
    def __init__(self):
        super(RewardPredictionNetwork, self).__init__()
        self.fc = nn.Linear(in_features=FLAGS.resolution * FLAGS.resolution * 64, out_features=3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, map_output):
        map_output_flat = map_output.view(map_output.size(0), -1)
        rp = self.softmax(self.fc(map_output_flat))
        return rp
