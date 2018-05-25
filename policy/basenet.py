from pysc2.lib import SCREEN_FEATURES, actions
import torch.nn as nn
from torch.nn import functional as F
from common.convlstm import ConvLSTM

import torch
from options import FLAGS


class BaseConvNet(nn.Module):
    def __init__(self):
        super(BaseConvNet, self).__init__()

        self.emb = nn.Embedding(num_embeddings=SCREEN_FEATURES.unit_type.scale,
                                embedding_dim=FLAGS.unit_type_emb_dim
                                )
        self.screen_conv1 = nn.Conv2d(in_channels=22,
                                      out_channels=16,
                                      kernel_size=5,
                                      stride=1,
                                      padding=2
                                      )
        self.minimap_conv1 = nn.Conv2d(in_channels=9,
                                       out_channels=16,
                                       kernel_size=5,
                                       stride=1,
                                       padding=2
                                       )
        self.screen_conv2 = nn.Conv2d(in_channels=16,
                                      out_channels=32,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1
                                      )
        self.minimap_conv2 = nn.Conv2d(in_channels=16,
                                       out_channels=32,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1
                                       )

    def forward(self, available_action_ids, minimap_numeric,
                player_relative_minimap, player_relative_screen,
                screen_numeric, screen_unit_type):
        units_embedded = self.emb(screen_unit_type)
        # Let's not one-hot zero which is background

        units_embedded = units_embedded.permute(0, 3, 1, 2)
        screen_output = torch.cat(
            (screen_numeric, units_embedded, player_relative_screen),
            dim=1
        )  # 13  5  4
        minimap_output = torch.cat(
            (minimap_numeric, player_relative_minimap),
            dim=1
        )  # 5 4
        screen_output = self.screen_conv1(screen_output)
        screen_output = self.screen_conv2(screen_output)
        minimap_output = self.minimap_conv1(minimap_output)
        minimap_output = self.minimap_conv2(minimap_output)
        map_output = torch.cat([screen_output, minimap_output], dim=1)
        return map_output  # -1 64 32 32


class BaseLSTMNet(nn.Module):
    def __init__(self):
        super(BaseLSTMNet, self).__init__()
        self.lstm = ConvLSTM(input_shape=(FLAGS.resolution, FLAGS.resolution),
                             num_channels=66,
                             kernel_size=3,
                             hidden_size=64,
                             num_layers=2,
                             batch_first=True
                             )
        self.fc1 = nn.Linear(in_features=FLAGS.resolution * FLAGS.resolution * 64, out_features=256)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, map_output, last_action_reward, lstm_state=None):
        map_output = torch.cat([map_output, last_action_reward], dim=1)
        map_output = torch.unsqueeze(map_output, 0)
        # print(lstm_state)
        # 1 -1 64 32 32
        lstm_state, lstm_output = self.lstm(map_output, lstm_state)
        lstm_output_flat = lstm_output.view(lstm_output.size(0), -1)
        fc = self.relu(self.fc1(lstm_output_flat))
        return lstm_output.squeeze(0), fc, lstm_state


class BaseValueNet(nn.Module):
    def __init__(self):
        super(BaseValueNet, self).__init__()
        self.value_fc = nn.Linear(in_features=256, out_features=1)

    def forward(self, x):
        x = torch.squeeze(self.value_fc(x))
        return x


class BasePolicyNet(nn.Module):
    def __init__(self):
        super(BasePolicyNet, self).__init__()
        self.spatial_action_conv = nn.Conv2d(in_channels=64,
                                             out_channels=1,
                                             kernel_size=1,
                                             stride=1
                                             )
        self.action_fc = nn.Linear(in_features=256,
                                   out_features=len(actions.FUNCTIONS)
                                   )

    def forward(self, available_action_ids, lstm_output, fc):
        sp_pi = self.spatial_action_conv(lstm_output)
        sp_pi = F.softmax(
            sp_pi.view(sp_pi.size(0), -1), dim=1
        )
        pi = F.softmax(self.action_fc(fc), dim=1)
        pi = pi * available_action_ids
        pi = pi / torch.sum(pi, dim=1, keepdim=True)
        return pi, sp_pi
