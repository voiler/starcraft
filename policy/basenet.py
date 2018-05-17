from pysc2.lib import SCREEN_FEATURES, MINIMAP_FEATURES
from torch import nn
from common.convlstm import ConvLSTM
from common.utils import one_hot_encoding
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
        player_relative_screen_one_hot = one_hot_encoding(
            player_relative_screen, SCREEN_FEATURES.player_relative.scale,
            (FLAGS.resolution, FLAGS.resolution))

        player_relative_minimap_one_hot = one_hot_encoding(
            player_relative_minimap, MINIMAP_FEATURES.player_relative.scale,
            (FLAGS.resolution, FLAGS.resolution))

        units_embedded_nchw = units_embedded.permute(0, 3, 1, 2)
        screen_numeric_all = torch.cat(
            (screen_numeric, units_embedded_nchw, player_relative_screen_one_hot),
            dim=1
        )  # 13  5  4
        minimap_numeric_all = torch.cat(
            (minimap_numeric, player_relative_minimap_one_hot),
            dim=1
        )  # 5 4
        screen_output_1 = self.screen_conv1(screen_numeric_all)
        screen_output_2 = self.screen_conv2(screen_output_1)
        minimap_output_1 = self.minimap_conv1(minimap_numeric_all)
        minimap_output_2 = self.minimap_conv2(minimap_output_1)
        map_output = torch.cat([screen_output_2, minimap_output_2], dim=1)
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

    def forward(self, map_output, last_action_reward, lstm_state=None):
        map_output = torch.cat([map_output, last_action_reward], dim=1)
        map_output = torch.unsqueeze(map_output, 0)
        # print(lstm_state)
        # 1 -1 64 32 32
        lstm_state, lstm_output = self.lstm(map_output, lstm_state)
        return lstm_output.squeeze(0), lstm_state[-1]


class BaseValueNet(nn.Module):
    def __init__(self):
        super(BaseValueNet, self).__init__()
        self.fc1 = nn.Linear(in_features=FLAGS.resolution * FLAGS.resolution * 64, out_features=256)
        self.value_fc = nn.Linear(in_features=256, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, lstm_output):
        lstm_output_flat = lstm_output.view(lstm_output.size(0), -1)
        fc_1 = self.relu(self.fc1(lstm_output_flat))
        value = torch.squeeze(self.value_fc(fc_1))
        return fc_1, value
