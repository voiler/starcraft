from pysc2.lib import SCREEN_FEATURES, actions, MINIMAP_FEATURES
from torch import nn
from torch.nn import functional as F
from common.convlstm import ConvLSTM
from common.utils import one_hot_encoding
import torch
from options import FLAGS


class LSTMConvPolicy(nn.Module):
    def __init__(self, unit_type_emb_dim):
        super(LSTMConvPolicy, self).__init__()

        self.emb = nn.Embedding(num_embeddings=SCREEN_FEATURES.unit_type.scale,
                                embedding_dim=unit_type_emb_dim
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
        self.spatial_action_conv = nn.Conv2d(in_channels=64,
                                             out_channels=1,
                                             kernel_size=1,
                                             stride=1
                                             )
        self.fc1 = nn.Linear(in_features=FLAGS.resolution * FLAGS.resolution * 64, out_features=256)
        self.lstm = ConvLSTM(input_shape=(FLAGS.resolution, FLAGS.resolution),
                             num_channels=64,
                             kernel_size=3,
                             hidden_size=64,
                             num_layers=2,
                             batch_first=True
                             )
        self.action_fc = nn.Linear(in_features=256,
                                   out_features=len(actions.FUNCTIONS)
                                   )
        self.value_fc = nn.Linear(in_features=256, out_features=1)

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

        # -1 64 32 32
        map_output = torch.unsqueeze(map_output, 0)
        # 1 -1 64 32 32
        lstm_output = self.lstm(map_output)[1].squeeze(0)
        spatial_action_logits = self.spatial_action_conv(lstm_output)
        spatial_action_probs = F.softmax(
            spatial_action_logits.view(spatial_action_logits.size(0), -1), dim=1
        )

        lstm_output_flat = lstm_output.view(lstm_output.size(0), -1)
        fc_1 = F.relu(self.fc1(lstm_output_flat))
        action_output = F.softmax(self.action_fc(fc_1), dim=1)
        value = torch.squeeze(self.value_fc(fc_1))

        action = action_output * available_action_ids
        action_id_probs = action / torch.sum(action, dim=1, keepdim=True)
        return action_id_probs, spatial_action_probs, value
