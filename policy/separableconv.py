import torch
from pysc2.lib import actions, MINIMAP_FEATURES, SCREEN_FEATURES
from torch import nn
from torch.nn import functional as F

from common.utils import one_hot_encoding
from options import FLAGS


class SepConvPolicy(nn.Module):
    def __init__(self, unit_type_emb_dim):
        super(SepConvPolicy, self).__init__()
        self.emb = nn.Embedding(
            num_embeddings=SCREEN_FEATURES.unit_type.scale,
            embedding_dim=unit_type_emb_dim
        )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.screen_conv1 = nn.Conv2d(
            in_channels=22,
            out_channels=22,
            kernel_size=5,
            stride=1,
            padding=2,
            groups=22,
            bias=False
        )
        self.minimap_conv1 = nn.Conv2d(
            in_channels=9,
            out_channels=9,
            kernel_size=5,
            stride=1,
            padding=2,
            groups=9,
            bias=False
        )
        self.screen_conv2 = nn.Conv2d(
            in_channels=22,
            out_channels=22,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=22,
            bias=False
        )
        self.minimap_conv2 = nn.Conv2d(
            in_channels=9,
            out_channels=9,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=9,
            bias=False
        )
        self.spatial_action_conv = nn.Conv2d(
            in_channels=31,
            out_channels=1,
            kernel_size=1,
            stride=1
        )
        self.fc1 = nn.Linear(in_features=31 * FLAGS.resolution * FLAGS.resolution, out_features=256)
        self.action_fc = nn.Linear(
            in_features=256,
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
            (32, 32))

        player_relative_minimap_one_hot = one_hot_encoding(
            player_relative_minimap, MINIMAP_FEATURES.player_relative.scale,
            (32, 32))
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
        spatial_action_logits = self.spatial_action_conv(map_output)
        spatial_action_probs = F.softmax(
            spatial_action_logits.view(spatial_action_logits.size(0), -1),
            dim=1
        )
        map_output_flat = map_output.view(map_output.size(0), -1)
        fc_1 = F.relu(self.fc1(map_output_flat))
        action_output = F.softmax(self.action_fc(fc_1), dim=1)
        value = torch.squeeze(self.value_fc(fc_1))

        action = action_output * available_action_ids
        action_id_probs = action / torch.sum(action, dim=1, keepdim=True)
        return action_id_probs, spatial_action_probs, value
