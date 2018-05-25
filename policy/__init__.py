from .fullyconv import FullyConvPolicy
from .convlstm import LSTMConvPolicy
from .basenet import BaseConvNet, BaseLSTMNet, BaseValueNet, BasePolicyNet
from .separableconv import SepConvPolicy
from options import Policy


def get_policy(policy_type: str):
    if policy_type == Policy.FullyConv:
        return FullyConvPolicy
    elif policy_type == Policy.ConvLSTM:
        return LSTMConvPolicy
    elif policy_type == Policy.SepConv:
        return SepConvPolicy
