import numpy as np
import torch as th
import torch.nn as nn


class Mode:
    A2C = "a2c"
    PPO = "ppo"
    UNREAL = "unreal"


def select_from_each_row(params, indices):
    indices = indices.long()
    if th.cuda.is_available():
        indices = indices.cuda()
    indices = indices.resize_(indices.shape[0], 1)
    return th.gather(params, 1, indices)


def calculate_n_step_reward(
        one_step_rewards: np.ndarray,
        discount: float,
        last_state_values: np.ndarray):
    discount = discount ** np.arange(one_step_rewards.shape[1], -1, -1)
    reverse_rewards = np.c_[one_step_rewards, last_state_values][:, ::-1]
    full_discounted_reverse_rewards = reverse_rewards * discount
    return (np.cumsum(full_discounted_reverse_rewards, axis=1) / discount)[:, :0:-1]


def general_n_step_advantage(
        one_step_rewards: np.ndarray,
        value_estimates: np.ndarray,
        gamma: float,
        lambda_par: float
):
    assert 0.0 < gamma <= 1.0
    assert 0.0 <= lambda_par <= 1.0
    batch_size, timesteps = one_step_rewards.shape
    assert value_estimates.shape == (batch_size, timesteps + 1)
    delta = one_step_rewards + gamma * value_estimates[:, 1:] - value_estimates[:, :-1]

    if lambda_par == 0:
        return delta

    delta_rev = delta[:, ::-1]
    adjustment = (gamma * lambda_par) ** np.arange(timesteps, 0, -1)
    advantage = (np.cumsum(delta_rev * adjustment, axis=1) / adjustment)[:, ::-1]
    return advantage


def combine_first_dimensions(x: np.ndarray):
    first_dim = x.shape[0] * x.shape[1]
    other_dims = x.shape[2:]
    dims = (first_dim,) + other_dims
    return x.reshape(*dims)


def ravel_index_pairs(idx_pairs, n_col):
    if th.cuda.is_available():
        return th.sum(idx_pairs * th.from_numpy(np.array([n_col, 1])[np.newaxis, ...]).float().cuda(), 1)
    else:
        return th.sum(idx_pairs * th.from_numpy(np.array([n_col, 1])[np.newaxis, ...]).float(), 1)


def dict_of_lists_to_list_of_dicts(x: dict):
    dim = {len(v) for v in x.values()}
    assert len(dim) == 1
    dim = dim.pop()
    return [{k: x[k][i] for k in x} for i in range(dim)]


def one_hot_encoding(x, num, shape):
    batch_size = x.size(0)
    x = th.unsqueeze(x, 1)
    res = th.zeros((batch_size,
                    num,
                    shape[0], shape[1]))
    if th.cuda.is_available():
        res = res.cuda()
        x = x.cuda()
    return res.scatter_(1, x, 1)[:, 1:, :, :]


def subsample(a, average_width):
    s = a.shape
    sh = s[1] // average_width, average_width, s[2] // average_width, average_width
    return a.reshape(sh).mean(-1).mean(1)


def calc_pixel_change(state, last_state):
    obs = {item1[0]: np.abs(item1[1] - item2[1])
           for item1, item2 in zip(state.items(), last_state.items())}
    # (1, 13, 32, 32)
    m = {k: np.mean(v, 1) for k, v in obs.items()}
    c = {k: subsample(v, 4) for k, v in m.items()}
    return c
