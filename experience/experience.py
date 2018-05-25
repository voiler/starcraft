# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from collections import deque
from pysc2.lib.actions import FUNCTIONS
from options import FLAGS

SIZE = len(FUNCTIONS)


class ExperienceFrame(object):
    def __init__(self, state, reward, action, terminal, pixel_change, last_action, last_reward):
        self.state = state
        self.action = action  # (Taken action with the 'state')
        self.reward = np.clip(reward, -1, 1)  # Reward with the 'state'. (Clipped)
        self.terminal = terminal  # (Whether terminated when 'state' was inputted)
        self.pixel_change = pixel_change
        self.last_action = last_action  # (After this last action was taken, agent move to the 'state')
        self.last_reward = np.clip(last_reward, -1,
                                   1)  # (After this last reward was received, agent move to the 'state') (Clipped)

    def get_last_action_reward(self):
        return ExperienceFrame.concat_action_and_reward(self.last_action, self.last_reward)

    def get_action_reward(self):
        return ExperienceFrame.concat_action_and_reward(self.action, self.reward)

    @staticmethod
    def concat_action_and_reward(action, reward):
        action, sp_action = action
        action_reward = np.ones([SIZE + 1], dtype=np.float32) * 1e-12
        action_reward[action] = 1.0
        action_reward[-1] = float(reward)
        action_reward = np.concatenate(
            [action_reward, np.ones(FLAGS.resolution * FLAGS.resolution - SIZE - 1,
                                    dtype=np.float32) * 1e-12
             ],
            0)
        action_reward = action_reward.reshape((FLAGS.resolution, FLAGS.resolution))
        rect_sp_action = np.ones((FLAGS.resolution, FLAGS.resolution), dtype=np.float32) * 1e-12
        rect_sp_action[sp_action[0]][sp_action[1]] = 1
        action_reward = np.expand_dims(action_reward, 0)
        rect_sp_action = np.expand_dims(rect_sp_action, 0)
        action_reward = np.concatenate([action_reward, rect_sp_action], 0)
        return action_reward


class Experience(object):
    def __init__(self, history_size):
        self._history_size = history_size
        self._frames = deque(maxlen=history_size)
        # frame indices for zero rewards
        self._zero_reward_indices = deque()
        # frame indices for non zero rewards
        self._non_zero_reward_indices = deque()
        self._top_frame_index = 0

    def add_frame(self, frame):
        if frame.terminal and len(self._frames) > 0 and self._frames[-1].terminal:
            # Discard if terminal frame continues
            print("Terminal frames continued.")
            return

        frame_index = self._top_frame_index + len(self._frames)
        was_full = self.is_full()

        # append frame
        self._frames.append(frame)

        # append index
        if frame_index >= 3:
            if frame.reward == 0:
                self._zero_reward_indices.append(frame_index)
            else:
                self._non_zero_reward_indices.append(frame_index)

        if was_full:
            self._top_frame_index += 1

            cut_frame_index = self._top_frame_index + 3
            # Cut frame if its index is lower than cut_frame_index.
            if len(self._zero_reward_indices) > 0 and \
                    self._zero_reward_indices[0] < cut_frame_index:
                self._zero_reward_indices.popleft()

            if len(self._non_zero_reward_indices) > 0 and \
                    self._non_zero_reward_indices[0] < cut_frame_index:
                self._non_zero_reward_indices.popleft()

    def is_full(self):
        return len(self._frames) >= self._history_size

    def sample_sequence(self, sequence_size):
        # -1 for the case if start pos is the terminated frame.
        # (Then +1 not to start from terminated frame.)
        start_pos = np.random.randint(0, self._history_size - sequence_size - 1)

        if self._frames[start_pos].terminal:
            start_pos += 1
            # Assuming that there are no successive terminal frames.

        sampled_frames = []

        for i in range(sequence_size):
            frame = self._frames[start_pos + i]
            sampled_frames.append(frame)
            if frame.terminal:
                break

        return sampled_frames

    def sample_rp_sequence(self):
        if np.random.randint(2) == 0:
            from_zero = True
        else:
            from_zero = False

        if len(self._zero_reward_indices) == 0:
            # zero rewards container was empty
            from_zero = False
        elif len(self._non_zero_reward_indices) == 0:
            # non zero rewards container was empty
            from_zero = True

        if from_zero:
            index = np.random.randint(len(self._zero_reward_indices))
            end_frame_index = self._zero_reward_indices[index]
        else:
            index = np.random.randint(len(self._non_zero_reward_indices))
            end_frame_index = self._non_zero_reward_indices[index]

        start_frame_index = end_frame_index - 3
        raw_start_frame_index = start_frame_index - self._top_frame_index

        sampled_frames = []

        for i in range(4):
            frame = self._frames[raw_start_frame_index + i]
            sampled_frames.append(frame)

        return sampled_frames
