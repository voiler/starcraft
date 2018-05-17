import numpy as np
import sys
from agents.ppo.agent import Agent
from common import ObsProcesser, ActionProcesser, FEATURE_KEYS
from common.utils import general_n_step_advantage, combine_first_dimensions, \
    dict_of_lists_to_list_of_dicts
from options import FLAGS
from environment.sc2_environment import SC2Environments


class Trainer(object):
    def __init__(
            self,
            env_args,
            n_envs,
            agent: Agent,
            n_steps=5,
            gamma=0.99,
            lambda_par=0.95,
            batch_size=128,
            n_epochs=3
    ):
        self.agent = agent
        self.obs_processer = ObsProcesser()
        self.action_processer = ActionProcesser(dim=FLAGS.resolution)
        self.n_steps = n_steps
        self.gamma = gamma
        self.batch_counter = 0
        self.episode_counter = 0
        self.ppo_batch_size = batch_size
        self.lambda_par = lambda_par
        self.n_epochs = n_epochs
        self.n_envs = n_envs
        assert n_steps * n_envs % self.ppo_batch_size == 0
        assert n_steps * n_envs >= self.ppo_batch_size
        self.environments = SC2Environments(n_envs, env_args)

    def reset(self):
        obs = self.environments.reset()
        self.latest_obs = self.obs_processer.process(obs)

    def close(self):
        self.environments.close()

    def _log_score_to_tb(self, score):
        self.agent.writer.add_scalar('score', score, self.episode_counter)

    def _handle_episode_end(self, timestep):
        score = timestep.observation["score_cumulative"][0]
        print("episode %d ended. Score %f" % (self.episode_counter, score))
        self._log_score_to_tb(score)
        self.episode_counter += 1

    def _train_ppo_epoch(self, full_input, obs):
        total_obs = self.n_steps * self.n_envs
        shuffle_idx = np.random.permutation(total_obs)
        batches = dict_of_lists_to_list_of_dicts({
            k: np.split(v[shuffle_idx], total_obs // self.ppo_batch_size)
            for k, v in full_input.items()
        })
        b_obs = dict_of_lists_to_list_of_dicts({
            k: np.split(v[shuffle_idx], total_obs // self.ppo_batch_size)
            for k, v in obs.items()
        })
        for b, _obs in zip(batches, b_obs):
            self.agent.train(b, _obs)

    def run_batch(self):
        mb_actions = []
        mb_obs = []
        mb_values = np.zeros((self.n_envs, self.n_steps + 1), dtype=np.float32)
        mb_rewards = np.zeros((self.n_envs, self.n_steps), dtype=np.float32)

        latest_obs = self.latest_obs

        for n in range(self.n_steps):
            # could calculate value estimate from obs when do training
            # but saving values here will make n step reward calculation a bit easier
            action_ids, spatial_action_2ds, value_estimate = self.agent.step(latest_obs)
            mb_values[:, n] = value_estimate
            mb_obs.append(latest_obs)
            mb_actions.append((action_ids, spatial_action_2ds))

            actions_pp = self.action_processer.process(action_ids, spatial_action_2ds)
            obs_raw = self.environments.step(actions_pp)
            latest_obs = self.obs_processer.process(obs_raw)
            mb_rewards[:, n] = [t.reward for t in obs_raw]

            for t in obs_raw:
                if t.last():
                    self._handle_episode_end(t)

        mb_values[:, -1] = self.agent.get_value(latest_obs)

        n_step_advantage = general_n_step_advantage(
            mb_rewards,
            mb_values,
            self.gamma,
            lambda_par=self.lambda_par
        )

        full_input = {
            # these are transposed because action/obs
            # processers return [time, env, ...] shaped arrays
            FEATURE_KEYS.advantage: n_step_advantage.transpose().astype(np.float32),
            FEATURE_KEYS.value_target: (n_step_advantage + mb_values[:, :-1]).transpose().astype(np.float32)
        }
        full_input.update(self.action_processer.combine_batch(mb_actions))
        obs = self.obs_processer.combine_batch(mb_obs)
        obs = {k: combine_first_dimensions(v) for k, v in obs.items()}
        full_input = {k: combine_first_dimensions(v) for k, v in full_input.items()}
        for epoch in range(self.n_epochs):
            self._train_ppo_epoch(full_input, obs)
        self.agent.update_policy()
        self.latest_obs = latest_obs
        self.batch_counter += 1
        sys.stdout.flush()
