import numpy as np
import sys
from agents.a2c.agent import Agent
from environment.sc2_environment import SC2Environments
from common import ObsProcesser, ActionProcesser, FEATURE_KEYS
from common.utils import general_n_step_advantage, combine_first_dimensions
from options import FLAGS


class Trainer(object):
    def __init__(self,
                 env_args,
                 n_envs,
                 agent: Agent,
                 n_steps=5,
                 gamma=0.99
                 ):
        self.agent = agent
        self.obs_processer = ObsProcesser()
        self.action_processer = ActionProcesser(dim=FLAGS.resolution)
        self.n_steps = n_steps
        self.gamma = gamma
        self.batch_counter = 0
        self.episode_counter = 0
        self.n_envs = n_envs
        self.environments = SC2Environments(n_envs, env_args)

    def close(self):
        self.environments.close()

    def reset(self):
        obs = self.environments.reset()
        self.latest_obs = self.obs_processer.process(obs)

    def _log_score_to_tb(self, score):
        self.agent.writer.add_scalar('score', score, self.episode_counter)

    def _handle_episode_end(self, timestep):
        score = timestep.observation["score_cumulative"][0]
        print("episode %d ended. Score %f" % (self.episode_counter, score))
        self._log_score_to_tb(score)
        self.episode_counter += 1

    def run_batch(self):
        mb_actions = []
        mb_obs = []
        mb_values = np.zeros((self.n_envs, self.n_steps + 1), dtype=np.float32)
        mb_rewards = np.zeros((self.n_envs, self.n_steps), dtype=np.float32)
        latest_obs = self.latest_obs
        for n in range(self.n_steps):
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
            lambda_par=1.0
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
        self.agent.train(full_input, obs)
        self.latest_obs = latest_obs
        self.batch_counter += 1
        sys.stdout.flush()
