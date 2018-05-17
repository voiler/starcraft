import sys
import os
from absl import flags
from pysc2.env import sc2_env
from pysc2.lib import actions

from agents.dqn import trainer

_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_ALL = [0]
_NOT_QUEUED = [0]

step_mul = 1
steps = 2000

FLAGS = flags.FLAGS
PROJ_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    FLAGS(sys.argv)
    with sc2_env.SC2Env(
            map_name="DefeatZerglingsAndBanelings",
            step_mul=step_mul,
            visualize=False,
            game_steps_per_episode=steps * step_mul) as env:
        trainer.learn(
            env,
            num_actions=3,
            lr=1e-4,
            max_timesteps=1000000,
            buffer_size=100000,
            exploration_fraction=0.5,
            exploration_final_eps=0.01,
            train_freq=1,
            learning_starts=10000,
            target_network_update_freq=1000,
            gamma=0.99,
            prioritized_replay=True,
            num_cpu=2
        )


if __name__ == '__main__':
    main()
