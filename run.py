import os
import sys

from common.utils import Mode
from policy import get_policy
from options import FLAGS
FLAGS(sys.argv)


def main():
    model_path = os.path.join(FLAGS.model_path, FLAGS.agent_mode, FLAGS.policy, FLAGS.map_name)
    full_log_path = os.path.join(FLAGS.log_path, FLAGS.agent_mode, FLAGS.policy, FLAGS.map_name)
    replay_path = os.path.join(FLAGS.replay_dir, FLAGS.agent_mode, FLAGS.policy, FLAGS.map_name)
    if not os.path.exists(replay_path):
        os.makedirs(replay_path)
    if FLAGS.agent_mode == Mode.A2C:
        from runner.a2c_runner import Runner
        env_args = dict(map_name=FLAGS.map_name,
                        step_mul=FLAGS.step_mul,
                        game_steps_per_episode=0,
                        screen_size_px=(FLAGS.resolution,) * 2,
                        minimap_size_px=(FLAGS.resolution,) * 2,
                        visualize=FLAGS.visualize,
                        save_replay_episodes=FLAGS.save_replay_episodes,
                        replay_dir=replay_path
                        )
        agent_args = dict(spatial_dim=FLAGS.resolution,
                          unit_type_emb_dim=FLAGS.unit_type_emb_dim,
                          loss_value_weight=FLAGS.loss_value_weight,
                          entropy_weight_action_id=FLAGS.entropy_weight_action,
                          entropy_weight_spatial=FLAGS.entropy_weight_spatial,
                          policy=get_policy(FLAGS.policy),
                          summary_path=full_log_path)
        if FLAGS.local_t_max is None:
            n_steps_per_batch = 16
        else:
            n_steps_per_batch = FLAGS.local_t_max
        trainer_args = dict(gamma=FLAGS.gamma, n_steps=n_steps_per_batch)
        runner = Runner(env_args, agent_args, trainer_args, model_path)
        runner.run()
    elif FLAGS.agent_mode == Mode.PPO:
        from runner.ppo_runner import Runner
        env_args = dict(map_name=FLAGS.map_name,
                        step_mul=FLAGS.step_mul,
                        game_steps_per_episode=0,
                        screen_size_px=(FLAGS.resolution,) * 2,
                        minimap_size_px=(FLAGS.resolution,) * 2,
                        visualize=FLAGS.visualize,
                        save_replay_episodes=FLAGS.save_replay_episodes,
                        replay_dir=replay_path
                        )
        agent_args = dict(spatial_dim=FLAGS.resolution,
                          unit_type_emb_dim=FLAGS.unit_type_emb_dim,
                          loss_value_weight=FLAGS.loss_value_weight,
                          entropy_weight_action_id=FLAGS.entropy_weight_action,
                          entropy_weight_spatial=FLAGS.entropy_weight_spatial,
                          policy=get_policy(FLAGS.policy),
                          summary_path=full_log_path)

        if FLAGS.local_t_max is None:
            n_steps_per_batch = 128
        else:
            n_steps_per_batch = FLAGS.local_t_max

        trainer_args = dict(gamma=FLAGS.gamma,
                            n_steps=n_steps_per_batch,
                            lambda_par=FLAGS.ppo_lambda,
                            batch_size=FLAGS.ppo_batch_size or n_steps_per_batch,
                            n_epochs=FLAGS.ppo_epochs
                            )
        runner = Runner(env_args, agent_args, trainer_args, model_path)
        runner.run()
    elif FLAGS.agent_mode == Mode.UNREAL:
        from runner.unreal_runner import Runner
        model_path = os.path.join(FLAGS.model_path, FLAGS.agent_mode, FLAGS.map_name)
        full_log_path = os.path.join(FLAGS.log_path, FLAGS.agent_mode, FLAGS.map_name)
        replay_path = os.path.join(FLAGS.replay_dir, FLAGS.agent_mode, FLAGS.map_name)
        env_args = dict(map_name=FLAGS.map_name,
                        step_mul=FLAGS.step_mul,
                        game_steps_per_episode=0,
                        screen_size_px=(FLAGS.resolution,) * 2,
                        minimap_size_px=(FLAGS.resolution,) * 2,
                        visualize=FLAGS.visualize,
                        save_replay_episodes=FLAGS.save_replay_episodes,
                        replay_dir=replay_path
                        )
        runner = Runner(env_args, full_log_path, model_path)
        runner.run()


if __name__ == '__main__':
    main()
