import os
import sys
from datetime import datetime
from options import FLAGS
from .runner import BaseRunner
from agents.ppo import Trainer, Agent, Saver

model_path = os.path.join(FLAGS.model_path, FLAGS.agent_mode, FLAGS.policy, FLAGS.map_name)
full_log_path = os.path.join(FLAGS.log_path, FLAGS.agent_mode, FLAGS.policy, FLAGS.map_name)
replay_path = os.path.join(FLAGS.replay_dir, FLAGS.agent_mode, FLAGS.policy, FLAGS.map_name)


class Runner(BaseRunner):
    def __init__(self, env_args, agent_args, trainer_args, model_path):
        super(Runner, self).__init__()
        self.agent = Agent(**agent_args)
        self.saver = Saver(self.agent, model_path)
        self.trainer = Trainer(env_args, FLAGS.parallel_size, self.agent, **trainer_args)

    def run(self):
        self.trainer.reset()
        if FLAGS.max_time_step >= 0:
            n_batches = FLAGS.max_time_step * 1000
        else:
            n_batches = -1

        if self.saver.path:
            i, episode = self.saver.restore()
            self.trainer.episode_counter = episode
            print(">>> global step set: ", i)
        else:
            print("Could not find old checkpoint")
            i = 0
        try:
            while True:
                if i % 500 == 0:
                    print(datetime.now())
                    print("# batch %d" % i)
                    sys.stdout.flush()
                if i % FLAGS.save_interval_step == 0:
                    self.saver.save(i, self.trainer.episode_counter)
                    sys.stdout.flush()
                self.trainer.run_batch()
                i += 1
                if 0 <= n_batches <= i:
                    break
        except KeyboardInterrupt:
            pass

        print("Okay. Work is done")
        print(datetime.now())
        print("# batch %d" % i)
        sys.stdout.flush()
        self.saver.save(i, self.trainer.episode_counter)
        sys.stdout.flush()
        self.trainer.close()
