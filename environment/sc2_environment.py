from torch.multiprocessing import Process, Pipe
from pysc2.env import sc2_env
from pysc2.lib.actions import FUNCTIONS
from functools import partial
from environment import environment

COMMAND_RESET = 0
COMMAND_STEP = 1
COMMAND_TERMINATE = 2


def worker(remote, env_fn_wrapper):
    env = env_fn_wrapper.x()
    while True:
        cmd, action = remote.recv()
        if cmd == COMMAND_STEP:
            timesteps = env.step([action])
            assert len(timesteps) == 1
            remote.send(timesteps[0])
        elif cmd == COMMAND_RESET:
            timesteps = env.reset()
            assert len(timesteps) == 1
            remote.send(timesteps[0])
        elif cmd == COMMAND_TERMINATE:
            remote.close()
            break
        else:
            raise NotImplementedError


class CloudpickleWrapper(object):
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


def make_sc2env(**kwargs):
    env = sc2_env.SC2Env(**kwargs)
    return env


class SC2Environment(environment.Environment):
    def __init__(self, env_args):
        super(SC2Environment, self).__init__()
        env = partial(make_sc2env, **env_args)
        self.conn, child_conn = Pipe()
        self.proc = Process(target=worker, args=(child_conn, CloudpickleWrapper(env)))
        self.proc.start()
        self.reset()

    @staticmethod
    def get_action_size():
        return len(FUNCTIONS)

    def reset(self):
        self.conn.send([COMMAND_RESET, None])
        return [self.conn.recv()]

    def close(self):
        self.conn.send([COMMAND_TERMINATE, None])
        self.conn.close()
        self.proc.join()
        print("SC2 environment closed")

    def step(self, actions):
        self.conn.send([COMMAND_STEP, actions])
        obs = self.conn.recv()
        return [obs], obs.reward, obs.last()


class SC2Environments(environment.Environment):
    def __init__(self, n_envs, env_args):
        super(SC2Environments, self).__init__()
        envs = (partial(make_sc2env, **env_args),) * n_envs
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(n_envs)])
        self.ps = [Process(target=worker, args=(work_remote, CloudpickleWrapper(env)))
                   for (work_remote, env) in zip(self.work_remotes, envs)]
        for p in self.ps:
            p.start()

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send((COMMAND_STEP, action))
        timesteps = [remote.recv() for remote in self.remotes]
        return timesteps

    def reset(self):
        for remote in self.remotes:
            remote.send((COMMAND_RESET, None))
        timesteps = [remote.recv() for remote in self.remotes]
        return timesteps

    def close(self):
        for remote in self.remotes:
            remote.send((COMMAND_TERMINATE, None))
            remote.close()
        for p in self.ps:
            p.join()
