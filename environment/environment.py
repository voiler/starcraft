# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Environment(object):
    action_size = -1

    @staticmethod
    def create_environment(env_args):
        from . import sc2_environment
        return sc2_environment.SC2Environment(env_args)

    @staticmethod
    def get_action_size():
        if Environment.action_size >= 0:
            return Environment.action_size
        from . import sc2_environment
        Environment.action_size = sc2_environment.SC2Environment.get_action_size()
        return Environment.action_size

    def __init__(self):
        pass

    def step(self, **kwargs):
        pass

    def reset(self):
        pass

    def close(self):
        pass

