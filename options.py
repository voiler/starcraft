from common.utils import Mode
from absl import flags


class Policy:
    FullyConv = 'fullyconv'
    ConvLSTM = 'convlstm'
    SepConv = 'sepconv'


FLAGS = flags.FLAGS
flags.DEFINE_bool("visualize", False, "Whether to render with pygame.")
flags.DEFINE_integer("resolution", 32, "Resolution for screen and minimap feature layers.")
flags.DEFINE_string("map_name", "DefeatRoaches", "Name of a map to use.")
# DefeatRoaches DefeatZerglingsAndBanelings FindAndDefeatZerglings
flags.DEFINE_integer("save_replay_episodes", 4000, "episodes to save game replay ")
flags.DEFINE_string("replay_dir", "./replays", "the dir to save game replays")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")
flags.DEFINE_integer("parallel_size", 1, "Number of environments to run in parallel")
flags.DEFINE_string("model_path", "./models", "Path for agent checkpoints")
flags.DEFINE_string("log_path", "./log", "Path for tensorboard summaries")
flags.DEFINE_integer("max_time_step", 850,
                     "Number of training batches to run in thousands")
flags.DEFINE_float("max_gradient_norm", 500.0, "good value might depend on the environment")
flags.DEFINE_float("gamma", 0.95, "Reward-discount for the agent")
flags.DEFINE_enum("if_output_exists", "overwrite", ["fail", "overwrite", "continue"],
                  "What to do if summary and model output exists, only for training, is ignored if notraining")
flags.DEFINE_float("grad_norm_clip", 500.0, "good value might depend on the environment")
flags.DEFINE_float("loss_value_weight", 1.0, "good value might depend on the environment")
flags.DEFINE_float("entropy_weight_spatial", 1e-6,
                   "entropy of spatial action distribution loss weight")
flags.DEFINE_float("entropy_weight_action", 1e-6, "entropy of action-id distribution loss weight")
flags.DEFINE_integer("unit_type_emb_dim", 5, "unit type emb dim")
flags.DEFINE_integer("local_t_max", None,
                     "Number of steps per batch, if None use 8 for a2c and 128 for ppo")
# ppo
flags.DEFINE_float("ppo_lambda", 0.95, "lambda parameter for ppo")
flags.DEFINE_integer("ppo_batch_size", None, "batch size for ppo, if None use n_steps_per_batch")
flags.DEFINE_integer("ppo_epochs", 3, "epochs per update")
# unreal
flags.DEFINE_float("initial_alpha_low", 1e-4, "log_uniform low limit for learning rate")
flags.DEFINE_float("initial_alpha_high", 5e-3, "log_uniform high limit for learning rate")
flags.DEFINE_float("initial_alpha_log_rate", 0.5, "log_uniform interpolate rate for learning rate")
flags.DEFINE_boolean("use_pixel_change", True, "whether to use pixel change")
flags.DEFINE_boolean("use_value_replay", True, "whether to use value function replay")
flags.DEFINE_boolean("use_reward_prediction", True, "whether to use reward prediction")
flags.DEFINE_float("entropy_beta", 0.001, "entropy regularization constant")
flags.DEFINE_float("pixel_change_lambda", 0.01, "pixel change lambda")
flags.DEFINE_float("gamma_pc", 0.9, "discount factor for pixel control")
flags.DEFINE_integer("experience_history_size", 2000, "experience replay buffer size")
flags.DEFINE_integer("save_interval_step", 100 * 1000, "saving interval steps")
flags.DEFINE_float("alpha", 0.99, "decay parameter for rmsprop")
flags.DEFINE_float("epsilon", 0.1, "epsilon parameter for rmsprop")

flags.DEFINE_enum("agent_mode", Mode.UNREAL, [Mode.A2C, Mode.PPO, Mode.UNREAL], " use A2C or PPO")
flags.DEFINE_enum('policy', Policy.SepConv, [Policy.FullyConv, Policy.ConvLSTM, Policy.SepConv],
                  "policy")
