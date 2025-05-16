# renew/envs/__init__.py

from copy import deepcopy
import warnings
import gymnasium as gym

from .wrappers.multitask import MultitaskWrapper
from .wrappers.tensor     import TensorWrapper

def missing_dependencies(task):
    raise ValueError(f'Missing dependencies for task {task}; install dependencies to use this environment.')

try:
    from .dmcontrol    import make_env as make_dm_control_env
except:
    make_dm_control_env = missing_dependencies

try:
    from .maniskill    import make_env as make_maniskill_env
except:
    make_maniskill_env = missing_dependencies

try:
    from .metaworld    import make_env as make_metaworld_env
except:
    make_metaworld_env = missing_dependencies

try:
    from .myosuite     import make_env as make_myosuite_env
except:
    make_myosuite_env = missing_dependencies

warnings.filterwarnings('ignore', category=DeprecationWarning)

def make_multitask_env(cfg):
    print('Creating multi-task environment with tasks:', cfg.tasks)
    envs = []
    for task in cfg.tasks:
        _cfg = deepcopy(cfg)
        _cfg.task = task
        _cfg.multitask = False
        env = make_env(_cfg)
        if env is None:
            raise ValueError('Unknown task:', task)
        envs.append(env)
    env = MultitaskWrapper(cfg, envs)
    cfg.obs_shapes      = env._obs_dims
    cfg.action_dims    = env._action_dims
    cfg.episode_lengths= env._episode_lengths
    return env

def make_env(cfg):
    gym.logger.set_level(40)
    if cfg.multitask:
        return make_multitask_env(cfg)
    env = None
    for fn in (make_dm_control_env, make_maniskill_env, make_metaworld_env, make_myosuite_env):
        try:
            env = fn(cfg)
            break
        except ValueError:
            continue
    if env is None:
        raise ValueError(f'Failed to make environment "{cfg.task}": please verify that dependencies are installed and that the task exists.')
    from .wrappers.tensor import TensorWrapper
    env = TensorWrapper(env)
    # fill back into cfg
    try:
        cfg.obs_shape = {k: v.shape for k,v in env.observation_space.spaces.items()}
    except:
        cfg.obs_shape = {cfg.get('obs','state'): env.observation_space.shape}
    cfg.action_dim    = env.action_space.shape[0]
    cfg.episode_length= env.max_episode_steps
    cfg.seed_steps    = max(1000, 5*cfg.episode_length)
    return env
