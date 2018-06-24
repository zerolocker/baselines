import gym
import numpy as np
import os, subprocess, shutil
import pickle
import random
import tempfile
import time
import zipfile
from baselines.common import tf_util as U
from baselines.common.gflag import gflag
from baselines import logger

def zipsame(*seqs):
    L = len(seqs[0])
    assert all(len(seq) == L for seq in seqs[1:])
    return zip(*seqs)


def unpack(seq, sizes):
    """
    Unpack 'seq' into a sequence of lists, with lengths specified by 'sizes'.
    None = just one bare element, not a list

    Example:
    unpack([1,2,3,4,5,6], [3,None,2]) -> ([1,2,3], 4, [5,6])
    """
    seq = list(seq)
    it = iter(seq)
    assert sum(1 if s is None else s for s in sizes) == len(seq), "Trying to unpack %s into %s" % (seq, sizes)
    for size in sizes:
        if size is None:
            yield it.__next__()
        else:
            li = []
            for _ in range(size):
                li.append(it.__next__())
            yield li


class EzPickle(object):
    """Objects that are pickled and unpickled via their constructor
    arguments.

    Example usage:

        class Dog(Animal, EzPickle):
            def __init__(self, furcolor, tailkind="bushy"):
                Animal.__init__()
                EzPickle.__init__(furcolor, tailkind)
                ...

    When this object is unpickled, a new Dog will be constructed by passing the provided
    furcolor and tailkind into the constructor. However, philosophers are still not sure
    whether it is still the same dog.

    This is generally needed only for environments which wrap C/C++ code, such as MuJoCo
    and Atari.
    """

    def __init__(self, *args, **kwargs):
        self._ezpickle_args = args
        self._ezpickle_kwargs = kwargs

    def __getstate__(self):
        return {"_ezpickle_args": self._ezpickle_args, "_ezpickle_kwargs": self._ezpickle_kwargs}

    def __setstate__(self, d):
        out = type(self)(*d["_ezpickle_args"], **d["_ezpickle_kwargs"])
        self.__dict__.update(out.__dict__)


def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)


def pretty_eta(seconds_left):
    """Print the number of seconds in human readable format.

    Examples:
    2 days
    2 hours and 37 minutes
    less than a minute

    Paramters
    ---------
    seconds_left: int
        Number of seconds to be converted to the ETA
    Returns
    -------
    eta: str
        String representing the pretty ETA.
    """
    minutes_left = seconds_left // 60
    seconds_left %= 60
    hours_left = minutes_left // 60
    minutes_left %= 60
    days_left = hours_left // 24
    hours_left %= 24

    def helper(cnt, name):
        return "{} {}{}".format(str(cnt), name, ('s' if cnt > 1 else ''))

    if days_left > 0:
        msg = helper(days_left, 'day')
        if hours_left > 0:
            msg += ' and ' + helper(hours_left, 'hour')
        return msg
    if hours_left > 0:
        msg = helper(hours_left, 'hour')
        if minutes_left > 0:
            msg += ' and ' + helper(minutes_left, 'minute')
        return msg
    if minutes_left > 0:
        return helper(minutes_left, 'minute')
    return 'less than a minute'


class RunningAvg(object):
    def __init__(self, gamma, init_value=None):
        """Keep a running estimate of a quantity. This is a bit like mean
        but more sensitive to recent changes.

        Parameters
        ----------
        gamma: float
            Must be between 0 and 1, where 0 is the most sensitive to recent
            changes.
        init_value: float or None
            Initial value of the estimate. If None, it will be set on the first update.
        """
        self._value = init_value
        self._gamma = gamma

    def update(self, new_val):
        """Update the estimate.

        Parameters
        ----------
        new_val: float
            new observated value of estimated quantity.
        """
        if self._value is None:
            self._value = new_val
        else:
            self._value = self._gamma * self._value + (1.0 - self._gamma) * new_val

    def __float__(self):
        """Get the current estimate"""
        return self._value


class SimpleMonitor(gym.Wrapper):
    def __init__(self, env):
        """Adds two qunatities to info returned by every step:

            num_steps: int
                Number of steps takes so far
            rewards: [float]
                All the cumulative rewards for the episodes completed so far.
        """
        super().__init__(env)
        # current episode state
        self._current_reward = None
        self._num_steps = None
        # temporary monitor state that we do not save
        self._time_offset = None
        self._total_steps = None
        # monitor state
        self._episode_rewards = []
        self._episode_lengths = []
        self._episode_end_times = []

    def _reset(self):
        obs = self.env.reset()
        # recompute temporary state if needed
        if self._time_offset is None:
            self._time_offset = time.time()
            if len(self._episode_end_times) > 0:
                self._time_offset -= self._episode_end_times[-1]
        if self._total_steps is None:
            self._total_steps = sum(self._episode_lengths)
        # update monitor state
        if self._current_reward is not None:
            self._episode_rewards.append(self._current_reward)
            self._episode_lengths.append(self._num_steps)
            self._episode_end_times.append(time.time() - self._time_offset)
        # reset episode state
        self._current_reward = 0
        self._num_steps = 0

        return obs

    def _step(self, action):
        obs, rew, done, info = self.env.step(action)
        self._current_reward += rew
        self._num_steps += 1
        self._total_steps += 1
        info['steps'] = self._total_steps
        info['rewards'] = self._episode_rewards
        return (obs, rew, done, info)

    def get_state(self):
        return {
            'env_id': self.env.unwrapped.spec.id,
            'episode_data': {
                'episode_rewards': self._episode_rewards,
                'episode_lengths': self._episode_lengths,
                'episode_end_times': self._episode_end_times,
                'initial_reset_time': 0,
            }
        }

    def set_state(self, state):
        assert state['env_id'] == self.env.unwrapped.spec.id
        ed = state['episode_data']
        self._episode_rewards = ed['episode_rewards']
        self._episode_lengths = ed['episode_lengths']
        self._episode_end_times = ed['episode_end_times']


def boolean_flag(parser, name, default=False, help=None):
    """Add a boolean flag to argparse parser.

    Parameters
    ----------
    parser: argparse.Parser
        parser to add the flag to
    name: str
        --<name> will enable the flag, while --no-<name> will disable it
    default: bool or None
        default value of the flag
    help: str
        help string for the flag
    """
    dest = name.replace('-', '_')
    parser.add_argument("--" + name, action="store_true", default=default, dest=dest, help=help)
    parser.add_argument("--no-" + name, action="store_false", dest=dest)


def get_wrapper_by_name(env, classname):
    """Given an a gym environment possibly wrapped multiple times, returns a wrapper
    of class named classname or raises ValueError if no such wrapper was applied

    Parameters
    ----------
    env: gym.Env of gym.Wrapper
        gym environment
    classname: str
        name of the wrapper

    Returns
    -------
    wrapper: gym.Wrapper
        wrapper named classname
    """
    currentenv = env
    while True:
        if classname == currentenv.class_name():
            return currentenv
        elif isinstance(currentenv, gym.Wrapper):
            currentenv = currentenv.env
        else:
            raise ValueError("Couldn't find wrapper named %s" % classname)


def relatively_safe_pickle_dump(obj, path, compression=False):
    """This is just like regular pickle dump, except from the fact that failure cases are
    different:

        - It's never possible that we end up with a pickle in corrupted state.
        - If a there was a different file at the path, that file will remain unchanged in the
          even of failure (provided that filesystem rename is atomic).
        - it is sometimes possible that we end up with useless temp file which needs to be
          deleted manually (it will be removed automatically on the next function call)

    The indended use case is periodic checkpoints of experiment state, such that we never
    corrupt previous checkpoints if the current one fails.

    Parameters
    ----------
    obj: object
        object to pickle
    path: str
        path to the output file
    compression: bool
        if true pickle will be compressed
    """
    temp_storage = path + ".relatively_safe"
    if compression:
        # Using gzip here would be simpler, but the size is limited to 2GB
        with tempfile.NamedTemporaryFile() as uncompressed_file:
            pickle.dump(obj, uncompressed_file)
            with zipfile.ZipFile(temp_storage, "w", compression=zipfile.ZIP_DEFLATED) as myzip:
                myzip.write(uncompressed_file.name, "data")
    else:
        with open(temp_storage, "wb") as f:
            pickle.dump(obj, f)
    os.rename(temp_storage, path)


def pickle_load(path, compression=False):
    """Unpickle a possible compressed pickle.

    Parameters
    ----------
    path: str
        path to the output file
    compression: bool
        if true assumes that pickle was compressed when created and attempts decompression.

    Returns
    -------
    obj: object
        the unpickled object
    """

    if compression:
        with zipfile.ZipFile(path, "r", compression=zipfile.ZIP_DEFLATED) as myzip:
            with myzip.open("data") as f:
                return pickle.load(f)
    else:
        with open(path, "rb") as f:
            return pickle.load(f)


def maybe_save_model(savedir, state):
    """deepq/?/?/train.py does not use this, but a same-name func in train.py"""
    if savedir is None or not gflag.resumable:
        return
    logger.log("Saving model. First create tmp file and then rename them to make this operation approximately transactional.")
    start_time = time.time()
    import subprocess

    fname1 = os.path.join(savedir, 'training_state.pkl.zip')
    fname2 = os.path.join(savedir, 'monitor_state.pkl')
    subprocess.call("rm -rf " + os.path.join(savedir, "tmp_model_dir"), shell=True)
    U.save_state(os.path.join(savedir, "tmp_model_dir", "saved"))
    relatively_safe_pickle_dump(state, fname1 + '.tmp', compression=True)
    relatively_safe_pickle_dump(state["monitor_state"], fname2 + '.tmp')

    subprocess.call("mv {0}.tmp {0}".format(fname1), shell=True)
    subprocess.call("mv {0}.tmp {0}".format(fname2), shell=True)
    subprocess.call("rm -rf " + os.path.join(savedir, "model-*"), shell=True)  # remove previously-saved models to save space
    subprocess.call("mv {0} {1}".format(
        os.path.join(savedir, "tmp_model_dir"),
        os.path.join(savedir, "model-{}".format(state["num_iters"]))), shell=True)
    logger.log("Saved model in {} seconds\n".format(time.time() - start_time))


def maybe_load_model(savedir):
    """Load model if present at the specified path."""
    if savedir is None:
        return
    state_path = os.path.join(os.path.join(savedir, 'training_state.pkl.zip'))
    found_model = os.path.exists(state_path)
    if found_model:
        state = pickle_load(state_path, compression=True)
        model_dir = "model-{}".format(state["num_iters"])
        U.load_state(os.path.join(savedir, model_dir, "saved"))
        logger.log("Loaded models checkpoint at {} iterations".format(state["num_iters"]))
        return state


def make_and_wrap_env(game_name, seed):
    # when updating this to non-deperecated ones, it is important to
    # copy over LazyFrames
    from baselines.common.atari_wrappers_deprecated import wrap_dqn
    env = gym.make(game_name + "NoFrameskip-v4")
    monitored_env = SimpleMonitor(env)  # puts rewards and number of steps in info, before environment is wrapped
    env = wrap_dqn(monitored_env)  # applies a bunch of modification to simplify the observation space (downsample, make b/w)
    if seed > 0:
        set_global_seeds(seed)
        env.unwrapped.seed(seed)
    if gflag.gym_monitor and gflag.save_dir:
        env = gym.wrappers.Monitor(env, os.path.join(gflag.save_dir, 'gym_monitor'), force=True)
    return env, monitored_env

def make_save_dir_and_log_basics(argdict):
    if not gflag.save_dir:
        assert not gflag.resumable, "You cannot set --resumable without setting --save-dir."
    else:
        assert gflag.resumable or (not os.path.exists(gflag.save_dir)), \
          "--save_dir '%s' already exists and resumable is False. " % (gflag.save_dir) + \
          "This might be because condor killed and rescheduled the original task."  + \
          "To prevent log.txt being overwritten/appended by a possibly different model's log, " + \
          "the program will terminate now."
        os.makedirs(gflag.save_dir, exist_ok=True)
        logger.configure(gflag.save_dir, format_strs=['log', 'stdout'])
        logger.logkvs(argdict)
        logger.dumpkvs()

        # copy related py files to save_dir to generate a snapshot of code being run
        snapshot_dir = gflag.save_dir + "/all_py_files_snapshot/"
        py_files = subprocess.check_output("find baselines | grep '\\.py$'", shell=True).decode('utf-8').split()
        py_files += subprocess.check_output("ls *.py", shell=True).decode('utf-8').split()
        for py_file in py_files:
            os.makedirs(snapshot_dir + os.path.dirname(py_file), exist_ok=True)
            shutil.copyfile(py_file, snapshot_dir + py_file)

def py3_import_model_by_filename(path):
    logger.log("importing: " + path)
    # import a file by path
    import importlib.util
    spec = importlib.util.spec_from_file_location("module.name", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.model

