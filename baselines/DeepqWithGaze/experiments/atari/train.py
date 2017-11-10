import argparse
import gym
import numpy as np
import os
import tensorflow as tf, keras as K
import tempfile, time, json
import ipdb

import baselines.common.tf_util as U
from baselines.common.gflag import gflag
import baselines.DeepqWithGaze.misc_utils as MU

from IPython import embed
from baselines import logger
from baselines import DeepqWithGaze
from baselines.DeepqWithGaze.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.common.misc_util import (
    boolean_flag,
    pickle_load,
    pretty_eta,
    relatively_safe_pickle_dump,
    set_global_seeds,
    RunningAvg,
    SimpleMonitor
)
from baselines.common.schedules import LinearSchedule, PiecewiseSchedule
# when updating this to non-deperecated ones, it is important to
# copy over LazyFrames
from baselines.common.atari_wrappers_deprecated import wrap_dqn
from baselines.common.azure_utils import Container
from .model import model, dueling_model


def parse_args():
    parser = argparse.ArgumentParser("DQN experiments for Atari games")
    # Environment
    parser.add_argument("--env", type=str, default="Pong", help="name of the game")
    parser.add_argument("--seed", type=int, default=-1, help="which seed to use. If negative, use system deafult")
    # Core DQN parameters
    parser.add_argument("--replay-buffer-size", type=int, default=int(1e6), help="replay buffer size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate for Adam optimizer")
    parser.add_argument("--num-steps", type=int, default=int(2e8), help="total number of steps to run the environment for")
    parser.add_argument("--batch-size", type=int, default=32, help="number of transitions to optimize at the same time")
    parser.add_argument("--learning-freq", type=int, default=4, help="number of iterations between every optimization step")
    parser.add_argument("--target-update-freq", type=int, default=40000, help="number of iterations between every target network update")
    parser.add_argument("--param-noise-update-freq", type=int, default=50, help="number of iterations between every re-scaling of the parameter noise")
    parser.add_argument("--param-noise-reset-freq", type=int, default=10000, help="maximum number of steps to take per episode before re-perturbing the exploration policy")
    # Bells and whistles
    boolean_flag(parser, "double-q", default=True, help="whether or not to use double q learning")
    boolean_flag(parser, "dueling", default=False, help="whether or not to use dueling model")
    boolean_flag(parser, "prioritized", default=False, help="whether or not to use prioritized replay buffer")
    boolean_flag(parser, "param-noise", default=False, help="whether or not to use parameter space noise for exploration")
    boolean_flag(parser, "layer-norm", default=False, help="whether or not to use layer norm (should be True if param_noise is used)")
    boolean_flag(parser, "gym-monitor", default=False, help="whether or not to use a OpenAI Gym monitor (results in slower training due to video recording)")
    parser.add_argument("--prioritized-alpha", type=float, default=0.6, help="alpha parameter for prioritized replay buffer")
    parser.add_argument("--prioritized-beta0", type=float, default=0.4, help="initial value of beta parameters for prioritized replay")
    parser.add_argument("--prioritized-eps", type=float, default=1e-6, help="eps parameter for prioritized replay buffer")
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default=None, help="directory in which training state and model should be saved.")
    parser.add_argument("--save-azure-container", type=str, default=None,
                        help="It present data will saved/loaded from Azure. Should be in format ACCOUNT_NAME:ACCOUNT_KEY:CONTAINER")
    parser.add_argument("--save-freq", type=int, default=1e6, help="save model once every time this many iterations are completed")
    boolean_flag(parser, "load-on-start", default=True, help="if true and model was previously saved then training will be resumed")
    
    args = parser.parse_args()
    gflag.init_me_as(args.__dict__)
    return args

def make_env(game_name):
    env = gym.make(game_name + "NoFrameskip-v4")
    monitored_env = SimpleMonitor(env)  # puts rewards and number of steps in info, before environment is wrapped
    env = wrap_dqn(monitored_env)  # applies a bunch of modification to simplify the observation space (downsample, make b/w)
    return env, monitored_env


def maybe_save_model(savedir, container, state):
    """This function checkpoints the model and state of the training algorithm."""
    if savedir is None:
        return
    start_time = time.time()
    model_dir = "model-{}".format(state["num_iters"])
    U.save_state(os.path.join(savedir, model_dir, "saved"))
    if container is not None:
        container.put(os.path.join(savedir, model_dir), model_dir)
    relatively_safe_pickle_dump(state, os.path.join(savedir, 'training_state.pkl.zip'), compression=True)
    if container is not None:
        container.put(os.path.join(savedir, 'training_state.pkl.zip'), 'training_state.pkl.zip')
    relatively_safe_pickle_dump(state["monitor_state"], os.path.join(savedir, 'monitor_state.pkl'))
    if container is not None:
        container.put(os.path.join(savedir, 'monitor_state.pkl'), 'monitor_state.pkl')
    logger.log("Saved model in {} seconds\n".format(time.time() - start_time))


def maybe_load_model(savedir, container):
    """Load model if present at the specified path."""
    if savedir is None:
        return

    state_path = os.path.join(os.path.join(savedir, 'training_state.pkl.zip'))
    if container is not None:
        logger.log("Attempting to download model from Azure")
        found_model = container.get(savedir, 'training_state.pkl.zip')
    else:
        found_model = os.path.exists(state_path)
    if found_model:
        state = pickle_load(state_path, compression=True)
        model_dir = "model-{}".format(state["num_iters"])
        if container is not None:
            container.get(savedir, model_dir)
        U.load_state(os.path.join(savedir, model_dir, "saved"))
        logger.log("Loaded models checkpoint at {} iterations".format(state["num_iters"]))
        return state


if __name__ == '__main__':
    debug_embed_last_time = time.time() # TODO this is temporary. delete it and its related code
    debug_embed_freq_sec = 600
    args = parse_args()
    
    # Parse savedir and azure container.
    savedir = args.save_dir
    if savedir is None:
        savedir = os.getenv('OPENAI_LOGDIR', None)
    if args.save_azure_container is not None:
        account_name, account_key, container_name = args.save_azure_container.split(":")
        container = Container(account_name=account_name,
                              account_key=account_key,
                              container_name=container_name,
                              maybe_create=True)
        if savedir is None:
            # Careful! This will not get cleaned up. Docker spoils the developers.
            savedir = tempfile.TemporaryDirectory().name
    else:
        container = None
    # Create and seed the env.
    env, monitored_env = make_env(args.env)
    if args.seed > 0:
        set_global_seeds(args.seed)
        env.unwrapped.seed(args.seed)

    if args.gym_monitor and savedir:
        env = gym.wrappers.Monitor(env, os.path.join(savedir, 'gym_monitor'), force=True)

    if savedir:
        if not os.path.exists(savedir): 
            os.makedirs(savedir)
        with open(os.path.join(savedir, 'args.json'), 'w') as f:
            json.dump(vars(args), f)

    with U.make_session(4) as sess:
        MU.keras_model_serialization_bug_fix()
        # WARNING there's also a 'load_model' down below. Use the same path for both ! (TODO This is hacky; delete it and use some other way)
        gaze_model = K.models.load_model("baselines/DeepqWithGaze/ImgOnly_gazeModels/seaquest-dp0.4-DQN+BNonInput.hdf5")
        # pixel_mean_of_gaze_model_trainset = np.load("baselines/DeepqWithGaze/Img+OF_gazeModels/seaquest.mean.npy")
        K.backend.set_learning_phase(0)
        debug_gaze_in = None
        def tf_op_set_debug_tensor(x):
            global debug_gaze_in
            debug_gaze_in = x
            return x
        # Create training graph and replay buffer
        def model_wrapper(img_in, num_actions, scope, **kwargs):
            # TODO the following is temporary. delete it and its related code
            # (checked) input to gaze model is the same as training; 
            # (checked) GHmap is reasonable; 
            # (checked) the model variables are reused
            # TODO the model is not trained (first search keras doc to know how to freeze weight)
            actual_model = dueling_model if args.dueling else model
            gaze_in = img_in # - pixel_mean_of_gaze_model_trainset unnecessary coz I im using BN-on-Input model
            gaze_in_debug = tf.identity(gaze_in, name='gaze_in_debug') # used for extracting 
                    # the content of img_in in IPython console
            GHmap = gaze_model(gaze_in)
            #debug_tensor = tf.py_func(tf_op_set_debug_tensor, [GHmap*img_in], [tf.float32], stateful=True, name='debug_tensor')
            #with tf.control_dependencies(debug_tensor):
            img_and_gaze_combined = tf.concat([img_in, GHmap*img_in], axis=-1)
            logger.log("img_and_gaze_combined shape: " + str(img_and_gaze_combined.shape)) 
            return actual_model(img_and_gaze_combined, num_actions, scope, layer_norm=args.layer_norm, **kwargs)
        act, train, update_target, debug = DeepqWithGaze.build_train(
            make_obs_ph=lambda name: U.Uint8Input(env.observation_space.shape, name=name),
            q_func=model_wrapper,
            num_actions=env.action_space.n,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr, epsilon=1e-4),
            gamma=0.99,
            grad_norm_clipping=10,
            double_q=args.double_q,
            param_noise=args.param_noise
        )

        approximate_num_iters = args.num_steps / 4
        exploration = PiecewiseSchedule([
            (0, 1.0),
            (approximate_num_iters / 50, 0.1),
            (approximate_num_iters / 5, 0.01)
        ], outside_value=0.01)

        if args.prioritized:
            replay_buffer = PrioritizedReplayBuffer(args.replay_buffer_size, args.prioritized_alpha)
            beta_schedule = LinearSchedule(approximate_num_iters, initial_p=args.prioritized_beta0, final_p=1.0)
        else:
            replay_buffer = ReplayBuffer(args.replay_buffer_size)

        U.initialize()
        gaze_model.load_weights('baselines/DeepqWithGaze/ImgOnly_gazeModels/seaquest-dp0.4-DQN+BNonInput.hdf5')
        update_target()
        num_iters = 0

        # Load the model
        state = maybe_load_model(savedir, container)
        if state is not None:
            num_iters, replay_buffer = state["num_iters"], state["replay_buffer"],
            monitored_env.set_state(state["monitor_state"])

        start_time, start_steps = None, None
        steps_per_iter = RunningAvg(0.999)
        iteration_time_est = RunningAvg(0.999)
        obs = env.reset()
        num_iters_since_reset = 0
        reset = True

        # Main trianing loop
        while True:
            num_iters += 1
            num_iters_since_reset += 1

            # Take action and store transition in the replay buffer.
            kwargs = {}
            if not args.param_noise:
                update_eps = exploration.value(num_iters)
                update_param_noise_threshold = 0.
            else:
                if args.param_noise_reset_freq > 0 and num_iters_since_reset > args.param_noise_reset_freq:
                    # Reset param noise policy since we have exceeded the maximum number of steps without a reset.
                    reset = True

                update_eps = 0.01  # ensures that we cannot get stuck completely
                # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                # for detailed explanation.
                update_param_noise_threshold = -np.log(1. - exploration.value(num_iters) + exploration.value(num_iters) / float(env.action_space.n))
                kwargs['reset'] = reset
                kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                kwargs['update_param_noise_scale'] = (num_iters % args.param_noise_update_freq == 0)

            action = act(np.array(obs)[None], update_eps=update_eps, **kwargs)[0]
            reset = False
            new_obs, rew, done, info = env.step(action)
            replay_buffer.add(obs, action, rew, new_obs, float(done))

            obs = new_obs
            if done:
                num_iters_since_reset = 0
                obs = env.reset()
                reset = True

            if (num_iters > max(5 * args.batch_size, args.replay_buffer_size // 20) and
                    num_iters % args.learning_freq == 0):
                # Sample a bunch of transitions from replay buffer
                if args.prioritized:
                    experience = replay_buffer.sample(args.batch_size, beta=beta_schedule.value(num_iters))
                    (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                else:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(args.batch_size)
                    weights = np.ones_like(rewards)
                # Minimize the error in Bellman's equation and compute TD-error
                td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights)
                # Update the priorities in the replay buffer
                if args.prioritized:
                    new_priorities = np.abs(td_errors) + args.prioritized_eps
                    replay_buffer.update_priorities(batch_idxes, new_priorities)
            # Update target network.
            if num_iters % args.target_update_freq == 0:
                update_target()

            if start_time is not None:
                steps_per_iter.update(info['steps'] - start_steps)
                iteration_time_est.update(time.time() - start_time)
            start_time, start_steps = time.time(), info["steps"]

            # Save the model and training state.
            if num_iters > 0 and (num_iters % args.save_freq == 0 or info["steps"] > args.num_steps):
                maybe_save_model(savedir, container, {
                    'replay_buffer': replay_buffer,
                    'num_iters': num_iters,
                    'monitor_state': monitored_env.get_state(),
                })

            if info["steps"] > args.num_steps:
                break

            if done:
                steps_left = args.num_steps - info["steps"]
                completion = np.round(info["steps"] / args.num_steps, 1)

                logger.record_tabular("% completion", completion)
                logger.record_tabular("steps", info["steps"])
                logger.record_tabular("iters", num_iters)
                logger.record_tabular("episodes", len(info["rewards"]))
                logger.record_tabular("reward (100 epi mean)", np.mean(info["rewards"][-100:]))
                logger.record_tabular("exploration", exploration.value(num_iters))
                if args.prioritized:
                    logger.record_tabular("max priority", replay_buffer._max_priority)
                fps_estimate = (float(steps_per_iter) / (float(iteration_time_est) + 1e-6)
                                if steps_per_iter._value is not None else "calculating...")
                logger.dump_tabular()
                logger.log()
                logger.log("ETA: " + pretty_eta(int(steps_left / fps_estimate)))
                logger.log()
                if time.time() - debug_embed_last_time > debug_embed_freq_sec:
                    print(list(map(np.linalg.norm,gaze_model.get_weights())))
                    #embed()
                    debug_embed_last_time = time.time()
