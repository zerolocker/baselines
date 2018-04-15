import argparse, gym, numpy as np, os, tensorflow as tf, keras as K, time, ipdb

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
    RunningAvg,
    maybe_load_model,
    maybe_save_model,
    make_and_wrap_env,
    make_save_dir_and_log_basics
)
from baselines.common.schedules import LinearSchedule, PiecewiseSchedule
from .model import model, dueling_model

import matplotlib.pyplot as plt

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
    parser.add_argument("--save-freq", type=int, default=1e6, help="save model once every time this many iterations are completed")
    boolean_flag(parser, "load-on-start", default=True, help="if true and model was previously saved then training will be resumed")
    boolean_flag(parser, "also-save-training-state", default=False, help="if true also save training state (huge replay buffer) so that training will be resumed")
    
    boolean_flag(parser, "debug-mode", default=False, help="if true ad-hoc debug-related code will be run and training may stop halfway")
    parser.add_argument("--ghmap-multiplier", type=int, default=16, help="multiply ghmap by this number to enlarge the scale")
    args = parser.parse_args()
    gflag.init_me_as(args.__dict__)
    return args

class KerasGazeModelFactory:
    def __init__(self):
        self.models = {}
        # Use compile=False to avoid loading optimizer state, because loading it adds tons of variables to the Graph in Tensorboard, making it ugly
        self.gaze_model_template = K.models.load_model(
            "baselines/DeepqWithGaze/ImgOnly_gazeModels/seaquest-dp0.4-DQN+BNonInput.hdf5", compile=False)

    def get_or_create(self, name):
        """ 
        Note: model weight is not set here because even if we do, U.initialize() will 
        be called below, and it will still override the weight we set here.
        so call initialze_weights_for_all_created_models() after U.initialize()
        """
        if name in self.models:
            logger.log("model named %s is reused" % name)
        else:
            logger.log("model named %s is created" % name)
            self.models[name] = K.models.Model.from_config(self.gaze_model_template.get_config())
        return self.models[name]

    def initialze_weights_for_all_created_models(self):
        for model in self.models.values():
            model.set_weights(self.gaze_model_template.get_weights())


if __name__ == '__main__':
    args = parse_args()
    make_save_dir_and_log_basics()
    MU.keras_model_serialization_bug_fix()

    env, monitored_env = make_and_wrap_env(args.env, args.seed)
    gaze_model_factory = KerasGazeModelFactory()

    with U.make_session(4) as sess:
        # pixel_mean_of_gaze_model_trainset = np.load("baselines/DeepqWithGaze/Img+OF_gazeModels/seaquest.mean.npy")
        K.backend.set_session(sess)
        if args.debug_mode:
            debug_gaze_in = None 
            #debug_saved_gaze_in = []
            def tf_op_set_debug_tensor(x):
                global debug_gaze_in
                debug_gaze_in = x
                #debug_saved_gaze_in.append(np.copy(x))
                return x
        # Create training graph and replay buffer
        def model_wrapper(img_in, num_actions, scope, **kwargs):
            logger.log("model_wrapper called: ", str(scope), str(kwargs))
            actual_model = dueling_model if args.dueling else model
            gaze_in = img_in # - pixel_mean_of_gaze_model_trainset unnecessary coz I im using BN-on-Input model
            with tf.name_scope(scope+'/ghmap'): # name_scope makes it look nicer on TensorBoard
                GHmap = gaze_model_factory.get_or_create(name=scope)(gaze_in) * gflag.ghmap_multiplier
            if args.debug_mode:
                debug_tensor = tf.py_func(tf_op_set_debug_tensor, [tf.concat([img_in, GHmap], axis=-1)], [tf.float32], stateful=True, name='debug_tensor')
                with tf.control_dependencies(debug_tensor):
                    img_and_gaze_combined = tf.concat([img_in, GHmap*img_in], axis=-1)
            else:
                img_and_gaze_combined = tf.concat([img_in, GHmap*img_in], axis=-1)
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
        gaze_model_factory.initialze_weights_for_all_created_models()
        update_target()
        num_iters = 0

        if args.load_on_start: # Load the model
            state = maybe_load_model(gflag.save_dir)
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
        if args.debug_mode:
            fig, axarr = plt.subplots(2,3) # TODO debug only
            debug_embed_last_time = time.time() # TODO this is temporary. delete it and its related code
            debug_embed_freq_sec = 10
        while True:
            num_iters += 1
            num_iters_since_reset += 1
            if args.debug_mode and debug_gaze_in is not None:
                for i in range(4):
                    axarr[int(i/2), i%2].imshow(debug_gaze_in[0,:,:,i]) 
                axarr[1,2].imshow(debug_gaze_in[0,:,:,4]) 
                plt.pause(0.1)

            if args.debug_mode and time.time() - debug_embed_last_time > debug_embed_freq_sec:
                embed()
                debug_embed_last_time = time.time()

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
                maybe_save_model(gflag.save_dir, {
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
