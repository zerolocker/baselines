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
    make_save_dir_and_log_basics,
    py3_import_model_by_filename
)
from baselines.common.schedules import LinearSchedule, PiecewiseSchedule
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser("DQN experiments for Atari games")
    # Environment
    parser.add_argument("--env", type=str, default="Pong", help="name of the game")
    parser.add_argument("--seed", type=int, default=int(time.time()*1000 % 65536), help="which seed to use. If negative, use system deafult")
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
    boolean_flag(parser, "resumable", default=False, help="if true model was previously saved then training will be resumed; \
      also save training state (including huge replay buffer) so that training can be resumed")
    
    parser.add_argument("--qfunc-model-filename", default=None, required=True, help="QFunc model arch file (e.g. experiments/atari/model.py). Available names are constantly changing as we experiment new models")
    parser.add_argument("--init-freeze-iter", type=int, default=0, required=False, help="num_iter before which the weights of the pre-initialized layers is frozen")
    boolean_flag(parser, "debug-mode", default=False, help="if true ad-hoc debug-related code will be run and training may stop halfway")
    boolean_flag(parser, "train-gaze", default=True, help="if false, gaze model weight will not be trained")
    args = parser.parse_args()
    gflag.add_read_only_from_dict(args.__dict__)
    return args

if __name__ == '__main__':
    args = parse_args()
    make_save_dir_and_log_basics(args.__dict__)
    MU.keras_model_serialization_bug_fix()
    model = py3_import_model_by_filename(os.path.dirname(__file__) + "/" + args.qfunc_model_filename)

    env, monitored_env = make_and_wrap_env(args.env, args.seed)
    with U.make_session(4) as sess:
        # commented this line out coz I im using BN-on-Input model
        # pixel_mean_of_gaze_model_trainset = np.load("baselines/DeepqWithGaze/Img+OF_gazeModels/seaquest.mean.npy")

        def model_wrapper(img_in, num_actions, scope, **kwargs):
            return model(img_in, num_actions, scope, layer_norm=args.layer_norm,
                         dueling=args.dueling, **kwargs)

        act, train, update_target, debug, tensorboard_summary = DeepqWithGaze.build_train(
            make_obs_ph=lambda name: U.Uint8Input(env.observation_space.shape, name=name),
            q_func=model_wrapper,
            num_actions=env.action_space.n,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr, epsilon=1e-4),
            gamma=0.99,
            grad_norm_clipping=10,
            double_q=args.double_q,
            param_noise=args.param_noise,
            train_gaze=args.train_gaze
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
        gflag.gaze_models.initialze_weights_for_all_created_models()
        gflag.qfunc_models.initialze_weights_for_all_created_models()
        update_target()
        num_iters = 0

        if args.resumable:
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

        # For tensorboard
        if args.save_dir != None:
            train_writer=tf.summary.FileWriter(gflag.save_dir, K.backend.get_session().graph)
            write_summary_last_time = time.time()
            write_summary_freq_sec = 3600

        # Main trianing loop
        if args.debug_mode:
            fig, axarr = plt.subplots(2,3) # TODO debug only
            debug_embed_last_time = time.time() # TODO this is temporary. delete it and its related code
            debug_embed_freq_sec = 10
        while True:
            num_iters += 1
            num_iters_since_reset += 1
            if args.debug_mode and gflag.exists('debug_gaze_in'):
                for i in range(4):
                    axarr[int(i/2), i%2].imshow(gflag.debug_gaze_in[0,:,:,i])
                axarr[1,2].imshow(gflag.debug_gaze_in[0,:,:,4])
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

            # Write summary to tensorboard (e.g. current game frame and gaze images)
            if args.save_dir != None and time.time()-write_summary_last_time > write_summary_freq_sec:
                write_summary_last_time = time.time()
                summary = tensorboard_summary(np.array(obs)[None])
                train_writer.add_summary(summary, num_iters)
                logger.log("%s: writing to tensorboard takes %.2fs" % \
                    (time.strftime("%Y-%m-%d %H:%M:%S %Z"), time.time()-write_summary_last_time))

            action = act(np.array(obs)[None], update_eps=update_eps, **kwargs)[0]
            reset = False
            new_obs, rew, done, info = env.step(action)
            replay_buffer.add(obs, action, rew, new_obs, float(done))

            obs = new_obs
            if done:
                num_iters_since_reset = 0
                obs = env.reset()
                reset = True

            if num_iters % 10000 == 0:
                logger.log("Norm of some weights: ( to see if the Keras model is actually training )")
                if hasattr(gflag.gaze_models.get('q_func'), 'interesting_layers'): # some model may not have this attribute
                    w_gaze = {l.name : np.linalg.norm(l.get_weights()[0]) for l in gflag.gaze_models.get('q_func').interesting_layers}
                    logger.log("gaze: %s" % w_gaze)
                if hasattr(gflag.qfunc_models.get('q_func'), 'interesting_layers'): # some model may not have this attribute
                    w_qfunc = {l.name : np.linalg.norm(l.get_weights()[0]) for l in gflag.qfunc_models.get('q_func').interesting_layers}
                    logger.log("qfunc: %s" % w_qfunc)

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
                td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights, num_iters<args.init_freeze_iter)
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
                completion = np.round(info["steps"] / args.num_steps * 100, 1)

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
                logger.log("ETA: " + pretty_eta(int(steps_left / fps_estimate)))
                logger.log("FPS: %.2f" % fps_estimate)
                logger.log(time.strftime("%Y-%m-%d %H:%M:%S %Z"))
                logger.log()
