import argparse, gym, os, keras as K, tensorflow as tf, numpy as np, time

from gym.monitoring import VideoRecorder

import baselines.common.tf_util as U
import baselines.DeepqWithGaze.misc_utils as MU

from baselines import DeepqWithGaze
from baselines.common.misc_util import (
    boolean_flag,
    SimpleMonitor,
)
from baselines.common.atari_wrappers_deprecated import wrap_dqn
from baselines.DeepqWithGaze.experiments.atari.model import model, dueling_model
import matplotlib.pyplot as plt
from IPython import embed

def parse_args():
    parser = argparse.ArgumentParser("Run an already learned DQN model.")
    # Environment
    parser.add_argument("--env", type=str, required=True, help="name of the game")
    parser.add_argument("--model-dir", type=str, default=None, help="load model from this directory. ")
    parser.add_argument("--video", type=str, default=None, help="Path to mp4 file where the video of first episode will be recorded.")
    boolean_flag(parser, "stochastic", default=True, help="whether or not to use stochastic actions according to models eps value")
    boolean_flag(parser, "dueling", default=False, help="whether or not to use dueling model")
    boolean_flag(parser, "layer-norm", default=False, help="whether or not to use layer norm (should be True if param_noise is used)")

    boolean_flag(parser, "debug-mode", default=False, help="if true ad-hoc debug-related code will be run and training may stop halfway")
    return parser.parse_args()


def make_env(game_name):
    env = gym.make(game_name + "NoFrameskip-v4")
    env = SimpleMonitor(env)
    env = wrap_dqn(env)
    return env


if __name__ == '__main__':
    with U.make_session(4) as sess:
        args = parse_args()
        env = make_env(args.env)
        MU.keras_model_serialization_bug_fix()
        gaze_model = K.models.load_model("baselines/DeepqWithGaze/ImgOnly_gazeModels/seaquest-dp0.4-DQN+BNonInput.hdf5")
        K.backend.set_learning_phase(0)
        if args.debug_mode:
            debug_gaze_in = None 
            #debug_saved_gaze_in = []
            def tf_op_set_debug_tensor(x):
                global debug_gaze_in
                debug_gaze_in = x
                #debug_saved_gaze_in.append(np.copy(x))
                return x
        def model_wrapper(img_in, num_actions, scope, **kwargs):
            actual_model = dueling_model if args.dueling else model
            gaze_in = img_in
            GHmap = gaze_model(gaze_in)
            if args.debug_mode:
                debug_tensor = tf.py_func(tf_op_set_debug_tensor, [tf.concat([img_in, GHmap], axis=-1)], [tf.float32], stateful=True, name='debug_tensor')
                with tf.control_dependencies(debug_tensor):
                    img_and_gaze_combined = tf.concat([img_in, GHmap*img_in], axis=-1)
            else:
                img_and_gaze_combined = tf.concat([img_in, GHmap*img_in], axis=-1)
            return actual_model(img_and_gaze_combined, num_actions, scope, **kwargs)
        
        act = DeepqWithGaze.build_act(
            make_obs_ph=lambda name: U.Uint8Input(env.observation_space.shape, name=name),
            q_func=model_wrapper,
            num_actions=env.action_space.n)
        U.load_state(os.path.join(args.model_dir, "saved"))
        gaze_model.load_weights('baselines/DeepqWithGaze/ImgOnly_gazeModels/seaquest-dp0.4-DQN+BNonInput.hdf5')

        num_episodes = 0
        video_recorder = None
        video_recorder = VideoRecorder(env, args.video, enabled=args.video is not None)
        obs = env.reset()
        if args.debug_mode:
            fig, axarr = plt.subplots(2,3)
            plt.show(block=False)
            debug_embed_last_time = time.time() # TODO this is temporary. delete it and its related code
            debug_embed_freq_sec = 600
        while True:
            if args.debug_mode and debug_gaze_in is not None:
                for i in range(4):
                    axarr[int(i/2), i%2].cla()
                    axarr[int(i/2), i%2].imshow(debug_gaze_in[0,:,:,i]) 
                axarr[1,2].cla()
                axarr[1,2].imshow(debug_gaze_in[0,:,:,4])
                fig.canvas.draw()

            if args.debug_mode and time.time() - debug_embed_last_time > debug_embed_freq_sec:
                embed()
                debug_embed_last_time = time.time()
            env.unwrapped.render()
            video_recorder.capture_frame()
            action = act(np.array(obs)[None], stochastic=args.stochastic)[0]
            obs, rew, done, info = env.step(action)
            if done:
                obs = env.reset()
            if len(info["rewards"]) > num_episodes:
                if len(info["rewards"]) == 1 and video_recorder.enabled:
                    # save video of first episode
                    print("Saved video.")
                    video_recorder.close()
                    video_recorder.enabled = False
                print(info["rewards"][-1])
                num_episodes = len(info["rewards"])
