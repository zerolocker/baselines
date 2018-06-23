import tensorflow as tf
import tensorflow.contrib.layers as layers
import keras as K, keras.layers as L
from keras.models import Model # keras/engine/training.py
import baselines.DeepqWithGaze.misc_utils as MU
from baselines.common.gflag import gflag
from baselines import logger
from IPython import embed


def layer_norm_fn(x, relu=True):
    x = layers.layer_norm(x, scale=True, center=True)
    if relu:
        x = tf.nn.relu(x)
    return x

class KerasGazeModelFactory:
    def __init__(self):
        self.models = {}
        self.PATH = "baselines/DeepqWithGaze/ImgOnly_gazeModels/seaquest-dp0.4-DQN+BNonInput.hdf5"

    def get(self, name):
        return self.models[name]

    def get_or_create(self, name, reuse):
        """
        Note: model weight is not set here because even if we do, U.initialize() will 
        be called below, and it will still override the weight we set here.
        so call initialze_weights_for_all_created_models() after U.initialize()
        """
        if name in self.models:
            assert reuse == True
            logger.log("Gaze model named %s is reused" % name)
        else:
            logger.log("Gaze model named %s is created" % name)
            # Use compile=False to avoid loading optimizer state, because loading it adds tons of variables to the Graph in Tensorboard, making it ugly
            model = K.models.load_model(self.PATH, compile=False)
            model.interesting_layers = [model.layers[-2]] # export variable interesting_layers for monitoring in train.py
            self.models[name] = model
        return self.models[name]

    def initialze_weights_for_all_created_models(self):
        for model in self.models.values():
            model.load_weights(self.PATH)

class QFuncModelFactory:
    def __init__(self):
        self.models = {}
        # Use compile=False to avoid loading optimizer state, because loading it adds tons of variables to the Graph in Tensorboard, making it ugly
        # self.gaze_model_template = K.models.load_model(self.PATH, compile=False)

    def get(self, name):
        return self.models[name]

    def get_or_create(self, gaze_model, name, reuse, num_actions, layer_norm):
        if name in self.models:
            assert reuse == True
            logger.log("QFunc model named %s is reused" % name)
        else:
            logger.log("QFunc model named %s is created" % name)
            assert reuse == False
            imgs=L.Input(shape=(84,84,4))

            gaze_heatmaps = gaze_model(imgs)
            g=gaze_heatmaps
            g=L.BatchNormalization()(g) # With this, gaze_model's gradient input is 50x larger; otherwise it won't train

            x=imgs
            x=L.Multiply(name="img_mul_gaze")([x,g])
            x_intermediate=x
            c1=L.Conv2D(32, (8,8), strides=4, padding='same', activation="relu", name='mul_c1')
            x=c1(x)
            c2=L.Conv2D(64, (4,4), strides=2, padding='same', activation="relu", name='mul_c2')
            x=c2(x)
            c3=L.Conv2D(64, (3,3), strides=1, padding='same', activation="relu", name='mul_c3')
            x=c3(x)
            # ============================ channel 2 ============================
            orig_x=imgs
            orig_x=L.Conv2D(32, (8,8), strides=4, padding='same', activation="relu")(orig_x)
            orig_x=L.Conv2D(64, (4,4), strides=2, padding='same', activation="relu")(orig_x)
            orig_x=L.Conv2D(64, (3,3), strides=1, padding='same', activation="relu")(orig_x)

            x=L.Average()([x,orig_x])
            x=L.Flatten()(x)
            x=L.Dense(512)(x)
            if layer_norm:
                logger.log("Warning: layer_norm is set to True, but Keras doesn't have it. Replacing with BatchNorm.")
                x=L.BatchNormalization()(x)
            x=L.Activation('relu')(x)
            last_dense=L.Dense(num_actions, name="logits")
            logits=last_dense(x)

            model=Model(inputs=[imgs], outputs=[logits, x_intermediate])
            model.interesting_layers = [c1,c2,c3,last_dense] # export variable interesting_layers for monitoring in train.py
            self.models[name] = model
        return self.models[name]
        
    def initialze_weights_for_all_created_models(self):
        pass

gflag.add_read_only('gaze_models', KerasGazeModelFactory())
gflag.add_read_only('qfunc_models', QFuncModelFactory())
logger.log("QFunc model filename is: " + __file__)

def model(img_in, num_actions, scope, reuse=False, layer_norm=False, return_gaze=False):
    with tf.variable_scope(scope, reuse=reuse):
        gaze_model = gflag.gaze_models.get_or_create(scope, reuse)
        action_model = gflag.qfunc_models.get_or_create(gaze_model, scope, reuse, num_actions, layer_norm)
        value_out, gaze  = action_model([img_in])

        if gflag.debug_mode and scope=='q_func' and reuse==False:
            def tf_op_set_debug_tensor(x):
                # TODO HACKY!: this violates and bypasses gflag's immutability, change this
                gflag._dict['debug_gaze_in'] = x  # line when I have more time to figure out a less hacky solution
                return x
            debug_tensor = tf.py_func(tf_op_set_debug_tensor, [tf.concat([img_in, gaze], axis=-1)], [tf.float32], stateful=True, name='debug_tensor')
            with tf.control_dependencies(debug_tensor):
                value_out = tf.identity(value_out)

        return value_out if not return_gaze else gaze


def dueling_model(img_in, num_actions, scope, reuse=False, layer_norm=False):
    """As described in https://arxiv.org/abs/1511.06581"""
    assert False, "not maintained because I was lazy; TODO: it should be the same as model() except for the dueling part."
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        with tf.variable_scope("convnet"):
            # original architecture
            out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        conv_out = layers.flatten(out)

        with tf.variable_scope("state_value"):
            state_hidden = layers.fully_connected(conv_out, num_outputs=512, activation_fn=None)
            if layer_norm:
                state_hidden = layer_norm_fn(state_hidden, relu=True)
            else:
                state_hidden = tf.nn.relu(state_hidden)
            state_score = layers.fully_connected(state_hidden, num_outputs=1, activation_fn=None)
        with tf.variable_scope("action_value"):
            actions_hidden = layers.fully_connected(conv_out, num_outputs=512, activation_fn=None)
            if layer_norm:
                actions_hidden = layer_norm_fn(actions_hidden, relu=True)
            else:
                actions_hidden = tf.nn.relu(actions_hidden)
            action_scores = layers.fully_connected(actions_hidden, num_outputs=num_actions, activation_fn=None)
            action_scores_mean = tf.reduce_mean(action_scores, 1)
            action_scores = action_scores - tf.expand_dims(action_scores_mean, 1)
        return state_score + action_scores
