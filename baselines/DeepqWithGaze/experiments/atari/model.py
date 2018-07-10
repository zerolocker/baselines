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

    def get(self, name):
        return self.models[name]

    def get_or_create(self, gaze_model, name, reuse, num_actions, layer_norm, dueling):
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
            if dueling:
                state_score = L.Dense(512)(x)
                if layer_norm:
                   state_score=L.BatchNormalization()(state_score)
                state_score = L.Activation('relu')(state_score)
                state_score = L.Dense(1)(state_score)
            action_score = L.Dense(512)(x)
            if layer_norm:
                logger.log("Warning: layer_norm is set to True, but Keras doesn't have it. Replacing with BatchNorm.")
                action_score=L.BatchNormalization()(action_score)
            action_score=L.Activation('relu')(action_score)
            last_dense=L.Dense(num_actions, name="logits")
            action_score=last_dense(action_score)
            if dueling:
                def wrapped_tf_ops(s):
                    action_score, state_score = s
                    return action_score - tf.reduce_mean(action_score, 1, keep_dims=True) + state_score
                action_score = L.Lambda(wrapped_tf_ops)([action_score, state_score])
            model=Model(inputs=[imgs], outputs=[action_score, gaze_heatmaps])
            model.interesting_layers = [c1,c2,c3,last_dense] # export variable interesting_layers for monitoring in train.py
            self.models[name] = model
        return self.models[name]
        
    def initialze_weights_for_all_created_models(self):
        pass

    def get_weight_names_for_initial_freeze(self, model_name):
        return []

gflag.add_read_only('gaze_models', KerasGazeModelFactory())
gflag.add_read_only('qfunc_models', QFuncModelFactory())
logger.log("QFunc model filename is: " + __file__)

def model(img_in, num_actions, scope, reuse=False, layer_norm=False, dueling=False, return_gaze=False):
    with tf.variable_scope(scope, reuse=reuse):
        gaze_model = gflag.gaze_models.get_or_create(scope, reuse)
        action_model = gflag.qfunc_models.get_or_create(gaze_model, scope, reuse, num_actions, layer_norm, dueling)
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
