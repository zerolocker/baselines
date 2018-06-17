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

# This particular model py file doesn't use gaze model. So this is an empty
class KerasGazeModelFactory:
    def __init__(self):
        self.models = {}
    def get(self, name):
        return self.models[name]
    def get_or_create(self, name, reuse):
        if name in self.models:
            assert reuse == True
        else:
            imgs=L.Input(shape=(84,84,4))
            model=Model(inputs=[imgs], outputs=[imgs])
            self.models[name] = model
        return self.models[name]
    def initialze_weights_for_all_created_models(self):
        pass

class QFuncModelFactory:
    def __init__(self):        
        self.GAZE_PRETRAIN_PATH = "baselines/DeepqWithGaze/ImgOnly_gazeModels/seaquest-dp0.4-DQN+BNonInput.hdf5"
        self.models = {}
        # Use compile=False to avoid loading optimizer state, because loading it adds tons of variables to the Graph in Tensorboard, making it ugly
        self.gaze_model_template = K.models.load_model(self.GAZE_PRETRAIN_PATH, compile=False)

    def get(self, name):
        return self.models[name]

    def get_or_create(self, name, reuse, num_actions, layer_norm):
        if name in self.models:
            assert reuse == True
            logger.log("QFunc model named %s is reused" % name)
        else:
            logger.log("QFunc model named %s is created" % name)
            assert reuse == False
            imgs=L.Input(shape=(84,84,4))
            x=imgs
            x=L.Conv2D(32, (8,8), strides=4, padding='same', activation="relu", name="conv2d_1")(x)
            x=L.Conv2D(64, (4,4), strides=2, padding='same', activation="relu", name="conv2d_2")(x)
            x=L.Conv2D(64, (3,3), strides=1, padding='same', activation="relu", name="conv2d_3")(x)
            x=L.Flatten()(x)
            x=L.Dense(512)(x)
            if layer_norm:
                logger.log("Warning: layer_norm is set to True, but Keras doesn't have it. Replacing with BatchNorm.")
                x=L.BatchNormalization()(x)
            x=L.Activation('relu')(x)
            logits=L.Dense(num_actions, name="logits")(x)

            model=Model(inputs=[imgs], outputs=[logits])
            self.models[name] = model
        return self.models[name]

    def initialze_weights_for_all_created_models(self):
        layers_to_init = ['conv2d_1', 'conv2d_2', 'conv2d_3']
        for model in self.models.values():
            for layer_name in layers_to_init:
                W = self.gaze_model_template.get_layer(layer_name).get_weights()
                model.get_layer(layer_name).set_weights(W)


gflag.add_read_only('gaze_models', KerasGazeModelFactory())
gflag.add_read_only('qfunc_models', QFuncModelFactory())
logger.log("QFunc model filename is: " + __file__)

def model(img_in, num_actions, scope, reuse=False, layer_norm=False, return_gaze=False):
    if return_gaze:  # no gaze available for this model file
        return img_in * 0.0;

    with tf.variable_scope(scope, reuse=reuse):
        gaze_model = gflag.gaze_models.get_or_create(scope, reuse)
        action_model = gflag.qfunc_models.get_or_create(scope, reuse, num_actions, layer_norm)
        value_out  = action_model([img_in])
        return value_out


def dueling_model(img_in, num_actions, scope, reuse=False, layer_norm=False):
    assert False, "not maintained because I was lazy; TODO: it should be the same as model() except for the dueling part."

