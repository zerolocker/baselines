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
        # Use compile=False to avoid loading optimizer state, because loading it adds tons of variables to the Graph in Tensorboard, making it ugly
        # self.gaze_model_template = K.models.load_model(self.PATH, compile=False)

    def get_or_create(self, name, reuse):
        """ 
        Note: model weight is not set here because even if we do, U.initialize() will 
        be called below, and it will still override the weight we set here.
        so call initialze_weights_for_all_created_models() after U.initialize()
        """
        if name in self.models:
            assert reuse == True
            logger.log("model named %s is reused" % name)
        else:
            logger.log("model named %s is created" % name)
            assert reuse == False
            dropout = 0.5
            inputs=L.Input(shape=(84,84,4))
            x=inputs # inputs is used by the line "Model(inputs, ... )" below
            x=L.BatchNormalization()(x)

            conv1=L.Conv2D(32, (8,8), strides=4, padding='valid')
            x = conv1(x)
            x=L.Activation('relu')(x)
            x=L.BatchNormalization()(x)
            x=L.Dropout(dropout)(x)

            conv2=L.Conv2D(64, (4,4), strides=2, padding='valid')
            x = conv2(x)
            x=L.Activation('relu')(x)
            x=L.BatchNormalization()(x)
            x=L.Dropout(dropout)(x)

            conv3=L.Conv2D(64, (3,3), strides=1, padding='valid')
            x = conv3(x)
            x=L.Activation('relu')(x)
            x=L.BatchNormalization()(x)
            x=L.Dropout(dropout)(x)

            deconv1 = L.Conv2DTranspose(64, (3,3), strides=1, padding='valid')
            x = deconv1(x)
            x=L.Activation('relu')(x)
            x=L.BatchNormalization()(x)
            x=L.Dropout(dropout)(x)

            deconv2 = L.Conv2DTranspose(32, (4,4), strides=2, padding='valid')
            x = deconv2(x)
            x=L.Activation('relu')(x)
            x=L.BatchNormalization()(x)
            x=L.Dropout(dropout)(x)

            deconv3 = L.Conv2DTranspose(1, (8,8), strides=4, padding='valid')
            x = deconv3(x)

            outputs = L.Activation(MU.my_softmax)(x)
            self.models[name] = Model(inputs=inputs, outputs=outputs)

            if not gflag.exists('predicted_gaze'):
                gflag.add_read_only('predicted_gaze', outputs)
        return self.models[name]

    def initialze_weights_for_all_created_models(self):
        for model in self.models.values():
            model.load_weights(self.PATH)

gflag.add_read_only('global_keras_gaze_model_factory', KerasGazeModelFactory())

def model(img_in, num_actions, scope, reuse=False, layer_norm=False):
    """As described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf"""
    with tf.variable_scope(scope, reuse=reuse):
        x = img_in
        gaze_model = gflag.global_keras_gaze_model_factory.get_or_create(scope, reuse)
        g = gaze_model(img_in)
        embed()
        with tf.variable_scope("convnet_x"):
            # original architecture
            x = layers.convolution2d(x, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            x = layers.convolution2d(x, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            x = layers.convolution2d(x, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        with tf.variable_scope("convnet_g"):
            # original architecture
            g = layers.convolution2d(g, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            g = layers.convolution2d(g, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            g = layers.convolution2d(g, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        conv_out = layers.flatten(x+g)

        with tf.variable_scope("action_value"):
            value_out = layers.fully_connected(conv_out, num_outputs=512, activation_fn=None)
            if layer_norm:
                value_out = layer_norm_fn(value_out, relu=True)
            else:
                value_out = tf.nn.relu(value_out)
            value_out = layers.fully_connected(value_out, num_outputs=num_actions, activation_fn=None)
        return value_out


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
