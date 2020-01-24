from Layers import *
from Configs import *
import keras.layers as layer
import keras

# This is the DenseNets!
# You can choose the config, and then call DenseNet function with config
# DenseNet function will return a Model, it is the DenseNet
# configXXX -> config for DenseNet with XXX layers
# Enable configs: config121, config169, config201, config264

def DenseNet(config):
    input_size = config[0]['input_size']
    num_classes = config[0]['num_classes']
    k0 = input_size[2]
    k = config[0]['k']

    input = layer.Input(shape=input_size, name='input')
    conv1 = conv(input, k * 2, (7, 7), strides=(2, 2), name='conv1')
    pool1 = layer.MaxPooling2D((3, 3), strides=(2, 2), name='pool1', padding='same')(conv1)

    c = {'k' : k, 'k0' : k0}
    dense1, k1 = denseBlock(pool1, k * 2, k, config[1], 'dense1')
    trans1, k2 = transition(dense1, k1, k, config[2], 'trans1')

    dense2, k3 = denseBlock(trans1, k2, k, config[3], 'dense2')
    trans2, k4 = transition(dense2, k3, k, config[4], 'trans2')

    dense3, k5 = denseBlock(trans2, k4, k, config[5], 'dense3')
    trans3, k6 = transition(dense3, k5, k, config[6], 'trans3')

    dense4, k7 = denseBlock(trans3, k6, k, config[7], 'dense4')
    global_pool = layer.AveragePooling2D((7, 7), name='global_pool')(dense4)

    flatten = layer.Flatten(name='flat')(global_pool)
    prediction = layer.Dense(num_classes, name='prediction')(flatten)

    model = Model(input=input, output=prediction)
    return model