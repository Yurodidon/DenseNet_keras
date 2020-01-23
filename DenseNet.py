from Layers import *
from Configs import *
import keras.layers as l
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
    conv1 = conv(input, k0, (7, 7), strides=(2, 2), name='conv1')
    pool1 = layer.MaxPooling2D((3, 3), strides=(2, 2), name='pool1', padding='same')(conv1)

    c = {'k' : k, 'k0' : k0}
    dense1, l1 = denseBlock(pool1, 3, config[1], c, 'dense1')
    trans1, l2 = transition(dense1, l1, config[2], c, 'trans1')

    dense2, l3 = denseBlock(trans1, l2, config[3], c, 'dense2')
    trans2, l4 = transition(dense2, l3, config[4], c, 'trans2')

    dense3, l5 = denseBlock(trans2, l4, config[5], c, 'dense3')
    trans3, l6 = transition(dense3, l5, config[6], c, 'trans3')

    dense4, l7 = denseBlock(trans3, l6, config[7], c, 'dense4')
    global_pool = layer.AveragePooling2D((7, 7), name='global_pool')(dense4)

    flatten = layer.Flatten(name='flat')(global_pool)
    prediction = layer.Dense(num_classes, name='prediction')(flatten)

    model = Model(input=input, output=prediction)
    return model