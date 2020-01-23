import keras.layers as layer
from keras import Model

# The new convolution layer, for more infomation, please refer to the paper
def conv(input, channel, kernel_size, name, strides=(1, 1)):
    x = layer.BatchNormalization(name=name + '_BN')(input)
    x = layer.ReLU(name=name + '_relu')(x)
    x = layer.Conv2D(channel, kernel_size, name=name+'_conv', activation='relu', padding='same', strides=strides)(x)
    return x

# Transition block connect two dense blocks together
def transition(x, l, theta, config, name):
    k0, k = config['k0'], config['k']
    x = layer.BatchNormalization(name=name + '_bn')(x)
    x = conv(x, int((k0 + k * (l - 1)) * theta), (1, 1), name=name+'_conv')
    x = layer.AveragePooling2D((2, 2), strides=(2, 2), name=name+'_pool')(x)
    # return next index of layer in order to be more convenient
    return x, l + 2

# One DenseBlock
def denseBlock(input, s_l, t, config, name):
    k0, k = config['k0'], config['k']
    def H(x, l, name_h):
        x = layer.BatchNormalization(name=name_h + '_bn')(x)
        x = layer.ReLU(name=name_h + '_H_relu')(x)
        x = conv(x, k0 + k * (l - 1), (3, 3), name=name_h + '_H_conv')
        return x

    prev = [input]
    for i in range(0, t):
        if(len(prev) == 1):
            c = input
        else:
            c = layer.concatenate(prev, axis=-1)
        x = H(c, s_l + i, name + '_' + str(i + 1))
        prev.append(x)
    return x, s_l + t