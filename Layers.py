import keras.layers as layer
from keras import Model

# The new convolution layer, for more infomation, please refer to the paper
def conv(input, channel, kernel_size, name, strides=(1, 1)):
    x = layer.BatchNormalization(name=name + '_BN')(input)
    x = layer.ReLU(name=name + '_relu')(x)
    x = layer.Conv2D(channel, kernel_size, name=name+'_conv', activation='relu', padding='same', strides=strides)(x)
    return x

# Transition block connect two dense blocks together
def transition(x, k0, k, theta, name):
    x = layer.BatchNormalization(name=name + '_bn')(x)
    x = conv(x, int(k0 * theta), (1, 1), name=name+'_conv')
    x = layer.AveragePooling2D((2, 2), strides=(2, 2), name=name+'_pool')(x)
    # return next index of layer in order to be more convenient
    return x, int(k0 * theta)

# One DenseBlock
def denseBlock(input, k0, k, t, name):
    def H(x, l, name_h):
        x = layer.BatchNormalization(name=name_h + '_bn')(x)
        x = layer.ReLU(name=name_h + '_H_relu')(x)
        x = conv(x, k0 + k * (l - 1), (3, 3), name=name_h + '_H_conv')
        return x

    prev = [input]
    for i in range(1, t + 1):
        if(len(prev) == 1):
            c = input
        else:
            c = layer.concatenate(prev, axis=-1)
        x = H(c, i + 1, name + '_' + str(i))
        prev.append(x)
    return x, k0 + k * t