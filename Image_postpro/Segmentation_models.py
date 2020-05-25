from tensorflow.keras.layers import Dense, Flatten, Conv2D, DepthwiseConv2D, MaxPool2D, Input, Dropout, UpSampling2D, concatenate
from tensorflow.keras.layers import BatchNormalization, ReLU, ZeroPadding2D, Softmax, Add
import numpy as np
import tensorflow as tf


def depthwise_conv_block(inputs, pointwise_conv_filters, alpha, depth_multiplier=1, strides=(1, 1), block_id=1):
    channel_axis = -1 # channels are last
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)
    x = ZeroPadding2D((1, 1), data_format='channels_last',
                      name='conv_pad_%d' % block_id)(inputs)
    x = DepthwiseConv2D((3, 3), data_format='channels_last',
                        padding='valid',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block_id)(x)
    x = BatchNormalization(
        axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
    x = ReLU(6., name='conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1), data_format='channels_last',
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(axis=channel_axis,
                           name='conv_pw_%d_bn' % block_id)(x)
    return ReLU(6., name='conv_pw_%d_relu' % block_id)(x)

def conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):

    channel_axis = -1
    filters = int(filters * alpha)
    x = ZeroPadding2D(padding=(1, 1), name='conv1_pad',
                      data_format='channels_last')(inputs)
    x = Conv2D(filters, kernel, data_format='channels_last',
               padding='valid',
               use_bias=False,
               strides=strides,
               name='conv1')(x)
    x = BatchNormalization(axis=channel_axis, name='conv1_bn')(x)
    return ReLU(6., name='conv1_relu')(x)

def up_depthwise_conv(input1, input2, pointwise_conv_filters, alpha, depth_multiplier=1, strides=(1, 1), block_id=1):

    x1 = UpSampling2D((2, 2))(input1)


    x2 = Conv2D(pointwise_conv_filters, (1, 1), data_format='channels_last',
              padding='same',
              use_bias=False,
              strides=(1, 1),
              name='up_conv_pw_%d' % block_id)(input2)

    concat = Add()([x1, x2])

    x = DepthwiseConv2D((3, 3), data_format='channels_last',
                        padding='valid',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        name='up_conv_dw_%d' % block_id)(concat)
    x = ZeroPadding2D(padding=(1, 1), name='conv_pad_%d' % block_id,
                      data_format='channels_last')(x)
    x = Conv2D(64, (1, 1), data_format='channels_last',
               padding='valid',
               use_bias=False,
               strides=(1, 1),
               name='up_conv_after_upsampling_%d' % block_id)(x)
    return ReLU(6., name='up_conv1_relu_%d' % block_id)(x)

def model_HairMatteNet(img_input):
    n_classes = 2
    # Encoder
    alpha = 1.0
    depth_multiplier = 1
    dropout = 1e-3
    x = conv_block(img_input, 32, alpha, strides=(2, 2))
    x = depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)
    f1 = x

    x = depthwise_conv_block(x, 128, alpha, depth_multiplier,
                              strides=(2, 2), block_id=2)
    x = depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)
    f2 = x

    x = depthwise_conv_block(x, 256, alpha, depth_multiplier,
                              strides=(2, 2), block_id=4)
    x = depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5)
    f3 = x

    x = depthwise_conv_block(x, 512, alpha, depth_multiplier,
                              strides=(2, 2), block_id=6)
    x = depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7)
    x = depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8)
    x = depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9)
    x = depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10)
    x = depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11)
    f4 = x

    x = depthwise_conv_block(x, 1024, alpha, depth_multiplier,
                              strides=(2, 2), block_id=12)
    x = depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=13)
    f5 = x

    x = up_depthwise_conv(f5, f4, 1024, alpha, depth_multiplier=1, strides=(1, 1), block_id=14)
    x = up_depthwise_conv(x, f3, 64, alpha, depth_multiplier=1, strides=(1, 1), block_id=15)
    x = up_depthwise_conv(x, f2, 64, alpha, depth_multiplier=1, strides=(1, 1), block_id=16)
    x = up_depthwise_conv(x, f1, 64, alpha, depth_multiplier=1, strides=(1, 1), block_id=17)
    x = UpSampling2D((2, 2))(x)
    x = DepthwiseConv2D((3, 3), data_format='channels_last',
                        padding='valid',
                        depth_multiplier=depth_multiplier,
                        strides=(1, 1),
                        use_bias=False,
                        name='Last_up_conv_dw')(x)
    x = ZeroPadding2D(padding=(1, 1), name='Last_conv_pad',
                      data_format='channels_last')(x)
    x = Conv2D(64, (1, 1), data_format='channels_last',
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='Last_up_conv_after_upsampling')(x)
    x = ReLU(6., name='up_conv1_relu')(x)
    x = Conv2D(n_classes, (1, 1), data_format='channels_last',
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='Last_conv')(x)
    out = Softmax(axis=-1)(x)
    return out

def model_simple_segmentation(img_input):

    n_classes = 2

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(img_input)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPool2D((2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPool2D((2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)

    up1 = concatenate([UpSampling2D((2, 2))(conv3), conv2], axis=-1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)

    up2 = concatenate([UpSampling2D((2, 2))(conv4), conv1], axis=-1)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)

    out = Conv2D( n_classes, (1, 1) , padding='same')(conv5)

    return out

def unet_model(img_input):
    x = img_input

    # Downsampling through the model
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
      x = up(x)
      concat = tf.keras.layers.Concatenate()
      x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
        2, 3, strides=2,
        padding='same')  #64x64 -> 128x128

    x = last(x)

    return tf.keras.Model(inputs=img_input, outputs=x)