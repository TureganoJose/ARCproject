from tensorflow.keras.layers import Dense, Flatten, Conv2D, DepthwiseConv2D, MaxPool2D, Input, Dropout, UpSampling2D, concatenate, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization, ReLU, ZeroPadding2D, Softmax, Add, MaxPooling2D, Conv2DTranspose, Activation, Reshape
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.applications import vgg16
from tensorflow.keras.layers import Layer
import tensorflow as tf
import keras.backend as K
import tensorflow.keras.utils as conv_utils

# HairMatteNet
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

# Tiramisu
# https://github.com/mad-Ye/FC-DenseNet-Keras/blob/master/layers.py
# https://arxiv.org/pdf/1611.09326.pdf
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

def model_simple_segmentation_v2(img_input):

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
    pool3 = MaxPool2D((2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)


    up1 = concatenate([UpSampling2D((2, 2))(conv4), conv3], axis=-1)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(up1)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)

    up2 = concatenate([UpSampling2D((2, 2))(conv5), conv2], axis=-1)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(up2)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)

    up3 = concatenate([UpSampling2D((2, 2))(conv6), conv1], axis=-1)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(up3)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)

    out = Conv2D( n_classes, (1, 1) , padding='same')(conv7)

    return out

def C_FCN_model(img_input):
    n_classes = 2
    conv1 = Conv2D(10, (3, 3), activation='relu', padding='same')(img_input)
    pool1 = MaxPool2D((2, 2))(conv1)

    conv2 = Conv2D(20, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPool2D((2, 2))(conv2)

    conv3 = Conv2D(40, (3, 3), activation='relu', padding='same')(pool2)
    pool3 = MaxPool2D((2, 2))(conv3)

    conv4 = Conv2D(n_classes, (1, 1), activation='sigmoid', padding='same')(pool3)

    out = UpSampling2D((8, 8), interpolation='bilinear')(conv4)
    return out

def C_UNETplusplus_model(img_input):
    n_classes = 2
    conv1 = Conv2D(8, (3, 3), activation='relu', padding='same')(img_input)
    pool1 = MaxPool2D((2, 2))(conv1)

    dw1 = DepthwiseConv2D((3, 3), activation='relu', padding='same', depth_multiplier=2)(pool1)
    pool2 = MaxPool2D((2, 2))(dw1)

    dw2 = DepthwiseConv2D((3, 3), activation='relu', padding='same', depth_multiplier=2)(pool2)
    pool3 = MaxPool2D((2, 2))(dw2)

    dw2 = DepthwiseConv2D((3, 3), activation='relu', padding='same', depth_multiplier=2)(pool3)
    pool4 = MaxPool2D((2, 2))(dw2)

    # dw3 = DepthwiseConv2D((3, 3), activation='relu', padding='same', depth_multiplier=2)(pool4)
    dw3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool4)
    pool5 = MaxPool2D((2, 2))(dw3)

    dw4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool5)
    # dw4 = DepthwiseConv2D((3, 3), activation='relu', padding='same', depth_multiplier=2)(pool5)

    deconv1 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(dw4)
    deconv2 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(deconv1)
    deconv3 = Conv2DTranspose (64, (2, 2), strides=(2, 2), padding='same')(deconv2)
    deconv4 = Conv2DTranspose (32, (2, 2), strides=(2, 2), padding='same')(deconv3)
    deconv5 = Conv2DTranspose (16, (2, 2), strides=(2, 2), padding='same')(deconv4)

    conv2 = Conv2D(16, (3, 3), activation='relu', padding='same')(deconv5)
    out = Conv2D(n_classes, (1, 1), activation='sigmoid', padding='same')(conv2)
    return out





def BN_ReLU_Conv(inputs, n_filters, filter_size=3, dropout_p=0.2):
    '''Apply successivly BatchNormalization, ReLu nonlinearity, Convolution and Dropout (if dropout_p > 0)'''

    l = BatchNormalization()(inputs)
    l = Activation('relu')(l)
    l = Conv2D(n_filters, filter_size, padding='same', kernel_initializer='he_uniform', kernel_regularizer=tf.keras.regularizers.l2(0.01))(l)
    if dropout_p != 0.0:
        l = Dropout(dropout_p)(l)
    return l

def TransitionDown(inputs, n_filters, dropout_p=0.2):
    """ Apply first a BN_ReLu_conv layer with filter size = 1, and a max pooling with a factor 2  """
    l = BN_ReLU_Conv(inputs, n_filters, filter_size=1, dropout_p=dropout_p)
    l = MaxPooling2D((2,2))(l)
    return l

def TransitionUp(skip_connection, block_to_upsample, n_filters_keep):
    '''Performs upsampling on block_to_upsample by a factor 2 and concatenates it with the skip_connection'''
    #Upsample and concatenate with skip connection
    l = Conv2DTranspose(n_filters_keep, kernel_size=3, strides=2, padding='same', kernel_initializer='he_uniform')(block_to_upsample)
    l = concatenate([l, skip_connection], axis=-1)
    return l

def SoftmaxLayer(inputs, n_classes):
    """
    Performs 1x1 convolution followed by softmax nonlinearity
    The output will have the shape (batch_size  * n_rows * n_cols, n_classes)
    """
    l = Conv2D(n_classes, kernel_size=1, padding='same', kernel_initializer='he_uniform')(inputs)
#    l = Reshape((-1, n_classes))(l)
    l = Activation('sigmoid')(l)#or softmax for multi-class
    return l

def model_Tiramisu(img_input):
    n_classes = 2
    n_filters_first_conv = 36
    n_pool = 5
    growth_rate = 12
    n_layers_per_block = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
    dropout_p = 0.3

    #####################
    # First Convolution #
    #####################
    stack = Conv2D(filters=n_filters_first_conv, kernel_size=3, padding='same', kernel_initializer='he_uniform')(img_input)
    n_filters = n_filters_first_conv

    #####################
    # Downsampling path #
    #####################
    skip_connection_list = []

    for i in range(n_pool):
        for j in range(n_layers_per_block[i]):
            l = BN_ReLU_Conv(stack, growth_rate, dropout_p=dropout_p)
            stack = concatenate([stack, l])
            n_filters += growth_rate

        skip_connection_list.append(stack)
        stack = TransitionDown(stack, n_filters, dropout_p)
    skip_connection_list = skip_connection_list[::-1]

    #####################
    #    Bottleneck     #
    #####################
    block_to_upsample = []

    for j in range(n_layers_per_block[n_pool]):
        l = BN_ReLU_Conv(stack, growth_rate, dropout_p=dropout_p)
        block_to_upsample.append(l)
        stack = concatenate([stack, l])
    block_to_upsample = concatenate(block_to_upsample)

    #####################
    #  Upsampling path  #
    #####################
    for i in range(n_pool):
        n_filters_keep = growth_rate * n_layers_per_block[n_pool + i]
        stack = TransitionUp(skip_connection_list[i], block_to_upsample, n_filters_keep)

        block_to_upsample = []
        for j in range(n_layers_per_block[n_pool + i + 1]):
            l = BN_ReLU_Conv(stack, growth_rate, dropout_p=dropout_p)
            block_to_upsample.append(l)
            stack = concatenate([stack, l])
        block_to_upsample = concatenate(block_to_upsample)

    #####################
    #  Softmax          #
    #####################
    output = SoftmaxLayer(stack, n_classes)
    return output

def FCN8_helper(img_input):
    n_classes = 2
    model = vgg16.VGG16(
        include_top=False,
        weights='imagenet', input_tensor=img_input,
        pooling=None,
        classes=1000)

    o = Conv2D(
        filters=4096,
        kernel_size=(
            7,
            7),
        padding="same",
        activation="relu",
        name="fc6")(
            model.output)
    o = Dropout(rate=0.5)(o)
    o = Conv2D(
        filters=4096,
        kernel_size=(
            1,
            1),
        padding="same",
        activation="relu",
        name="fc7")(o)
    o = Dropout(rate=0.5)(o)

    o = Conv2D(filters=n_classes, kernel_size=(1, 1), padding="same", activation="relu", kernel_initializer="he_normal",
               name="score_fr")(o)

    o = Conv2DTranspose(filters=n_classes, kernel_size=(2, 2), strides=(2, 2), padding="valid", activation=None,
                        name="score2")(o)
    fcn8 = Model(inputs=img_input, outputs=o)
    # mymodel.summary()
    return fcn8


def model_FCN8(img_input):
    n_classes = 2
    fcn8 = FCN8_helper(img_input)

    # Conv to be applied on Pool4
    skip_con1 = Conv2D(n_classes, kernel_size=(1, 1), padding="same", activation=None, kernel_initializer="he_normal",
                       name="score_pool4")(fcn8.get_layer("block4_pool").output)
    Summed = Add()([skip_con1, fcn8.output])

    x = Conv2DTranspose(n_classes, kernel_size=(2, 2), strides=(2, 2), padding="valid", activation=None,
                        name="score4")(Summed)

    ###
    skip_con2 = Conv2D(n_classes, kernel_size=(1, 1), padding="same", activation=None, kernel_initializer="he_normal",
                       name="score_pool3")(fcn8.get_layer("block3_pool").output)
    Summed2 = Add()([skip_con2, x])

    #####
    Up = Conv2DTranspose(n_classes, kernel_size=(8, 8), strides=(8, 8),
                         padding="valid", activation=None, name="upsample")(Summed2)

    Up = Reshape((-1, n_classes))(Up)
    Up = Activation("softmax")(Up)

    return Up


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