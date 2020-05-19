#https://divamgupta.com/image-segmentation/2019/06/06/deep-learning-semantic-segmentation-keras.html

# import labelme
# https://github.com/wkentaro/labelme/blob/master/README.md

import glob, os
from shutil import copyfile, rmtree
import imgaug as ia
import imgaug.augmenters as iaa
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D, DepthwiseConv2D, MaxPool2D, Input, Dropout, UpSampling2D, concatenate
from tensorflow.keras.layers import *
import tensorflow as tf
from tensorflow_core.examples.models.pix2pix import pix2pix
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# files = glob.iglob(os.path.join('D://Rotated_new_camera_position_test2', "*.json"))
# for file in files:
#     if os.path.isfile(file):
#         #start_pos = file.find("\\2")
#         start_pos = file.find("\\T")
#         end_pos = file.find(".json")
#         base_filename = file[start_pos+1:end_pos]
#         base_folder = base_filename.replace(".","_")
#         path = file[:start_pos]+"\\"
#         os.system("labelme_json_to_dataset " + file) # create dataset
#         os.rename( path + base_folder +"_json"+ "\\label.png" , path + base_folder + "_json\\" + base_filename + "_json.png" )
#         copyfile( path + base_folder + "_json\\" + base_filename + "_json.png",  path + base_filename + "_json.png")
#         rmtree( path + base_folder + "_json\\" )


def pre_process_data(folder, input_height: int, input_width: int):
    input_files = glob.iglob(os.path.join(folder, "*.jpg"))

    processed_images = []
    processed_labels = []

    # Loop through image folder
    for file in input_files:
        start_pos = file.find("\\2")
        end_pos = file.find(".jpg")
        base_filename = file[start_pos+1:end_pos]
        label_file = file[:end_pos] + "_json.png"
        # output_files = glob.iglob(os.path.join(folder, "*.jpg"))
        if not os.path.isfile(label_file):
            print("No label file for {0}".format(file))

        else:
            # Resize inputs and outputs
            resized_in = np.asarray(Image.open(file).resize((input_width, input_height)))
            resized_out = np.asarray(Image.open(label_file).resize((input_width, input_height)))
            resized_out = resized_out[:,:,np.newaxis]

            processed_images.append(resized_in)
            processed_labels.append(resized_out)

            # Augment data
            resized_aug_in, resized_aug_out = augment_seg(resized_in, resized_out)
            resized_aug_out = resized_aug_out[:, :, np.newaxis]

            # Plot augmented data
            # plt.figure()
            # plt.imshow(resized_aug_in)
            # plt.imshow(resized_aug_out, alpha=0.2)

            processed_images.append(resized_aug_in)
            processed_labels.append(resized_aug_out)
    return processed_images, processed_labels

def augment_seg(img, seg):
    seq = iaa.Sequential([
        iaa.Crop(px=(0, 16)),  # crop images from each side by 0 to 16px (randomly chosen)
        iaa.Fliplr(0.5),  # horizontally flip 50% of the images
        iaa.GaussianBlur(sigma=(0, 3.0))  # blur images with a sigma of 0 to 3.0
    ])
    aug_det = seq.to_deterministic()
    image_aug = aug_det.augment_image(img)

    segmap = ia.SegmentationMapOnImage(seg, nb_classes=np.max(seg) + 1, shape=img.shape)
    segmap_aug = aug_det.augment_segmentation_maps(segmap)
    segmap_aug = segmap_aug.get_arr_int()

    return image_aug, segmap_aug


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


def model_definition(img_input):

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

    # Block 1

    # weight_decay = 0.
    # x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', kernel_regularizer=l2(weight_decay))(
    #     img_input)
    # x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', kernel_regularizer=l2(weight_decay))(
    #     x)
    # x = MaxPool2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    #
    # # Block 2
    # x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1',
    #            kernel_regularizer=l2(weight_decay))(x)
    # x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2',
    #            kernel_regularizer=l2(weight_decay))(x)
    # x = MaxPool2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    #
    # # Block 3
    # x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1',
    #            kernel_regularizer=l2(weight_decay))(x)
    # x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2',
    #            kernel_regularizer=l2(weight_decay))(x)
    # x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3',
    #            kernel_regularizer=l2(weight_decay))(x)
    # x = MaxPool2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    #
    # # Block 4
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1',
    #            kernel_regularizer=l2(weight_decay))(x)
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2',
    #            kernel_regularizer=l2(weight_decay))(x)
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3',
    #            kernel_regularizer=l2(weight_decay))(x)
    # x = MaxPool2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    #
    # # Block 5
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1',
    #            kernel_regularizer=l2(weight_decay))(x)
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2',
    #            kernel_regularizer=l2(weight_decay))(x)
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3',
    #            kernel_regularizer=l2(weight_decay))(x)
    # x = MaxPool2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    #
    # # Convolutional layers transfered from fully-connected layers
    # x = Conv2D(4096, (7, 7), activation='relu', padding='same', name='fc1', kernel_regularizer=l2(weight_decay))(x)
    # x = Dropout(0.5)(x)
    # x = Conv2D(4096, (1, 1), activation='relu', padding='same', name='fc2', kernel_regularizer=l2(weight_decay))(x)
    # x = Dropout(0.5)(x)
    # # classifying layer
    # x = Conv2D(n_classes, (1, 1), kernel_initializer='he_normal', activation='linear', padding='valid', strides=(1, 1),
    #            kernel_regularizer=l2(weight_decay))(x)
    #
    # out = BilinearUpSampling2D(size=(32, 32))(x)

    return out

def display(display_list):
    plt.figure(figsize=(15, 15))
    title = ['True Mask', 'Predicted Mask']
    plt.subplot(1, 2, 1)
    plt.title(title[0])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[0]))
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[1]), alpha=0.15)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(title[1])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[0]))
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[2][:, :, np.newaxis]), alpha=0.15)
    plt.axis('off')

    plt.show()


input_height = 300
input_width = 300
learning_rate = 0.01
batch_size = 5
training_epochs = 60

processed_images, processed_labels = pre_process_data("D://Training folder", input_height, input_width)

X_train_list, X_test_list, y_train_list, y_test_list = train_test_split(processed_images, processed_labels, test_size=0.2)

X_train = np.concatenate([arr[np.newaxis] for arr in X_train_list])/255.0
X_test = np.concatenate([arr[np.newaxis] for arr in X_test_list])/255.0
y_train = np.concatenate([arr[np.newaxis] for arr in y_train_list])
y_test = np.concatenate([arr[np.newaxis] for arr in y_test_list])

# Simple model
img_input = Input(shape=[input_height, input_width, 3])
model_output = model_definition(img_input)
model = keras.Model(inputs=img_input, outputs=model_output, name="simple_CNN")

# # img_input = Input(shape=[input_height, input_width, 3])
# base_model = tf.keras.applications.MobileNetV2(input_shape=[input_height, input_width, 3], include_top=False)
# # Use the activations of these layers
# layer_names = [
#     'block_1_expand_relu',   # 64x64
#     'block_3_expand_relu',   # 32x32
#     'block_6_expand_relu',   # 16x16
#     'block_13_expand_relu',  # 8x8
#     'block_16_project',      # 4x4
# ]
#
# layers = [base_model.get_layer(name).output for name in layer_names]
# down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)
# down_stack.trainable = True
# up_stack = [
#     pix2pix.upsample(512, 3),  # 4x4 -> 8x8
#     pix2pix.upsample(256, 3),  # 8x8 -> 16x16
#     pix2pix.upsample(128, 3),  # 16x16 -> 32x32
#     pix2pix.upsample(64, 3),   # 32x32 -> 64x64
# ]
#
#
# model = unet_model(img_input)



#tf.keras.utils.plot_model(model, show_shapes=True)
model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), #loss=keras.losses.categorical_crossentropy,
              metrics=['accuracy'])  # ['accuracy']

history = model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=training_epochs,
          verbose=1,
          validation_data=(X_test, y_test))

score = model.evaluate(X_test, y_test, verbose=0)

model.save('simple_CNN_300x300.h5')

y_pred = model.predict(X_test)
for i in range( X_test.shape[0]):
    image_list = []
    image_list.append(X_test[i])
    image_list.append(y_test[i])
    image_list.append(np.argmax(y_pred[i], axis=-1))
    display(image_list)


# plt.figure()
# plt.imshow(X_test[10])
# plt.imshow(np.argmax(y_pred[10] ,axis=-1) , alpha=0.2)

print("hola")


