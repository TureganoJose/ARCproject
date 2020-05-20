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


from tensorflow.keras.models import load_model




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


def display(display_list):
    plt.figure(figsize=(15, 7))
    title = ['Original', 'Ground truth', 'Predicted Mask']
    plt.subplot(1, 3, 1)
    plt.title(title[0])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[0]))
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title(title[1])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[0]))
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[1]), alpha=0.15)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title(title[2])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[0]))
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[2][:, :, np.newaxis]), alpha=0.15)
    plt.axis('off')

    plt.show()




# load model
model = load_model('simple_CNN.h5')

# summarize model.
# model.summary()
# load dataset
input_height = 200
input_width = 200
processed_images, processed_labels = pre_process_data("C://Workspaces//ARCproject//Image_postpro//Training folder", input_height, input_width)

X_train_list, X_test_list, y_train_list, y_test_list = train_test_split(processed_images, processed_labels, test_size=0.2)

X_train = np.concatenate([arr[np.newaxis] for arr in X_train_list])/255.0
X_test = np.concatenate([arr[np.newaxis] for arr in X_test_list])/255.0
y_train = np.concatenate([arr[np.newaxis] for arr in y_train_list])
y_test = np.concatenate([arr[np.newaxis] for arr in y_test_list])

score = model.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))


for i in range( X_test.shape[0]):
    y_pred = model.predict(X_test[i][np.newaxis])
    y_pred = y_pred.reshape((200,200,2))
    image_list = []
    image_list.append(X_test[i])
    image_list.append(y_test[i])
    image_list.append(np.argmax(y_pred, axis=-1))
    display(image_list)
