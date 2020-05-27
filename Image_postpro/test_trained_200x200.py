#https://divamgupta.com/image-segmentation/2019/06/06/deep-learning-semantic-segmentation-keras.html

# import labelme
# https://github.com/wkentaro/labelme/blob/master/README.md

import glob, os
from tensorflow.keras.layers import *
import imgaug as ia
import imgaug.augmenters as iaa
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time


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
            resized_in = np.asarray(Image.open(file).resize((input_width, input_height )))
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

def convert_to_png(img, a):
    #alpha and img must have the same dimenstons

    fin_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    fin_img = fin_img.astype(np.float32)
    alpha = a
    # plt.imshow(alpha)
    # plt.title('alpha image')
    # plt.show()
    # plt.imshow(img)
    # plt.title('original image')
    # plt.show()
    # plt.imshow(alpha)
    # plt.title('fin alpha image')
    # plt.show()
    fin_img[:,:, 0] = img[:,:,0]/255.0
    fin_img[:,:, 1] = img[:,:,1]/255.0
    fin_img[:,:, 2] = img[:,:,2]/255.0
    fin_img[:,:, 3] = 1-alpha*0.95
    # plt.imshow(fin_img)
    # plt.title('fin image')
    # plt.show()
    return fin_img




# load model
model = load_model('HairMatteNet.h5')

# summarize model.
# model.summary()
# load dataset
input_height = 224
input_width = 224
processed_images, processed_labels = pre_process_data("C://Workspaces//ARCproject//Image_postpro//Training folder", input_height, input_width)

X_train_list, X_test_list, y_train_list, y_test_list = train_test_split(processed_images, processed_labels, test_size=0.2)

X_train = np.concatenate([arr[np.newaxis] for arr in X_train_list])/255.0
X_test = np.concatenate([arr[np.newaxis] for arr in X_test_list])/255.0
y_train = np.concatenate([arr[np.newaxis] for arr in y_train_list])
y_test = np.concatenate([arr[np.newaxis] for arr in y_test_list])

# Loops through test images and plots them comparing original, its mask and predicted mask
# for i in range( X_test.shape[0]):
#     y_pred = model.predict(X_test[i][np.newaxis])
#     y_pred = y_pred.reshape((200,200,2))
#     image_list = []
#     image_list.append(X_test[i])
#     image_list.append(y_test[i])
#     image_list.append(np.argmax(y_pred, axis=-1))
#     display(image_list)

# Plays video and runs segmentation
cap = cv2.VideoCapture('D://Higher_cam_pos_Uni_Parks_3rd_test//flipped_testing_video_lowres.mp4')

# Comparison with tensorflow lite done with a image rather than video
# frame = cv2.imread("C://Workspaces\ARCproject//Image_postpro//Training folder//20200513-141759Tick82str-16.186993573536025.jpg")
# start_time = time.time()
# new_frame = cv2.resize(frame, (input_width, input_height))
# input_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB) / 255.0
# y_pred = model.predict(input_frame[np.newaxis])
# print("--- %s seconds ---" % (time.time() - start_time))

while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    new_frame = cv2.resize(frame, (input_width, input_height))
    input_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)/255.0
    cv2.imshow('Resized', new_frame)
    y_pred = model.predict(input_frame[np.newaxis])
    y_pred = y_pred.reshape((input_height, input_width,  2))
    pred_mask = np.argmax(y_pred, axis=-1).reshape(input_height, input_width, 1)

    # Sum all the columns in the mask to find out where the horizon seen by the camera is
    max_value = np.amax(np.sum(pred_mask[:, :, 0], axis=0))
    # Find what columns contain that value. The relative position of this column to the central one is how much you need to steer
    max_index = np.where(np.sum(pred_mask[:, :, 0], axis=0)==max_value)
    mean_max_index = int(round(np.mean(max_index)))

    pred_mask = np.append(pred_mask * 255 , np.ones((input_width, input_height,1)) * 0, axis=2 )
    pred_mask = np.append(pred_mask, np.ones((input_width, input_height, 1)) * 0, axis=2)

    # Central vertical black line as reference (in black)
    pred_mask[int(0.25*input_height):int(0.75*input_height), int(input_width/2), 0] = 255
    pred_mask[int(0.25*input_height):int(0.75*input_height), int(input_width/2), 1] = 255
    pred_mask[int(0.25*input_height):int(0.75*input_height), int(input_width/2), 2] = 255

    # Plot the position of the pixel column with more path detected
    pred_mask[int(0.1*input_height):int(0.9*input_height), mean_max_index, 0] = 0
    pred_mask[int(0.1*input_height):int(0.9*input_height), mean_max_index, 1] = 255
    pred_mask[int(0.1*input_height):int(0.9*input_height), mean_max_index, 2] = 0

    output = ((0.6 * new_frame) + (0.4 * pred_mask)).astype("uint8")
    # cv2.imshow('Mask', output)
    plt.imshow(output)
    plt.show()

