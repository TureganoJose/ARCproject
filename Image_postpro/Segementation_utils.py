import glob, os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import tensorflow as tf


# import labelme
# https://github.com/wkentaro/labelme/blob/master/README.md

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