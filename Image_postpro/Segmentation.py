#https://divamgupta.com/image-segmentation/2019/06/06/deep-learning-semantic-segmentation-keras.html

# import labelme
# https://github.com/wkentaro/labelme/blob/master/README.md


import glob, os
from shutil import copyfile, rmtree
import imgaug as ia
import imgaug.augmenters as iaa
from tensorflow.keras.layers import Dense, Flatten, Conv2D, DepthwiseConv2D, MaxPool2D, Input, Dropout, UpSampling2D, concatenate



# files = glob.iglob(os.path.join('D://Rotated', "*.json"))
# for file in files:
#     if os.path.isfile(file):
#         start_pos = file.find("\\2")
#         end_pos = file.find(".json")
#         base_filename = file[start_pos+1:end_pos]
#         base_folder = base_filename.replace(".","_")
#         path = file[:start_pos]+"\\"
#         os.system("labelme_json_to_dataset " + file) # create dataset
#         os.rename( path + base_folder +"_json"+ "\\label.png" , path + base_folder + "_json\\" + base_filename + "_json.png" )
#         copyfile( path + base_folder + "_json\\" + base_filename + "_json.png",  path + base_filename + "_json.png")
#         rmtree( path + base_folder + "_json\\" )



seq = iaa.Sequential([
    iaa.Crop(px=(0, 16)),  # crop images from each side by 0 to 16px (randomly chosen)
    iaa.Fliplr(0.5),  # horizontally flip 50% of the images
    iaa.GaussianBlur(sigma=(0, 3.0))  # blur images with a sigma of 0 to 3.0
])


def augment_seg(img, seg):
    aug_det = seq.to_deterministic()
    image_aug = aug_det.augment_image(img)

    segmap = ia.SegmentationMapOnImage(seg, nb_classes=np.max(seg) + 1, shape=img.shape)
    segmap_aug = aug_det.augment_segmentation_maps(segmap)
    segmap_aug = segmap_aug.get_arr_int()

    return image_aug, segmap_aug


input_height = 768
input_width = 1024
img_input = Input(shape=(input_height,input_width , 3 ))

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


