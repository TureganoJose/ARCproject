import glob, os
from PIL import Image


files = glob.iglob(os.path.join('D://', "*.jpg"))
for file in files:
    if os.path.isfile(file):
        im = Image.open(file)
        #im.rotate(180).show()
        im.rotate(180).save("D://Rotated//"+file[4:], "JPEG", subsampling=0, quality=100)

# X_Data = []
# Y_Data = []
# files = glob.iglob(os.path.join('D://Rotated//', "*.jpg"))
# for file in files:
#     if os.path.isfile(file):
#         # Original
#         #im1 = Image.open(file)
#         #im1.show()
#
#         #Resizing
#         #im = Image.open(file).resize((200, 66))
#         #im = Image.open(file)
#         #im_thumbnail = im.thumbnail((200, 66), Image.ANTIALIAS) # Keeps aspect ratio
#         #im_thumbnail.show()
#
#         im = Image.open(file).resize((200, 66))
#         #im.show()
#
#         # Transform to YUV array as specified in original paper (RGB normalised here, YUV still needs normalisation)
#         im_array = np.asarray(im)
#         #im_array_yuv = tf.image.rgb_to_yuv(im_array)
#         #image = scipy.misc.imread(file, mode='RGB') #scipy.misc.imresize(,[66, 200])
#         #scipy.misc.imshow(im)
#
#         # Extract steering angles
#         starting_pos = str.find(file, 'str')
#         ending_pos = str.find(file, '.jpg')
#         steering = float(file[starting_pos+3:ending_pos-1])
#
#         X_Data.append(im_array)
#         Y_Data.append(steering)
#
# np.save('X_Data', np.array(X_Data))
# np.save('Y_Data', np.array(Y_Data))