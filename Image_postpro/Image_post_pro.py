import glob, os
from PIL import Image
from shutil import copyfile, rmtree, copy2
import cv2
import numpy as np


## Flip video from camera
# cap = cv2.VideoCapture('D://Higher_cam_pos_Uni_Parks_3rd_test//testing_video_lowres.mkv')
# cv2.namedWindow("CurrentFrame", cv2.WINDOW_AUTOSIZE)
# 
# fourcc = cv2.VideoWriter_fourcc(*'h264') # is a MP4 codec
# 
# writer = cv2.VideoWriter('D://Higher_cam_pos_Uni_Parks_3rd_test//flipped_testing_video_lowres2.mp4', fourcc, 10, (1024, 768), 1)
# 
# iframe=1
# while(cap.isOpened()):
#     # Capture frame-by-frame
#     ret, frame = cap.read()
# 
#     # Our operations on the frame come here
#     new_frame = np.flip(frame)
#     new_frame = cv2.cvtColor(new_frame, cv2.COLOR_RGB2BGR)
# 
#     # Display the resulting frame
#     # cv2.imshow('video', new_frame)
#     writer.write(new_frame)
#     # To export to gift
#     #cv2.imwrite("image_"+str(iframe)+".jpg", new_frame)
#     #iframe+=1
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()


## Rotates images and save it as jpg
files = glob.iglob(os.path.join('/media/pi/UUI/Late_evening_4th_test', "*.jpg"))
for file in files:
    if os.path.isfile(file):
        im = Image.open(file)
        #im.rotate(180).show()
        im.rotate(180).save("/media/pi/UUI/Rotated_Late_evening_4th_test/"+file[38:], "JPEG", subsampling=0, quality=100)

## Once they are labeled, it copies json files, masks and image to a new folder
# files = glob.iglob(os.path.join('D://Rotated_Higher_cam_pos_Uni_Parks_3rd_test', "*.json"))
# for file in files:
#     if os.path.isfile(file):
#         start_pos = file.find("\\2")
#         end_pos = file.find(".json")
#         base_filename = file[start_pos+1:end_pos]
#         copy2(file[:-4]+"jpg", "D://Training folder") # copies original picture to training folder
#         base_folder = base_filename.replace(".","_")
#         path = file[:start_pos]+"\\"
#         os.system("labelme_json_to_dataset " + file) # create dataset
#         # Renames label file created by labelme
#         os.rename( path + base_folder +"_json"+ "\\label.png" , path + base_folder + "_json\\" + base_filename + "_json.png" )
#         # Copy label files to the root folder
#         copyfile( path + base_folder + "_json\\" + base_filename + "_json.png",  path + base_filename + "_json.png")
#         # and removes the folder created by labelme
#         rmtree( path + base_folder + "_json\\" )
#         copy2(path + base_filename + "_json.png", "D://Training folder") # Copies png label file to training folder
#         copy2(file, "D://Training folder") # Copies json file to training folder


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