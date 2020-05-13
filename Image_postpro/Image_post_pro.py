import glob, os
from PIL import Image



'''Copy captures and move them to an external drive'''
files = glob.iglob(os.path.join('D://', "*.jpg"))
for file in files:
    if os.path.isfile(file):
        im = Image.open(file)
        #im.rotate(180).show()
        im.rotate(180).save("D://Rotated//"+file[4:], "JPEG", subsampling=0, quality=100)