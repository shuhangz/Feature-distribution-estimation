import cv2
import numpy as np
import gdal
import os

# img_tiff = gdal.Open('./image/resample.tif')#
# wkt = img_tiff.GetProjection()


def read_world_file(filename):
    path_list = os.path.splitext(filename)
    ext = path_list[1]
    world_file_ext = ''
    if ext=='.jpg':
        world_file_ext = '.jgw'
    elif ext=='.png':
        world_file_ext = '.pgw'
    elif ext =='.tif':
        world_file_ext = '.tfw'
    world_file_name = (path_list[0]+world_file_ext)
    if os.path.exists(world_file_name):
        with open(world_file_name, 'r') as world_file:
            XCellSize = float(world_file.readline())
            dont_care = world_file.readline()  # should be 0
            dont_care = world_file.readline()  # should be 0
            YCellSize = float(world_file.readline())
            upperleft_X = float(world_file.readline())
            upperleft_Y = float(world_file.readline())
        return XCellSize, YCellSize, upperleft_X, upperleft_Y
    else:
        return -1,-1,-1,-1



# filename = './image/2_5m_xiamen.png'
# filename = './image/chenshan_gray_clip_resample.png'
# filename = './image/haimen_gray_resample1.png'
# filename = './image/xiamen_gray.png'
# filename = './image/xm_ortho2.tif'
# filename = './image/Clip.tif'
# filename = './image/haimen_ori.png'
filename = './image/jinhai/l17_a1.tif'
tag = "JH_"

# rgb file for gerenate svm training data


img = cv2.imread(filename)
read_world_file(filename)
img= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# sift = cv2.xfeatures2d.SIFT_create()
sift = cv2.xfeatures2d.SIFT_create(nOctaveLayers=1, sigma=.6, contrastThreshold=0.04, edgeThreshold=12) # modified for downsampling mostly ok4
# sift = cv2.xfeatures2d.SIFT_create(nOctaveLayers=1, sigma=0.5, contrastThreshold=0.02, edgeThreshold=8)
kp = sift.detect(img,None)
print(len(kp))
img3 = img.copy()
img3 = cv2.drawKeypoints(img,kp, outImage=img3)

index = []
for point in kp:
    temp = (point.pt[0],point.pt[1], point.size, point.angle, point.response, point.octave,
        point.class_id)
    index.append(temp)

cv2.imshow('keypoints',img3)

# change the coord to world
Xcellsize,Ycellsize, upperleft_X, upperleft_Y = read_world_file(filename)

point_list_world = []

if Xcellsize != -1:
    for point in index:
        point = list(point)
        point[0] = upperleft_X + point[0] * Xcellsize
        point[1] = upperleft_Y + point[1] * Ycellsize
        point_list_world.append(point)

# write the keypoints to file
f = open("keypoints_"+tag+os.path.basename(filename)+".csv", "w")
f.writelines('x,y,size,angle,response,octave,class_id\n')
if len(point_list_world) > 1:
    f.writelines( "%s\n" % str(item)[1:][:-1] for item in point_list_world)
else:
    f.writelines("%s\n" % str(item)[1:][:-1] for item in index)
f.close()

cv2.waitKey(0)
