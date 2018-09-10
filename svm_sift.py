import cv2
import numpy as np
import gdal
import os
from sklearn.externals import joblib
from sklearn import preprocessing
import pickle



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


def read_rgb(image, x, y):
    i=round(x)
    j=round(y)
    color = (image[i, j])
    return color


# rgb file for generating svm training data
# filename = './image/haimen_10_rgb.tif'
# filename = './image/chenshan_10_rgb.tif'
# filename = './image/simulation/utah_area.tif'
filename = './image/meilan/l17_clip.tif'
tag = "MEILAN_"



img = cv2.imread(filename)
img_color = img.copy()
read_world_file(filename)
img= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# sift = cv2.xfeatures2d.SIFT_create()
sift = cv2.xfeatures2d.SIFT_create(nOctaveLayers=4, sigma=1, contrastThreshold=0.04, edgeThreshold=12) # modified for downsampling mostly ok
# sift = cv2.xfeatures2d.SIFT_create(nOctaveLayers=2, sigma=3, contrastThreshold=0.04, edgeThreshold=12) # meilan lake
# sift = cv2.xfeatures2d.SIFT_create(nOctaveLayers=3, sigma=3, contrastThreshold=0.04, edgeThreshold=15) # for simulation

kp,des = sift.detectAndCompute(img,None)
print('===image size==')
print(np.size(img,0))
print(np.size(img,1))
print('sift point count:'+str(len(kp)))
print('==des size==')
print (np.size(des[0]))

# img3 = cv2.drawKeypoints(img3,kp, outImage=img3)

index = []
svm_train_array = np.zeros((len(kp),134),dtype=np.float64)

point_id=0
# generate training data
for point,descriptor in zip(kp,des):
    bgr = read_rgb(img_color, point.pt[1], point.pt[0])
    prop_array = np.array([point.size, point.angle, point.response])
    temp_array = np.concatenate((prop_array, bgr, descriptor.ravel()),axis=0)
    svm_train_array[point_id,:] = temp_array

    temp = (point.pt[0],point.pt[1], point.size, point.angle, point.response, point.octave,
        point_id)

    index.append(temp)
    point_id += 1

# standarize data
# svm_train_array = preprocessing.scale(svm_train_array)
# write training data to file
output = open("./svm_data/svm_training_"+tag+os.path.basename(filename)+".pkl", "wb")
pickle.dump(svm_train_array, output)
print("write svm train data"+"./svm_data/svm_training_"+tag+os.path.basename(filename)+".pkl")
output.close()

# use svm classifier
from sklearn.externals import joblib

# load classifier
clf = joblib.load('./svm_data/classifier.pkl')
print(clf)

data = preprocessing.scale(svm_train_array)
print(np.size(data[0]))
pd = clf.predict(data)
# f_pd = open("./svm_data/svm_result_"+tag+os.path.basename(filename)+".pkl", "wb")
# pickle.dump(pd, f_pd)
# print("classified:"+str(np.size(pd)))
# # output = np.concatenate(svm_train_array, pd)
#
# print("svm_success")
# # np.savetxt('./svm_data/result.csv', output, delimiter = ',')
#
#
#
# f_pd_read = open("./svm_data/svm_result_"+tag+os.path.basename(filename)+".pkl", "rb")
# pd = pickle.load(f_pd_read)
# change the coord to world
Xcellsize,Ycellsize, upperleft_X, upperleft_Y = read_world_file(filename)

point_list_world = []

if Xcellsize != -1:
    for point_tuple, is_keypoint in zip(index, pd):
        point = list(point_tuple)
        point.append(is_keypoint)
        point[0] = upperleft_X + point[0] * Xcellsize
        point[1] = upperleft_Y + point[1] * Ycellsize
        point_list_world.append(point)

# write the keypoints to file
f = open("./svm_data/keypoints_"+tag+os.path.basename(filename)+".csv", "w")
f.writelines('x,y,size,angle,response,octave,point_id,is_keypoint\n')
if len(point_list_world) > 1:
    f.writelines( "%s\n" % str(item)[1:][:-1] for item in point_list_world)
else:
    f.writelines("%s\n" % str(item)[1:][:-1] for item in index)
f.close()


