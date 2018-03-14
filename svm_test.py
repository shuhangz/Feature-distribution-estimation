from sklearn import datasets
from sklearn import svm
from sklearn import preprocessing
import numpy as np
import pickle

# class_data = np.loadtxt("./svm_data/chenshan_class.txt",delimiter=",")
# class_data = np.transpose(class_data)
# class_data = np.delete(class_data, 0, axis=0)
# class_data = class_data.astype(dtype=int)
# class_data = class_data.flatten()
#
# print(np.size(class_data))
#
# svm_train_path = './svm_data/svm_training_10gsd_rgb_chenshan_10_rgb.tif.pkl'
# f = open(svm_train_path, 'rb')
# svm_train = pickle.load(f)
#
# iris = datasets.load_iris()
# set = iris.target
#
# print(svm_train.shape)
# print(class_data.shape)
# svc = svm.SVC(kernel='linear')
# svc.fit(svm_train, class_data)
# # print(svc.predict([[ 5.0,  3.6,  1.3,  0.25]]))



# use svm classifier
from sklearn.externals import joblib
# save trained svm
# joblib.dump(svc, 'ssvvc.pkl')
clf = joblib.load('./svm_data/classifier.pkl')
print(clf)
f = open('./svm_data/svm_training_10gsd_rgb_resample.tif.pkl','rb')
data = pickle.load(f)
# standarize data
data = preprocessing.scale(data)
print(np.size(data[0]))
pd = clf.predict(data)
print(np.size(pd))
point_id = np.arange(np.size(pd))
res = np.concatenate((point_id, pd)).reshape([2,np.size(pd)], order='C')
print(np.size(res[0]))
res = res.astype(int)
np.savetxt('./svm_data/result.csv', res.transpose(), delimiter = ',')


pass