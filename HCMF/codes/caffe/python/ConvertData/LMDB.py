## Transfer cifar10 lmdb data to mat
import sys
import lmdb
import numpy as np
from array import array
import scipy.io as sio
import os
if os.path.exists('../caffe'):
    sys.path.append('..')
else:
    print 'Error : caffe(pycaffe) could not be found'
import caffe
from caffe.proto import caffe_pb2

#Train_or_Test = True
Train_or_Test = False
cifar10_train_data = '../../examples/cifar10/cifar10_train_lmdb'
cifar10_test_data  = '../../examples/cifar10/cifar10_test_lmdb'
if Train_or_Test:
    data_path = cifar10_train_data
    save_path = '../../examples/cifar10/cifar10_train_lmdb.mat'
else:
    data_path = cifar10_test_data
    save_path = '../../examples/cifar10/cifar10_test_lmdb.mat'

print 'LMDB DATA PATH : %s\n' %data_path 
lmdb_env = lmdb.open(data_path)
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
datum = caffe_pb2.Datum()

count = 0
for key, value in lmdb_cursor:
    datum.ParseFromString(value)

    label = datum.label
    data = caffe.io.datum_to_array(datum)

    #CxHxW to HxWxC in cv2
    image = np.transpose(data, (1,2,0))
    count = count + 1
    shape = image.shape
print 'Count the number of data are %d\n' %count

save_image = np.zeros((count,shape[0],shape[1],shape[2]), dtype=np.uint8)
save_label = np.zeros(count, dtype=np.uint8)

count = 0
for key, value in lmdb_cursor:
    datum.ParseFromString(value)

    save_label[count] = datum.label
    data = caffe.io.datum_to_array(datum)

    #CxHxW to HxWxC in cv2
    save_image[count,:,:,:] = np.transpose(data, (1,2,0))
    count = count + 1

sio.savemat(save_path,{'image':save_image,'label':save_label})
print 'Convert Done !\n Saved in %s \n'  %save_path
