from datetime import datetime
import os
import struct
import numpy as np
import scipy.misc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import DistanceMetric
"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""

def read(dataset, path):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)
    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows*cols)


    return lbl,img
# def show(image):
#     """
#     Render a given numpy.uint8 2D array of pixel data.
#     """

#     fig = pyplot.figure()
#     ax = fig.add_subplot(1,1,1)
#     imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
#     imgplot.set_interpolation('nearest')
#     ax.xaxis.set_ticks_position('top')
#     ax.yaxis.set_ticks_position('left')
#     pyplot.show()

label_train,image_train=read("training","F:\\sem8\\cs771a\\ass2\\")
# for label_train,image_train in get_imge:
#     show(image)
dist=DistanceMetric.get_metric('euclidean')         #for different metric
knn=KNeighborsClassifier(n_neighbors=3,
                        metric='manhattan')

knn.fit(image_train,label_train)

label_test,image_test=read("testing","F:\\sem8\\cs771a\\ass2\\")
prediction=np.array(knn.predict(image_test))
wrong_pred=(prediction!=label_test).sum()
f=open('pred_num.txt','a')
f.write(str(datetime.now()))
f.write('\n')
f.write(str(prediction))
f.write('\n')
f.write('no. of wrong prediction= %d out of 10000' %(wrong_pred))
f.write('\n')
f.close()  
