import scipy.io 
import pickle
from sklearn import svm
import numpy as np

mat = scipy.io.loadmat('spamTrain.mat')
X, y = mat['X'] , mat['y']

linear_svm = svm.SVC(C=0.1, kernel='linear')

linear_svm.fit(X, y.ravel())

pickle.dump( linear_svm, open("linear_svm.svm", "wb") )

print "Training data performance"

print "Shape of training data %d, %d" % (np.shape(X)) 

predictions = linear_svm.predict(X)
predictions = predictions.reshape( np.shape(predictions)[0], 1 )

print "Accuracy : %f" % (( predictions == y ).mean() * 100.0)

print "Test data performance"

mat = scipy.io.loadmat('spamTest.mat')
X, y = mat['Xtest'] , mat['ytest']

predictions = linear_svm.predict(X)
predictions = predictions.reshape( np.shape(predictions)[0], 1 )

print "Accuracy : %f" % (( predictions == y ).mean() * 100.0)
