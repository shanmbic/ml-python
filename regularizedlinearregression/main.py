import sys
import scipy.io, scipy.misc, scipy.optimize, scipy.special
from numpy import * 
import pylab

from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.mlab as mlab 

print "Reading data...."
mat = scipy.io.loadmat( "ex5data1.mat" )
X, y = mat['X'], mat['y']
X_val, y_val = mat['Xval'] , mat['yval']
X_test , y_test = mat['Xtest'], mat['ytest'] 
m,n=shape(X)

print "Plotting"

pyplot.scatter(X, y, c='r', marker='x', s=30)
pyplot.xlim([-55, 45])
pyplot.ylim([-0.5,45])
pyplot.xlabel('Change in water level')
pyplot.ylabel('Water flowing out of the dam')
pyplot.show()

theta=array([[1,1]]).T 


def train(X, y, lamda):
	theta = zeros( (shape(X)[1], 1) )
	result = scipy.optimize.fmin_cg( computeCost, fprime = computeGrad, x0 = theta,args = (X, y, lamda), maxiter = 200, disp = True, full_output = True )
	return result[1], result[0]

def computeCost(theta, X, y, lamda):
	theta = theta.reshape( shape(X)[1], 1 )
	m = shape( X )[0]
	first=X.dot(theta)-y
	left_term=first.T.dot( first ) / (2 * m)
	right_term = theta[1:].T.dot( theta[1:] ) * (lamda / (2*m))
	J = (left_term + right_term).flatten()[0]
	return J


def computeGrad(theta, X, y, lamda):
	theta= theta.reshape(shape(X)[1], 1)
	m = shape(X)[0]
	grad=X.dot(theta)-y
	grad=X.T.dot(grad)/m 
	grad[1:]	= grad[1:] + theta[1:] * lamda / m

	return grad.flatten()


def linearRegCostfunction(theta, X, y, lamda):
	cost=computeCost(theta, X, y, lamda)
	grad=computeGrad(theta, X, y, lamda)
	return cost , grad 

def learningCurve( X, y, X_val, y_val, lamda ):
	m = shape( X )[0]
	X = c_[ones((m, 1)), X]
	error_train = []
	error_val = []

	m_val = shape( X_val )[0]
	X_val = c_[ones((m_val, 1)), X_val]

	for i in range( 0, m ):
		cost, theta = train( X[0:i+1,:], y[0:i+1,:], lamda )

		error_train.append( computeCost( theta, X[0:i+1,:], y[0:i+1,:], lamda ) )
		error_val.append( computeCost( theta, X_val, y_val, lamda ) )

	error_train = array(error_train)
	error_val = array(error_val)

	# number of training examples
	temp = array([x for x in range(1, m+1)])

	pyplot.ylabel('Error')
	pyplot.xlabel('Number of training examples')
	pyplot.ylim([-2, 100])
	pyplot.xlim([0, 13])
	pyplot.plot( temp, error_train, color='b', linewidth=2, label='Train' )
	pyplot.plot( temp, error_val, color='g', linewidth=2, label='Cross Validation' )
	pyplot.legend()
	pyplot.show( block=True )
	return error_train, error_val


print linearRegCostfunction( theta, c_[ones((shape(X)[0], 1)), X  ] , y, 1.0 )

print "Training "

pyplot.scatter( X, y, marker='x', c='r', s=30, linewidth=2 )
pyplot.xlim([-55, 45])
pyplot.ylim([-5, 45])
pyplot.xlabel('Change in water level(x)')
pyplot.ylabel('Water flowing out of the dam(y)')

lamda = 0.0
X_bias = c_[ones(shape(X)), X]
cost, theta = train( X_bias, y, lamda )

pyplot.plot( X, X_bias.dot( theta ), linewidth=2 )
pyplot.show()

lamda = 0.0
print learningCurve( X, y, X_val, y_val, lamda )