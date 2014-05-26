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


def polyFeatures(X, p):
	out = copy(X)
	for i in range(1, p):
		out = c_[out, X**(i+1)]
	return out

def featureNormalize(data):
	mu = mean( data, axis=0 )
	data_norm = data - mu
	sigma = std( data_norm, axis=0, ddof=1 )
	data_norm = data_norm / sigma
	return data_norm, mu, sigma


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

def plotFit(min_x, max_x, mu, sigma, theta, p):
	x = arange( min_x - 15, max_x + 25, 0.05 )
	X_poly = polyFeatures( x, p )
	X_poly = (X_poly - mu) / sigma
	X_poly = c_[ ones((shape(x)[0], 1)), X_poly ]
	pyplot.plot( x, X_poly.dot(theta), linestyle='--', linewidth=3 )

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


def validationCurve( X, y, X_val, y_val ):
	lamda_vec = array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]).T
	error_train = []
	error_val = []

	for lamda in lamda_vec:
		cost, theta = train( X, y, lamda )

		error_train.append( computeCost( theta, X, y, lamda ) )
		error_val.append( computeCost( theta, X_val, y_val, lamda ) )

	error_train = array( error_train )
	error_val = array( error_val )

	pyplot.ylabel('Error')
	pyplot.xlabel('Lambda')
	pyplot.plot( lamda_vec, error_train, 'b', label='Train' )
	pyplot.plot( lamda_vec, error_val, 'g', label='Cross Validation' )
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

print "Feature mapping for polynomial regression"

p = 8
m, n = shape( X )

X_poly = polyFeatures( X, p )
X_poly, mu, sigma = featureNormalize( X_poly )
X_poly = c_[ones((m, 1)), X_poly]

X_poly_test = polyFeatures( X_test, p )
X_poly_test = X_poly_test - mu
X_poly_test = X_poly_test / sigma
X_poly_test = c_[ones(( shape(X_poly_test)[0], 1)), X_poly_test]

X_poly_val = polyFeatures( X_val, p )
X_poly_val = X_poly_val - mu
X_poly_val = X_poly_val / sigma
X_poly_val = c_[ones(( shape(X_poly_val)[0], 1)), X_poly_val]

print X_poly[0, :]

lamda = 0.0

cost, theta = train( X_poly, y, lamda )

pyplot.scatter( X, y, marker='x', c='r', s=30, linewidth=2 )
pyplot.xlim([-80, 80])
pyplot.ylim([-60, 40])
pyplot.xlabel('Change in water level(x)')
pyplot.ylabel('Water flowing out of the dam(y)')

pyplot.text( -15, 45, 'Lambda = %.1f' %lamda )
plotFit( min(X), max(X), mu, sigma, theta, p )

pyplot.show()

learningCurve( X_poly, y, X_poly_val, y_val, lamda )

print "Validation of the data"

validationCurve( X_poly, y, X_poly_val, y_val )