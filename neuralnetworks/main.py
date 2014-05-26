import sys
import scipy.io, scipy.misc, scipy.optimize, scipy.special
from numpy import * 
import pylab

from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.mlab as mlab 

print "Reading data...."
mat = scipy.io.loadmat( "ex4data1.mat" )
X, y = mat['X'], mat['y']
m,n=shape(X)


input_layer_size=400
hidden_layer_size=25
num_labels=10

print "Displaying data"

def computeCost( nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamda, yk = None, X_bias = None ):
	m, n = shape( X )
	theta1, theta2 = paramunroll( nn_params, input_layer_size, hidden_layer_size, num_labels )
	a1, a2, a3, z2, z3 = feedforward( theta1, theta2, X, X_bias )

# calculating cost
	if yk is None:
		yk = recodelabel( y, num_labels )
	assert shape(yk) == shape(a3), "Error, shape of recoded y is different from a3"

	term1 = -yk * log( a3 )
	term2 = (1 - yk) * log( 1 - a3 )
	left_term = sum(term1 - term2) / m
	right_term = sum(theta1[:,1:] ** 2) + sum(theta2[:,1:] ** 2)

	return left_term + right_term * lamda / (2 * m)

def predict( X, theta1, theta2 ):
	a1 = r_[ones((1, 1)), X.reshape( shape(X)[0], 1 )]
	z2 = sigmoid( theta1.dot( a1 ))
	z2 = r_[ones((1, 1)), z2]
	z3 = sigmoid(theta2.dot( z2 ))
	return argmax(z3) + 1

def displaydata(X, theta1=None, theta2=None):
	width=20
	rows,cols=10,10
	out = zeros(( width * rows, width*cols ))
	rand_indices = random.permutation( 5000 )[0:rows * cols]

	counter = 0
	for y in range(0, rows):
		for x in range(0, cols):
			start_x = x * width
			start_y = y * width
			out[start_x:start_x+width, start_y:start_y+width] = X[rand_indices[counter]].reshape(width, width).T
			counter += 1

	img = scipy.misc.toimage( out )
	figure = pyplot.figure()
	axes = figure.add_subplot(111)
	axes.imshow( img )

	if theta1 is not None and theta2 is not None:
		result_matrix = []
		X_biased = c_[ ones( shape(X)[0] ), X ]
		for idx in rand_indices:
				result = predict( X[idx], theta1, theta2 )
				result_matrix.append( result )
		result_matrix = array( result_matrix ).reshape( rows, cols ).transpose()
		print result_matrix

	pyplot.show( )


displaydata(X)

print "Reading predefined theta"

theta=scipy.io.loadmat('ex4weights.mat')
theta1, theta2=theta['Theta1'] , theta['Theta2']

nn_params=r_[theta1.T.flatten(), theta2.T.flatten()]

def mod(length, divisor):
	dividend=array([x for x in range(1,length+1)])
	divisor=array([divisor for x in range(1,length+1)])
	return mod( dividend, divisor ).reshape(1, length )

def sigmoid(x):
	return scipy.special.expit(x)

def feedforward( theta1, theta2, X, X_bias = None ):
	one_rows = ones((1, shape(X)[0] ))
	a1 = r_[one_rows, X.T] if X_bias is None else X_bias
	z2 = theta1.dot( a1 )
	a2 = sigmoid(z2)
	a2 = r_[one_rows, a2]
	z3 = theta2.dot( a2 )
	a3 = sigmoid( z3 )

	return (a1, a2, a3, z2, z3)

def recodelabel( y, k ):
	m = shape(y)[0]
	out = zeros( ( k, m ) )
	for i in range(0, m):
		out[y[i]-1, i] = 1

	return out

def sigmoidGradient( z ):
	sig = sigmoid(z)
	return sig * (1 - sig)

def paramunroll(nn_params, input_layer_size, hidden_layer_size, num_labels):
	theta1_elems = ( input_layer_size + 1 ) * hidden_layer_size
	theta1_size = ( input_layer_size + 1, hidden_layer_size )
	theta2_size = ( hidden_layer_size + 1, num_labels )
	theta1 = nn_params[:theta1_elems].T.reshape( theta1_size ).T	
	theta2 = nn_params[theta1_elems:].T.reshape( theta2_size ).T

	return (theta1, theta2)


def nnCostfunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, xlamda):
	m,n=shape(X)
	theta1, theta2=paramunroll(nn_params, input_layer_size, hidden_layer_size, num_labels)
	a1, a2, a3, z2, z3=feedforward(theta1, theta2, X)

	yk=recodelabel(y, num_labels)

	assert shape(yk)==shape(a3) , "Error, shape of recoded y is different from a3"
	firstterm = (-1)*yk*log(a3)
	secondterm=(1-yk)*log(1-a3)
	left_term=(firstterm-secondterm)/m
	right_term=sum(theta1[:,1:] ** 2) + sum(theta2[:,1:] ** 2)
	right_term = right_term * xlamda / (2 * m)
	cost = left_term + right_term

	sigma3 = a3 - yk
	sigma2 = theta2.T.dot( sigma3 ) * sigmoidGradient( r_[ ones((1, m)), z2 ] )
	sigma2 = sigma2[1:,:]

	accum1 = sigma2.dot( a1.T ) / m
	accum2 = sigma3.dot( a2.T ) / m

	accum1[:,1:] = accum1[:,1:] + (theta1[:,1:] * xlamda / m)
	accum2[:,1:] = accum2[:,1:] + (theta2[:,1:] * xlamda / m)
	gradient = array([accum1.T.reshape(-1).tolist() + accum2.T.reshape(-1).tolist()]).T

	return (cost, gradient)

def randInitializeWeights(L_in, L_out):
	e = 0.12
	w = random.random((L_out, L_in + 1)) * 2 * e - e
	return w

def debugInitializeWeights(fan_out, fan_in):
	num_elements = fan_out * (1+fan_in)
	w = array([sin(x) / 10 for x in range(1, num_elements+1)])
	w = w.reshape( 1+fan_in, fan_out ).T
	return w


def computeGradient( nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, xlambda, yk = None, X_bias = None ):
	m, n = shape( X )
	theta1, theta2 = paramunroll( nn_params, input_layer_size, hidden_layer_size, num_labels )
	a1, a2, a3, z2, z3 = feedforward( theta1, theta2, X, X_bias )

	# back propagate
	if yk is None:
		yk = recodeLabel( y, num_labels )
	assert shape(yk) == shape(a3), "Error, shape of recoded y is different from a3"

	sigma3 = a3 - yk
	sigma2 = theta2.T.dot( sigma3 ) * sigmoidGradient( r_[ones((1, m)), z2 ] )
	sigma2 = sigma2[1:,:]
	accum1 = sigma2.dot( a1.T ) / m
	accum2 = sigma3.dot( a2.T ) / m

	accum1[:,1:] = accum1[:,1:] + (theta1[:,1:] * xlambda / m)
	accum2[:,1:] = accum2[:,1:] + (theta2[:,1:] * xlambda / m)
	accum = array([accum1.T.reshape(-1).tolist() + accum2.T.reshape(-1).tolist()]).T
	return ndarray.flatten(accum)



def computeNumericalGradient( theta, input_layer_size, hidden_layer_size, num_labels, X, y, xlambda ):
	numgrad = zeros( shape(theta) )
	perturb = zeros( shape(theta) ) #38 x 1
	e = 1e-4

	num_elements = shape(theta)[0]
	yk = recodeLabel( y, num_labels )

	for p in range(0, num_elements) :
		perturb[p] = e
		loss1 = computeCost( theta - perturb, input_layer_size, hidden_layer_size, num_labels, X, y, xlambda, yk )
		loss2 = computeCost( theta + perturb, input_layer_size, hidden_layer_size, num_labels, X, y, xlambda, yk )
		numgrad[p] = (loss2 - loss1) / (2 * e)
		perturb[p] = 0

	return numgrad

def checkNNGradients(xlambda):
	if xlambda==None:
		xlambda=0
	input_layer_size = 3;
	hidden_layer_size = 5;
	num_labels = 3;
	m = 5;

	theta1 = debugInitializeWeights( hidden_layer_size, input_layer_size )
	theta2 = debugInitializeWeights( num_labels, hidden_layer_size )

	x=debugInitializeWeights(m, input_layer_size-1)
	y = 1 + mod( m, num_labels )

	nn_params 	= array([theta1.T.reshape(-1).tolist() + theta2.T.reshape(-1).tolist()]).T
	gradient 	= nnCostFunction( nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamda )[1]
	numgrad = computeNumericalGradient( nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamda )
	diff = linalg.norm( numgrad - gradient ) / (linalg.norm( numgrad + gradient ))
	print diff







print "Feedforward using neural networks"

xlambda=0

print "Cost at parameters (loaded from ex4weights). \n(this value should be about 0.287629)" 
print nnCostfunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, xlambda)

print "Checking cost function with regularization"
xlambda=1

print "Cost at parameters (loaded from ex4weights). \n(this value should be about 0.383770)" 
print nnCostfunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, xlambda)

g=sigmoidGradient([1,-0.5,0,0.5,1])

print "Sigmoid Gradient evaluated at 1 -0.5 0 0.5 1", g

print "Initializing neural network parameters"

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)
initial_nn_parameters=r_[initial_Theta1.T.flatten(), initial_Theta2.T.flatten()]

print "Checking back propogation with regularization"
xlambda=3
debug_J  = nnCostfunction(initial_nn_parameters, input_layer_size, hidden_layer_size, num_labels, X, y, xlambda)

print "Cost at parameters (loaded from ex4weights). \n(this value should be about 0.576051)" , debug_J


xlambda=1

theta1 = randInitializeWeights( 400, 25 )
theta2 = randInitializeWeights( 25, 10 )

yk = recodelabel( y, num_labels )
unraveled = r_[theta1.T.flatten(), theta2.T.flatten()]

X_bias = r_[ ones((1, shape(X)[0] )), X.T]

result = scipy.optimize.fmin_cg( computeCost, fprime=computeGradient, x0=unraveled, \
args=(input_layer_size, hidden_layer_size, num_labels, X, y, xlambda, yk, X_bias), \
maxiter=50, disp=True, full_output=True )

print result[1]
theta1, theta2 = paramunroll( result[0], input_layer_size, hidden_layer_size, num_labels )

displaydata( X, theta1, theta2 )

counter = 0
for i in range(0, m):
	prediction = predict( X[i], theta1, theta2 )
	actual = y[i]
	if( prediction == actual ):
		counter+=1
print counter * 100 / m

