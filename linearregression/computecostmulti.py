import numpy as np 
def computeCost(X,y,theta):
	J=0
	m=len(y)
	predictions=np.subtract(np.dot(X, theta),y)
	x=np.dot(predictions, np.matrix.transpose(predictions))
	J = x/(2*m)
	return J