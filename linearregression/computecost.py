import numpy as np

def computeCost(X,y,theta):
	J=0
	m=len(y)
	predictions=np.dot(X, theta)
	sqrerrors=[x*x for x in np.subtract(predictions,y)]
	J = sum(sqrerrors) / (2*m)
	return J
