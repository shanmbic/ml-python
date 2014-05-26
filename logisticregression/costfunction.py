import numpy as np
def sigmoid(x):
	ans=1.0/(1.0 + np.exp(-1.0*x))
	return ans

def costFunc(X,y,theta):
	m=len(y)
	J=0
	s=0
	grad=[0,0,0]
	htheta=np.dot(X,theta)
	for i in range(len(htheta)):
		htheta[i]=sigmoid(htheta[i])
	for i in range(m):
		s=s+((-1.0)*y[i]*np.log(htheta[i]) - (1-y[i])*np.log(1-htheta[i]))
	J=s/m
	workmatrix=np.subtract(htheta,y)
	for i in range(3):
		s1=0
		for j in range(m):
			s1=s1+workmatrix[j]*X[j][i]
		grad[i]=s1/m
	return J, grad
