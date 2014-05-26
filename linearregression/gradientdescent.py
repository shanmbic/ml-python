import numpy as np 
from computecost import computeCost

def gradientDescent(X,y,theta,alpha,num_iters):
	m=len(y)
	J_hist=[0 for i in range(num_iters)]
	for i in range(num_iters):
		htheta=np.dot(X,theta)
		workmatrix=np.subtract(htheta,y)
		sum1=0
		sum2=0
		for count in range(len(workmatrix)):
			sum1=sum1+workmatrix[count]*X[count][0]
			sum2=sum2+workmatrix[count]*X[count][1]
		theta0=theta[0] - (alpha/m)*sum1
		theta1=theta[1] - (alpha/m)*sum2
		theta[0]=theta0
		theta[1]=theta1
		J_hist[i]=computeCost(X,y,theta)
	return theta

