import numpy as np 
from computecostmulti import computeCost

def gradientDescent(X,y,theta,alpha,num_iter):
	m=len(y)
	J_hist=[0 for i in range(num_iter)]
	for i in range(num_iter):
		htheta=np.dot(X,theta)
		workmatrix=np.subtract(htheta,y)
		sum1=0
		sum2=0
		sum3=0
		for count in range(len(workmatrix)):
			sum1=sum1+workmatrix[count]*X[count][0]
			sum2=sum2+workmatrix[count]*X[count][1]
			sum3=sum3+workmatrix[count]*X[count][2]
		theta0=theta[0] - (alpha/m)*sum1
		theta1=theta[1] - (alpha/m)*sum2
		theta2=theta[2] - (alpha/m)*sum3
		theta[0]=theta0
		theta[1]=theta1
		theta[2]=theta2
		J_hist[i]=computeCost(X,y,theta)
	return theta