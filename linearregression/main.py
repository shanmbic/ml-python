import numpy as np 
import time
from matplotlib import pyplot as plt
import os
from computecost import computeCost
from gradientdescent import gradientDescent

"""Reading the machine learning data from file"""
data=np.loadtxt('data1.txt', dtype=float, delimiter=',')

"""Breaking the data into matrix X and Y"""
x=[]
y=[]
for i in data:
	x.append(i[0])
	y.append(i[1])

"""Mapping the data on plot"""
plt.xlabel('Profit in $10,000s')
plt.ylabel('Population of City in 10,000s')
plt.plot(x,y,'ro')
plt.axis([4, 24, -5, 25])
plt.show()

X=[]
for i in range(len(x)):
	X.append([1, x[i]])
theta=[0,0]

""""Basic settings for gradient descent"""	
m=len(y) 
iterations=1500 
alpha=0.01 

"""Computing the cost"""
print "Initial cost is %f" % computeCost(X,y,theta)
print "--------------------------------------------"
print "Running Gradient descent"
"""Running gradientdescent"""
start_time=time.time()
theta=gradientDescent(X,y,theta,alpha,iterations)
elapsed_time=time.time() - start_time
print "Theta calculated by gradient descent"
print theta
print "Total time elapsed for training the model is %s secs" % (str(elapsed_time))

print "--------------------------------------------"

predict1=np.dot([1,3.5], theta)
print "For population of 35000 we predict a profit of %f" % (predict1*10000)
predict2=np.dot([1,7], theta)
print "For population of 70000 we predict a profit of %f" % (predict2*10000)

