import numpy as np 
from matplotlib import pyplot as plt 
import time
import os 
from featurenormalize import featureNormalize
from gradientdescentmulti import gradientDescent
from computecostmulti import computeCost

"""Reading the machine learning data from dataset"""
data=np.loadtxt('data2.txt', dtype=float, delimiter="," )

"""Creating the matrix X and y"""
X=[]
y=[]
for i in data:
	X.append([i[0],i[1]])
	y.append(i[2])
m=len(y)

"""First 10 examples of the dataset"""
print "First 10 examples from the dataset"
for i in range(10):
	print "X=%f,%f , y=%f" % (X[i][0], X[i][1], y[i] )
print "--------------------------------------------"

for i in X:
	i.insert(0,1)

"""Feature Normalization"""
print "Performing Feature Normalization on the dataset"
X=featureNormalize(X)
print ""
print "The matrix after feature normalization"
for i in range(10):
	print "%f,%f,%f" % (X[i][0], X[i][1], X[i][2])
print "---------------------------------------------"
print "Performing Gradient descent"
alpha=0.03
iterations=100
theta=[0,0,0]
print ""
print "The initial cost is %f" % computeCost(X,y,theta)

theta=gradientDescent(X,y,theta,alpha,iterations)
print "Theta found by gradient descent is \n theta1=%f \n theta2=%f \n theta3=%f" % (theta[0],theta[1], theta[2])