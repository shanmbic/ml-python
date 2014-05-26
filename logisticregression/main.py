import numpy as np 
from plotdata import plot
from costfunction import costFunc
import scipy.optimize as op
data=np.loadtxt('data1.txt', dtype=float, delimiter=',')
X=[]
y=[]
for i in data:
	X.append([i[0], i[1]])
	y.append(i[2])
plot(X,y)

for i in X: 
	i.insert(0,1)
initial_theta=[0,0,0]
cost, grad = costFunc(X,y,initial_theta)
print "The initial cost at theta(zeros) %f \n" % cost
print "Gradient at theta(zeros) \n theta0=%f \n theta1=%f \n theta2=%f" % (grad[0],grad[1],grad[2]) 

print "----------------------------------------------------"
print "Training the model with optimization algorithm"
print ""

result = op.minimize(fun = costFunc, x0 = initial_theta, args = (X, y), method = 'TNC',jac = grad)
print result 