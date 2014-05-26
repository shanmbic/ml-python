import numpy as np 

def featureNormalize(X):
	mean=[]
	std=[]
	x1=[i[1] for i in X]
	x2=[i[2] for i in X]
	mean.append(np.mean(x1))
	mean.append(np.mean(x2))
	std.append(np.std(x1))
	std.append(np.std(x2))
	print "The mean and standard deviation of first feature is %f, %f" % (mean[0], std[0])
	print "The mean and standard deviation of second feature is %f, %f" % (mean[1], std[1])
	for i in X:
		i[1]=(i[1]-mean[0])/std[0]
		i[2]=(i[2]-mean[1])/std[1]
	return X