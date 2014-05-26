from matplotlib import pyplot as plt 

def plot(X,y):
	m=len(y)
	xpos=[]
	xneg=[]
	for i in range(m):
		if y[i]==1.0:
			xpos.append([X[i][0], X[i][1]])
		else:
			xneg.append([X[i][0], X[i][1]])
	plt.figure(1)
	plt.axis([30,100,30,100])
	plt.plot([x[0] for x in xpos], [x[1] for x in xpos], linestyle='', marker='.', color='b', ms=8.0)
	plt.plot([x[0] for x in xneg], [x[1] for x in xneg], linestyle='', marker='+', color='r', ms=10.0)
	plt.text(85,95, 'Admitted:Red')
	plt.text(82,90, 'Not Admitted:Blue')
	plt.xlabel('Exam 1 score')
	plt.ylabel('Exam 2 score')
	plt.show()