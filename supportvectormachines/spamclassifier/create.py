yimport pickle
import os , sys 
import emailprocess
import scipy.io 
import numpy as np 

mat = scipy.io.loadmat('spamTrain.mat')
X, y = mat['X'] , mat['y'] 


X = list(X)

print len(X)
for filename in os.listdir('/home/shantanu/ml-python/supportvectormachines/spamclassifier/newdata/emails'):
	filen = '/home/shantanu/ml-python/supportvectormachines/spamclassifier/newdata/emails/' + filename
	os.chmod(filen, 0o777)
	with open(filen, 'rw') as f:
		contents = f.read()
		word_indices = emailprocess.process( contents )
		features = emailprocess.emailFeatures( word_indices )

		#pdb.set_trace()
		X.append([features])
		y = np.vstack((y, 1))

print len(X)

mat['X'] = X
mat['y'] = y

scipy.io.savemat('spamTrain.mat' , mat)
 