from emailprocess import *
import scipy.io
import pickle
import pdb 
import numpy as np

email_contents = ''

with open('email.txt' , 'r') as f:
		email_contents = f.read()

word_indices = process( email_contents )
pdb.set_trace()
features = emailFeatures( word_indices ).transpose()

linear_svm = pickle.load( open('linear_svm.svm' , 'rb'))

print linear_svm.predict(features)

