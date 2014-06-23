import os , sys 
import emailprocess
import scipy.io 
import nltk, nltk.stem.porter
import re

vocab=[]
mainlist={}

"""
with open('vocab.txt','rw') as f:
	for lin in f.readlines():
		vocab.append(lin)
"""

for filename in os.listdir('/home/shantanu/ml-python/supportvectormachines/spamclassifier/newdata/emails')[:1000]:
	filen = '/home/shantanu/ml-python/supportvectormachines/spamclassifier/newdata/emails/' + filename
	os.chmod(filen, 0o777)
	with open(filen, 'rw') as f:
		contents = f.read()

		contents=contents.lower()
		contents = re.sub('<[^<>]',' ', contents)
		contents = re.sub( '[0-9]+' , 'number', contents)
		contents = re.sub( '[$]+' , 'dollar' , contents)
		contents = re.sub('(http|https)://[^/s]*' , 'httpaddr' , contents)
		contents = re.sub('[^\s]+@[^\s]+' , 'emailaddr' , contents)


		stemmer = nltk.stem.porter.PorterStemmer()
		tokens = re.split( '[ ' + re.escape("@$/#.-:&*+=[]?!(){},'\">_<;%") + ']' , contents )

		for token in tokens:
			token = re.sub( '[^a-zA-Z0-9]', '', token )
			token = stemmer.stem( token.strip() )

			if len(token) < 10 : 
				if token in vocab:
					mainlist[token]=mainlist[token]+1
				else:
					mainlist[token]=0


threshold = ( min(mainlist.values()) + max(mainlist.values()) )/ 2

for word in mainlist:
	if mainlist[word] > threshold:
		if word in vocab:
			continue

		else:
			vocab.append(word)


print threshold
print mainlist
print len(vocab)




