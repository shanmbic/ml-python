import re, csv
import nltk, nltk.stem.porter
import numpy as np 

def getVocabList():
	vocab = {}
	with open('vocab.txt', 'r') as file:
		reader = csv.reader( file, delimiter='\t' )
		for row in reader:
			vocab[row[1]] = int(row[0])

	return vocab

vocab_list=getVocabList()

def emailFeatures( word_indices ):
	features = [0 for i in range(len(vocab_list))]
	for index in word_indices:
		features[index-1] = 1
	return features



def process(contents):

	contents=contents.lower()

	word_indices = []

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

		if len(token) == 0:
			continue

		if token in vocab_list:
			word_indices.append( vocab_list[token] )



	return word_indices


