"""This is a sample file for hw3. 
It contains the functions that should be submitted,
except all it does is output a random value.
- Dr. Licato"""

import string
import random
from gensim.parsing.preprocessing import remove_stopwords
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
import gensim.models.keyedvectors as word2vec
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import cosine as cosDist
from sklearn.model_selection import train_test_split
import numpy as np
import json
from numpy.linalg import norm

model = None
linregModel = None

# updates the global model to be the keyed vectors
def setModel(Model):
	path = get_tmpfile("word2vec.model")
	global model
	Model = word2vec.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary = True)
	model = Model
	model.save("word2vec.model")

# gets the vectors for words based on the Google News vectors (given in slides)
def getVector(w):
	global model
	if w in model:
		return model[w]
	else:
		return np.zeros(300)

# returns the index of the most plagiarized sentence
def findPlagiarism(sentences, target):
	global model
	scores = [] # array to score the highest similarity

	# remove stop words and split into an array of words
	target = remove_stopwords(target)
	targetWords = target.split()
	targetArray = np.zeros(300)
	# for each word in the array get the vector and add it to the target array
	for word in targetWords:
		targetArray += getVector(word)
	# for each sentence, break it down into words, and get the word vectors
	for sentence in sentences:
		filtered = remove_stopwords(sentence)
		filteredWords = filtered.split()
		sentenceArray = np.zeros(300)
		for word in filteredWords:
			sentenceArray += getVector(word)
		# append the score of the sentence to the scores arracy
		scores.append(np.dot(sentenceArray, targetArray) / (norm(sentenceArray)) * (norm(targetArray)))

	plagiarized = np.argmax(scores)
	return plagiarized

def classifySubreddit_train(trainFile):
	global linregModel
	newFile = [json.loads(x) for x in open(trainFile, 'r', encoding="utf-8")]
	subredditArray = []
	reviewArray = []
	# for each review in the file, partition the sections into their own arrays
	for line in newFile:
		reviewArray.append(str(line['body']))
		subredditArray.append(str(line['subreddit']))

	# for each review, break it into words
	bodyVec = []
	for review in reviewArray:
		review = review.replace(string.punctuation, "")
		bodyWords = review.lower().split()
		bodyVec.append(bodyWords)
	bodyArray = []
	# for each review, get the word vecors of each word
	for review in bodyVec:
		arr = np.zeros(300)
		for word in review:
			arr += getVector(word)
		bodyArray.append(arr)

	# make a logreg model and fit it
	linregModel = LogisticRegression(max_iter=1000000000)
	linregModel.fit(bodyArray, subredditArray)

	pass

def classifySubreddit_test(text):
	global linregModel

	# split the test text into words
	text = text.replace(string.punctuation, "")
	review = text.lower().split()

	# get the word vectors for the array
	textArray = np.zeros(300)
	for word in review:
		textArray += getVector(word)

	# make a prediction based on the vector array
	prediction = linregModel.predict([textArray])
	# set the return to be the first element of the prediction array (missing from first submission and caused no outputs for p2)
	prediction = prediction[0]
	return prediction

