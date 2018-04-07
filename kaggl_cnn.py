import numpy as np
#from gensim import corpora, models, similarities
#import gensim.models.keyedvectors as word2vec
import string
import os
import csv
from keras.utils import np_utils
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.merge import Concatenate
from keras.layers import Conv1D, MaxPooling1D,Conv2D, MaxPooling2D, Merge, Dropout, add, concatenate, Reshape, Concatenate
from nltk.corpus import stopwords
from keras.utils import multi_gpu_model
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
#from keras.layers import Conv1D, MaxPooling1D,Conv2D, MaxPooling2D, Merge, Dropout, add, concatenate, Reshape, Concatenate
from keras.utils import plot_model
#from keras.layers import Conv2D, MaxPooling2D, Merge, Dropout, add, concatenate, Reshape, Concatenate
from keras.optimizers import SGD
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.layers import Input
from keras.models import Model

import pandas as pd
#from sklearn.metrics import classification_report,confusion_matrix

os.environ["CUDA_VISIBLE_DEVICES"]="0"
stop_words = set(stopwords.words("english"))

df1=pd.read_csv("train.csv")
l=df1.values.tolist()

df2=pd.read_csv("test.csv")
l2=df2.values.tolist()

ytrain=[]


line_tr=[]
line_ts=[]
for i in range(len(l)):
	lab=[]
	lab.append(l[i][2])
	lab.append(l[i][3])
	lab.append(l[i][4])
	lab.append(l[i][5])
	lab.append(l[i][6])
	lab.append(l[i][6])
	ytrain.append(lab)
	words=l[i][1].split()
	stripped = filter(lambda x: x not in string.punctuation, words)
	stripped = filter(lambda x: x not in stop_words, stripped)
	stripped = [word for word in stripped if word.isalpha()]
	stack=[]
	c=1
	for k in stripped:
		if c<201:
			stack.append(k)
		c+=1
				
	w = " ".join(stack)
	line_tr.append(w)

ytrain=np.array(ytrain)
#######################################

for i in range(len(l2)):
	#if i<5000:
	words=l2[i][1].split()
	stripped = filter(lambda x: x not in string.punctuation, words)
	stripped = filter(lambda x: x not in stop_words, stripped)
	stripped = [word for word in stripped if word.isalpha()]
	stack=[]
	c=1
	for k in stripped:
		if c<201:
			stack.append(k)
		c+=1		
	w = " ".join(stack)
	line_ts.append(w)
############################
embed_index=dict()
f = open('./glove.6B/glove.6B.100d.txt')
for line in f:
	values = line.split()
	word = values[0]
	coefs = asarray(values[1:], dtype='float32')
	embed_index[word] = coefs
f.close()
#########################

train=[]

for s in line_tr:
	c=1	
	sen=[]
	for w in s.split():
		if c<201:
			if w in embed_index:
				ws=embed_index.get(w)
				if ws is not 'nan':
					senv=np.array(ws)
					sen.append(senv)								
					c+=1			
	while c<201:
		senp=np.zeros((100))
		#senp=np.array(senp)				
		if c<201:
			sen.append(senp)
			c+=1
	#print s
	sen=np.array(sen)
	#print sen	
	sen=sen.reshape(200,100,1).astype('float64')
	train.append(sen)
train=np.array(train)




test=[]
for s in line_ts:
	c=1	
	sen=[]
	lin=s.split()
	for w in lin:
		if c<201:
			if w in embed_index:
				ws=embed_index.get(w)
				if ws is not 'nan':
					senv=np.array(ws)
					sen.append(senv)			
					c+=1				
	while c<201:
		senp=np.zeros((100))
		senp=np.array(senp)				
		if c<201:
			sen.append(senp)
			c+=1
	sen=np.array(sen)
	sen=sen.reshape(200,100,1).astype('float64')
	test.append(sen)
test=np.array(test)

print ("train",train.shape)
print ("test",test.shape)
###########################################################################################


input1=Input(shape=(200,100,1,)) 
conv=Conv2D(64,(2,100),data_format='channels_last',activation='relu')(input1)
pool=MaxPooling2D(1,2)(conv)
flat=Flatten()(pool)
den1=Dense(128, activation='relu')(flat)


den2=Dense(64, activation='relu')(den1)
den3=Dense(32, activation='relu')(den2)
den4=Dense(16, activation='relu')(den3)
output=Dense(6,activation='sigmoid')(den4)
model=Model(inputs=[input1],outputs=output)
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
myFile = open('pred_cnn.csv', 'w')
for sd in range(1):
	
	model.fit([train],ytrain, batch_size = 64, epochs =10, verbose = 1,shuffle=True)
	score1 = model.predict([test])
	with myFile:
    		writer = csv.writer(myFile)
    		writer.writerows(score1)
	#fw.close()
