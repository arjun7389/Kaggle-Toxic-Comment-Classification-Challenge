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

train_text=[]
test_text=[]

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
	train_text.append(w)


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
	test_text.append(w)
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




tok_text=Tokenizer()
tok_text.fit_on_texts(train_text)
text_vocab_size=len(tok_text.word_index)+1
encode_text=tok_text.texts_to_sequences(train_text)
max_len_text=200
padded_text=pad_sequences(encode_text, maxlen=max_len_text, padding='post')
embed_mat_text=zeros((text_vocab_size,100))






for word, i in tok_text.word_index.items():
	embedding_vector = embed_index.get(word)
	if embedding_vector is not None:
		embed_mat_text[i] = embedding_vector


tok_test_text=Tokenizer()
tok_test_text.fit_on_texts(test_text)
test_text_vocab_size=len(tok_test_text.word_index)+1
encode_test_text=tok_test_text.texts_to_sequences(test_text)
max_len_test_text=200
padded_test_text=pad_sequences(encode_test_text, maxlen=max_len_test_text, padding='post')



###########################################################################################

input1=Input(shape=(200,))              ########newsbody

embed1=Embedding(output_dim=100,input_dim=text_vocab_size, input_length=200, weights=[embed_mat_text], trainable=False)(input1)


lstm1=Bidirectional(LSTM(100, return_sequences=False))(embed1)
den1=Dense(64, activation='relu')(lstm1)




den3=Dense(32, activation='relu')(den1)
#den3=Dense(32, activation='relu')(den2)
den4=Dense(16, activation='relu')(den3)
output=Dense(6,activation='sigmoid')(den4)
model=Model(inputs=[input1],outputs=output)
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
myFile = open('pred_bi_new.csv', 'w')
for sd in range(1):
	
	model.fit([padded_text],ytrain, batch_size = 64, epochs =10, verbose = 1,shuffle=True)
	score1 = model.predict([padded_test_text])
	with myFile:
    		writer = csv.writer(myFile)
    		writer.writerows(score1)
	#fw.close()
