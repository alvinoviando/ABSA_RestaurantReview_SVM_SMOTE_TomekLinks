import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import sklearn
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Embedding, GRU, Dense, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.backend import clear_session


sns.set_style("whitegrid")
np.random.seed(0)

max_seq_len = 12
MAX_NB_WORDS = 100000
num_epochs = 30
embed_dim = 300 


#load data
def read_data(data_file):
	df = pd.read_csv(data_file)
	#st.dataframe(df)
	train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

	y_train = train_df['label_score'].values
	y_val = val_df['label_score'].values

	raw_docs_train = train_df['title'].tolist()
	raw_docs_val = val_df['title'].tolist()
	return raw_docs_train, raw_docs_val, y_train, y_val

#proprocessing
def preprocessing(raw_docs):
	stop_words = set(stopwords.words('indonesian'))
	stop_words.update(['.', ',','!', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])

	processed_docs = []
	for doc in raw_docs:
		tokens = word_tokenize(doc)
		filtered = [word for word in tokens if word not in stop_words]
		processed_docs.append(" ".join(filtered))

	return processed_docs

def step_preproccess(raw_docs):
	list_stopword = set(stopwords.words('indonesian'))

	#step by step preprocessing
	step_preproccess = []
	token_0 = word_tokenize(raw_docs[2])
	stop_0 = [word for word in token_0 if word not in list_stopword]
	step_preproccess.append(" ".join(stop_0))

	list_stopword.update(['.', ',', '!', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])

	punct_0 = [word for word in stop_0 if word not in list_stopword]
	step_preproccess.append(" ".join(punct_0))

	return step_preproccess

def tokenization_training(processed_docs_train, processed_docs_val):
	tokenizer = Tokenizer(num_words = MAX_NB_WORDS, lower=True, char_level=False)
	tokenizer.fit_on_texts(processed_docs_train)
	train_sequences = tokenizer.texts_to_sequences(processed_docs_train)
	val_sequences = tokenizer.texts_to_sequences(processed_docs_val)
	word_index = tokenizer.word_index
	word_seq_train = pad_sequences(train_sequences, maxlen = max_seq_len)
	word_seq_val = pad_sequences(val_sequences, maxlen= max_seq_len)

	with open('tokenizer.pkl', 'wb') as handle:
		pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

	return word_index, word_seq_train, word_seq_val

def model_building(word_index, word_seq_train, word_seq_val, y_train, y_val):
	words_not_found = []
	nb_words = min(MAX_NB_WORDS, len(word_index)+1)
	embedding_matrix = np.zeros((nb_words, embed_dim))

	#load fasttext
	fasttext_word_to_index = pickle.load(open("fasttext_voc", 'rb'))

	for word, i in word_index.items():
		if i >= nb_words:
			continue
		embedding_vector = fasttext_word_to_index.get(word)
		if (embedding_vector is not None) and len(embedding_vector) > 0:
			embedding_matrix[i] = embedding_vector
		else:
			words_not_found.append(word)
	#print('number of null word embeddings: ',np.sum(np.sum(embedding_matrix, axis=1) == 0) )

	#model = tf.keras.Sequential()
	model = Sequential()
	model.add(Embedding(nb_words,embed_dim,input_length=max_seq_len, weights=[embedding_matrix],trainable=False))
	model.add(GRU(32))
	model.add(Dropout(0.4))
	model.add(Dense(1,activation='sigmoid'))
	model.summary()

	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	es_callback = EarlyStopping(monitor='val_loss', patience=3)

	history = model.fit(word_seq_train, y_train, batch_size=256,epochs=num_epochs, validation_data=(word_seq_val, y_val), callbacks=[es_callback], shuffle=False)

	#generate loss plots
	
	plt.figure()
	plt.plot(history.history['loss'], lw=2.0, color='b', label='train')
	plt.plot(history.history['val_loss'], lw=2.0, color='r', label='val')
	plt.suptitle('Loss History', fontsize=15)
	plt.xlabel('Epochs')
	plt.ylabel('Cross-Entropy Loss')
	plt.legend(loc='upper right')
	#plt.savefig('static/images/loss.png')
	lossfile = BytesIO()
	plt.savefig(lossfile, format='png')
	lossfile.seek(0)
	loss_png = "data:image/png;base64,"
	loss_png += base64.b64encode(lossfile.getvalue()).decode('utf-8')

	#generate accuracy plots
	plt.figure()
	plt.plot(history.history['accuracy'], lw=2.0, color='b', label='train')
	plt.plot(history.history['val_accuracy'], lw=2.0, color='r', label='val')
	plt.suptitle('Acccuracy History', fontsize=15)
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.legend(loc='upper left')
	#plt.savefig('static/images/accuracy.png')
	accfile = BytesIO()
	plt.savefig(accfile, format='png')
	accfile.seek(0)
	acc_png = "data:image/png;base64,"
	acc_png += base64.b64encode(accfile.getvalue()).decode('utf-8')

	model.save('gru.h5')
	#predictions = model.predict_classes(word_seq_train)
	#y_pred = (predictions>0.5).astype('int32')
	#accuracy = sklearn.metrics.accuracy_score(y_train, y_pred)
	loss_and_acc_train = model.evaluate(word_seq_train, y_train, verbose=2)
	loss_train = round(loss_and_acc_train[0],2)
	accuracy_train = round(loss_and_acc_train[1],2)

	loss_and_acc_val = model.evaluate(word_seq_val, y_val, verbose=2)
	loss_val = round(loss_and_acc_val[0],2)
	accuracy_val = round(loss_and_acc_val[1],2)

	clear_session()

	return loss_png, acc_png, loss_train, accuracy_train, loss_val, accuracy_val

def testing(file):

	df = pd.read_csv(file)
	raw_docs_test = df['title'].tolist()
	y_test = df['label_score'].values

	stop_words = set(stopwords.words('indonesian'))
	stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])

	processed_docs_test = []
	for doc in raw_docs_test:
		tokens = word_tokenize(doc)
		filtered = [word for word in tokens if word not in stop_words]
		processed_docs_test.append(" ".join(filtered))

	with open('tokenizer.pkl', 'rb') as f:
		tokenizer = pickle.load(f)

	test_sequences = tokenizer.texts_to_sequences(processed_docs_test)
	word_seq_test = pad_sequences(test_sequences, maxlen = max_seq_len)

	model = load_model('gru.h5')
	predictions = model.predict(word_seq_test)
	sample = df
	sample['pred'] = (predictions>0.5).astype('int32')

	#clickbait_dict = {1: 'clickbait', 0:'non-clickbait'}
	output_pred = sample[['title', "label_score", 'pred']]
	#output_pred.replace(clickbait_dict, inplace=True)

	#print("Accuracy With fastText :")
	accuracy_score = sklearn.metrics.accuracy_score(sample['label_score'], sample['pred'])
	accuracy = round(accuracy_score, 2)
	return accuracy, output_pred

def predict(data_pred):

	stop_words = set(stopwords.words('indonesian'))
	stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])

	processed_docs_pred = []
	tokens = word_tokenize(data_pred)
	filtered = [word for word in tokens if word not in stop_words]
	processed_docs_pred.append(" ".join(filtered))

	with open('tokenizer.pkl', 'rb') as f:
		tokenizer = pickle.load(f)

	pred_sequences = tokenizer.texts_to_sequences(processed_docs_pred)
	word_seq_pred = pad_sequences(pred_sequences, maxlen = max_seq_len)

	model = load_model('gru.h5')
	predictions = model.predict(word_seq_pred)
	if (predictions > 0.5):
		hasil_pred = 'Clickbait'
		return hasil_pred
	else:
		hasil_pred = 'Non-clickbait'
		return hasil_pred