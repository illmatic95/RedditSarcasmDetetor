# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 19:28:03 2018

@author: mengr
"""


# coding: utf-8

# In[1]:


# some necessary imports
#import string
#import nltk
#from nltk.stem import WordNetLemmatizer
#from nltk.corpus import stopwords 
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import re

'''
load data
'''
PATH_TO_DATA = 'train-balanced-sarcasm.csv'

train_df = pd.read_csv(PATH_TO_DATA, error_bad_lines=False)


#Some comments are missing, so we drop the corresponding rows.
train_df.dropna(subset=['comment'], inplace=True)

train_sample = train_df.sample(frac = 0.3,random_state=1)
#We split data into training and validation parts.
train_texts, valid_texts, y_train, y_valid = train_test_split(train_sample['comment'], 
                                                              train_sample['label'], 
                                                              random_state=17)

'''
construct unbalanced validation set
'''
#train, valid= train_test_split(train_sample, random_state=17)
#
#train_texts = train['comment']
#y_train = train['label']
#
#valid_positive = valid[valid['label'] == 1]
#valid_negative = valid[valid['label'] == 0]
#
#valid_unbalenced = valid_negative.append(valid_positive.sample(frac = 0.2,random_state=1))
#
#valid_unbalenced = valid_unbalenced.sample(frac = 1,random_state=1)
#
#valid_texts = valid_unbalenced['comment']
#y_valid = valid_unbalenced['label']
#train_texts, valid_texts, y_train, y_valid = train_test_split(train_df['comment'], 
#                                                              train_df['label'], 
#                                                              random_state=17)
'''
extract GloVe vectors into a dictionary
'''
embeddings_dict = {}
f = open('glove.twitter.27B.100d.txt',encoding='utf8')
#f = open('glove.6B.100d.txt',encoding='utf8')
for line in f:
    val = line.split()
    word = val[0]
    try:
        vec = np.asarray(val[1:], dtype='float32')
    except ValueError:
        continue
    embeddings_dict[word] = vec
f.close()

embed_size = 100

#regex = re.compile('[^a-z]')
#def sent2vec(s):
#    M = []
#    for word in s.split():
#        word = word.lower()
#        word = regex.sub('', word)
#        try:
#            M.append(embeddings_dict[word])
#        except:
#            continue
#    M = np.array(M)
#    v = M.sum(axis=0)
#    if type(v) != np.ndarray:
#        return np.zeros(embed_size)
#    return v / np.sqrt((v ** 2).sum())
#
#xtrain_glove = [sent2vec(x) for x in train_texts]
#xvalid_glove = [sent2vec(x) for x in valid_texts]


#scl = preprocessing.StandardScaler()
#xtrain = scl.fit_transform(xtrain_glove)
#xvalid = scl.transform(xvalid_glove)


from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Bidirectional, SpatialDropout1D
from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping

'''
keras preprocessing
'''
# using keras tokenizer here
token = text.Tokenizer(num_words=None)
max_len = 70

token.fit_on_texts(list(train_texts) + list(valid_texts))
xtrain_seq = token.texts_to_sequences(train_texts)
xvalid_seq = token.texts_to_sequences(valid_texts)

# zero pad the sequences
xtrain_pad = sequence.pad_sequences(xtrain_seq, maxlen=max_len)
xvalid_pad = sequence.pad_sequences(xvalid_seq, maxlen=max_len)

word_index = token.word_index

'''
embedding matrix
'''
# create an embedding matrix for the words we have in the dataset
embedding_matrix = np.zeros((len(word_index) + 1, embed_size))
for word, i in word_index.items():
    embedding_vector = embeddings_dict.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        
'''
the model
'''
embed_out_size = 100
model = Sequential()
model.add(Embedding(len(word_index) + 1,
                     embed_out_size,
                     weights=[embedding_matrix],
                     input_length=max_len,
                     trainable=False))

model.add(Conv1D(embed_out_size, 3, kernel_initializer='he_normal', padding='valid',
                                activation='sigmoid',
                                input_shape=(1, max_len)))
model.add(MaxPooling1D(pool_size=3))
model.add(Conv1D(embed_out_size, 3, kernel_initializer='he_normal', padding='valid',
                                activation='sigmoid',
                                input_shape=(1, max_len-2)))
#model.add(MaxPooling1D(pool_size=3))

#model.add(SpatialDropout1D(0.2))
model.add(Dropout(0.25))
#model.add(LSTM(embed_size, dropout=0.2, recurrent_dropout=0.2))
#model.add(GRU(embed_size, kernel_initializer='he_normal', activation='sigmoid', 
#              dropout=0.3, recurrent_dropout=0.3, return_sequences=True))
#model.add(GRU(embed_size, kernel_initializer='he_normal', activation='sigmoid', 
#              dropout=0.3, recurrent_dropout=0.3))

model.add(GRU(embed_out_size, kernel_initializer='he_normal', activation='sigmoid', 
               dropout=0.5,return_sequences=True))
model.add(Bidirectional(LSTM(embed_out_size, kernel_initializer='he_normal', activation='sigmoid', 
               dropout=0.5))) 
#model.add(LSTM(embed_size, kernel_initializer='he_normal', activation='sigmoid', 
#               dropout=0.5))
#model.add(Dense(1024, activation='relu'))
#model.add(Dropout(0.8))

model.add(Dense(128, kernel_initializer='he_normal', activation='sigmoid'))
model.add(Dropout(0.8))

model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#history = model.fit(xtrain_pad, y=y_train, batch_size=1024, epochs=70, 
#          verbose=1, validation_data=(xvalid_pad, y_valid))
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())
#from keras import backend as K
#K.tensorflow_backend._get_available_gpus()
# Fit the model with early stopping callback
#earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, 
#                          verbose=0, mode='auto')

'''
f1
'''
#from keras.callbacks import Callback
#from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
#class Metrics(Callback):
#    def on_train_begin(self, logs={}):
#         self.val_f1s = []
#         self.val_recalls = []
#         self.val_precisions = []
# 
#    def on_epoch_end(self, epoch, logs={}):
#         val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
#         val_targ = self.validation_data[1]
#         _val_f1 = f1_score(val_targ, val_predict)
#         _val_recall = recall_score(val_targ, val_predict)
#         _val_precision = precision_score(val_targ, val_predict)
#         self.val_f1s.append(_val_f1)
#         self.val_recalls.append(_val_recall)
#         self.val_precisions.append(_val_precision)
#         print (" — val_f1: %f — val_precision: %f — val_recall %f" %(_val_f1, _val_precision, _val_recall))
#         return
# 
#metrics = Metrics()
'''
training
'''

from keras.utils import np_utils
ytrain_cat = np_utils.to_categorical(y_train)
yvalid_cat = np_utils.to_categorical(y_valid)
history = model.fit(xtrain_pad, y=ytrain_cat, validation_data=(xvalid_pad, yvalid_cat), batch_size=1024, epochs=50, 
          verbose=1)

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])  
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


#
#train_sample = train_df.sample(frac = 0.1,random_state=1)
#train_texts, valid_texts, y_train, y_valid = train_test_split(train_sample['comment'], train_sample['label'], random_state=17)


#
#
## In[14]:
#def plot_confusion_matrix(actual, predicted, classes,
#                          normalize=False,
#                          title='Confusion matrix', figsize=(7,7),
#                          cmap=plt.cm.Blues, path_to_save_fig=None):
#    """
#        This function prints and plots the confusion matrix.
#        Normalization can be applied by setting `normalize=True`.
#        """
#    import itertools
#    cm = confusion_matrix(actual, predicted).T
#    if normalize:
#        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#    
#    plt.figure(figsize=figsize)
#    plt.imshow(cm, interpolation='nearest', cmap=cmap)
#    plt.title(title)
#    plt.colorbar()
#    tick_marks = np.arange(len(classes))
#    plt.xticks(tick_marks, classes, rotation=90)
#    plt.yticks(tick_marks, classes)
#
#    fmt = '.2f' if normalize else 'd'
#    thresh = cm.max() / 2.
#    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#        plt.text(j, i, format(cm[i, j], fmt),
#                 horizontalalignment="center",
#                 color="white" if cm[i, j] > thresh else "black")
#    plt.tight_layout()
#    plt.ylabel('Predicted label')
#    plt.xlabel('True label')
#
#    if path_to_save_fig:
#        plt.savefig(path_to_save_fig, dpi=300, bbox_inches='tight')
##
##
##print(accuracy_score(y_valid, valid_pred))
#plot_confusion_matrix(y_valid, valid_pred, 
#                      tfidf_logit_pipeline.named_steps['logit'].classes_, figsize=(8, 8))

