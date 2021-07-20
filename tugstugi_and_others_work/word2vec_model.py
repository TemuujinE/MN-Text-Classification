# Author: sharavsambuu


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import sentencepiece as spm

import nltk
from sklearn.feature_extraction.text import *
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

import time
import itertools
import pickle
import string
import warnings
warnings.filterwarnings('ignore')

stopwordsmn = ['аа','аанхаа','алив','ба','байдаг','байжээ','байна','байсаар','байсан',
               'байхаа','бас','бишүү','бол','болжээ','болно','болоо','бэ','вэ','гэж','гэжээ',
               'гэлтгүй','гэсэн','гэтэл','за','л','мөн','нь','тэр','уу','харин','хэн','ч',
               'энэ','ээ','юм','үү','?','', '.', ',', '-','ийн','ын','тай','г','ийг','д','н',
               'ний','дээр','юу']

sp = spm.SentencePieceProcessor()
sp.Load('../mongolian_bert_sentencepiece/mn_uncased.model')


# For convenience
df_path = '../../main_dataset/'
trained_model_and_fitted_encoder_path = '../models/'
plots_path = '../plots'

# Loading model trained using Kaggle kernel
reconstructed_model = keras.models.load_model(trained_model_and_fitted_encoder_path + \
                                              'word2vec_fb_pretrained_model.h5')

# Loading fitted encoder classes
encoder = LabelBinarizer()
encoder.classes_ = np.load(trained_model_and_fitted_encoder_path + 'type_texts.npy')

# Loading word index for encoding
with open('../dataset/word_index.pickle', 'rb') as handle:
    word_index = pickle.load(handle)


# Sample texts for trying out the model
to_pred = ['Эмийн найдвартай байдлыг хангаж өгнө үү . Эм чимээгүй дайн үүсгэж байна.Ямарч баталгаа алга жнь: хугацаа нь дууссан гэх мэт',
          'Сайн байна уу? Та бүхэндээ баярлалаа. Миний санал: өмнөх жилийн удаан буудайн борлуулалтын үнэ 350000 төгрөг байсан. Энэ жил яагаад бууруулахгүй гэчихээд бага үнээр авав?',
          'Засгийн газраас хэрэглүүлж байгаа 100000 орон сууцны хөрөнгөө төрийн банкинд өгөөч ээ байрандаа ормоор байна',
          'Энэ засгийн газар хүн бүрд сая төгрөг өгөх болох уу?']


def sp_tokenize(w):
    return sp.EncodeAsPieces(w)

MAX_LEN = 512
def encode_content(text):
    return [word_index.get(i, 2) for i in text]

# For storing each nested-lists
to_pred_preprocessed = []

for entry in to_pred:
    sentences = nltk.sent_tokenize(entry)
    content_sentences_stopwords = []
    
    # Removing stop words from all sentences in a sentence
    for sentence in sentences:
        tokens = nltk.word_tokenize(sentence)
        tokens = [w.lower() for w in tokens]
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        words = [word for word in stripped if word.isalpha()]
        words_stopwords = [w for w in words if not w in stopwordsmn]
        
        content_sentences_stopwords.append(words_stopwords)
    
    # Storing stop word removed sentences        
    to_pred_preprocessed.append(content_sentences_stopwords)

# Joining each nested-lists
for i in range(len(to_pred_preprocessed)):
    to_pred_preprocessed[i] = list(itertools.chain(*to_pred_preprocessed[i]))[:MAX_LEN]

# Preparing text words for prediction
to_pred_entry = [encode_content(sent) for sent in to_pred_preprocessed]
to_pred_entry = keras.preprocessing.sequence.pad_sequences(to_pred_entry,
                                                           value = word_index["<PAD>"],
                                                           padding = 'post',
                                                           maxlen = MAX_LEN)

for i in range(len(to_pred_entry)):
    data_words = " ".join(to_pred_preprocessed[i])
    data_indexes = to_pred_entry[i]
    print(f'Text_{i}:', data_words)

    pred = reconstructed_model.predict([[data_indexes]])
    print('Predicted type:', encoder.classes_[np.argmax(pred)], '\n')
