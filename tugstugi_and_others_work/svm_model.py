# Author: tugstugi

import sentencepiece as spm
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

from sklearn.feature_extraction.text import *
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

import time
import pickle # or pickle5
import itertools
import warnings
warnings.filterwarnings('ignore')

# For convenience
df_path = '../../main_dataset/'
trained_model_and_fitted_encoder_path = '../models/'
plots_path = '../plots/'

# 1111 dataset with 'Асуулт' label created and all rows where 'content' contains numbers are dropped.
url = 'https://drive.google.com/file/d/1PCMROt6zbd90AfODLSEGogYfqzHnohTx/view?usp=sharing'
path = 'https://drive.google.com/uc?export=download&id=' + url.split('/')[-2]
df = pd.read_csv(path)

sp = spm.SentencePieceProcessor()
sp.Load('../mongolian_bert_sentencepiece/mn_uncased.model')

def sp_tokenize(w):
    return sp.EncodeAsPieces(w)

# Stratified train and test split
train, test = train_test_split(df, test_size = 0.1, random_state = 999, stratify = df['type_text'])

# Creating SVM model pipeline
text_clf = Pipeline([('vect', CountVectorizer(tokenizer = sp_tokenize, lowercase = True)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss = 'hinge', penalty = 'l2', alpha = 1e-4, max_iter = 5, random_state = 0))])

t = time.time()
text_clf = text_clf.fit(train['content'], train['type_text'])
t = time.time()-t
print("Training time in seconds: ", t)

t = time.time()
predicted = text_clf.predict(test['content'])
t = time.time()-t
print("Prediction time in seconds: ", t)

print("Feature count:", len(text_clf.named_steps['vect'].vocabulary_))
print("Classifier accuracy: ", np.mean(predicted == test['type_text']))
