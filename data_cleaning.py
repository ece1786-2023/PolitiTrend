import pandas as pd

import string
import re

import matplotlib.pyplot as plt
import pandas as pd
import sklearn

from sklearn.model_selection import train_test_split
import numpy as np

import nltk

from nltk.corpus import stopwords
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download()

nltk.download('stopwords')

import nltk


from nltk.stem import PorterStemmer

ps = PorterStemmer()

# import tensorflow as tf

df = pd.read_csv("/Users/lifeifan/Desktop/ece1786/project/data1.csv")

df['cmp_code'] = df['cmp_code'].replace({'conservatism': 0, 'progressivism': 1})
df = df.rename(columns={'cmp_code': 'label'})

def remove_stop_words (dataframe,target_column_name,new_column_name) :
    dataframe[new_column_name] = dataframe[target_column_name].apply(lambda x:' '.join([item for item in x.split() if item not in stopwords.words('english')]))
    return dataframe


def remove_punctuations(dataframe,target_column_name,new_column_name):
    dataframe[new_column_name] = dataframe[target_column_name].apply(lambda x: "".join([char for char in x if char not in string.punctuation]))
    return dataframe

def stem_text(dataframe,target_column_name,new_column_name):
    dataframe[new_column_name] = dataframe[target_column_name].apply(lambda x: ' '.join([ps.stem(word) for word in x.split()]))
    return dataframe