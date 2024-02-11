import openai

import pickle
import random
import numpy
import json

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import SGD

Lemmatizer = WordNetLemmatizer

intents = json.loads(open('intents.json').read())

words =[]
classes = []
documents = []

