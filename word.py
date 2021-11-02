import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pickle 


next_word = pickle.load(open(r'C:\Users\jayes\word\Word_model.pkl' , 'rb'))


st.title('Next Word Predictor')

df=pd.read_csv(r'C:\Users\jayes\Downloads\archive\abcnews-date-text.csv')

from nltk.corpus import stopwords

sw = stopwords.words("english")


sentence = st.text_input('Input your sentence here:') 

import string

def clean_text(msg):
    msg = msg.translate(str.maketrans(' ',' ', string.punctuation))
    msg = msg.lower()
    msg = msg.split(' ')
    words = [x for x in msg if x not in sw]
    return  ' '.join(msg)

df['cleaned_text']  = df['headline_text'].apply(clean_text)

from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer_obj=Tokenizer()
tokenizer_obj.fit_on_texts(df['cleaned_text'])

vocabulary_size=len(tokenizer_obj.word_index)
encoded_vec=[]
for text in df['cleaned_text']:
  encoded_vec.append(tokenizer_obj.texts_to_sequences([text])[0])

  max_len_vec=0
for values in encoded_vec:
  if len(values)>max_len_vec:
    max_len_vec=len(values)

from tensorflow.keras.preprocessing.sequence import pad_sequences
padded_vec=pad_sequences(encoded_vec,maxlen=max_len_vec,padding='pre')

x=padded_vec[:,0:-1]
y=padded_vec[:,-1]

from tensorflow import keras

model=keras.Sequential()
#Embedding layer
#input_dim is vocabulary_size + 1 because 0 is not considered when calculating vocab size
model.add(keras.layers.Embedding(input_dim=vocabulary_size+1,output_dim=20,input_length=max_len_vec-1,))#Bcz we have taken the last value as target label
model.add(keras.layers.LSTM(units=100))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(units=100,activation='relu'))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(units=vocabulary_size,activation='softmax'))


def predict_next_word(seed_text):
  cleaned_text=clean_text(seed_text)
  token_list = tokenizer_obj.texts_to_sequences([cleaned_text])[0]
  token_list = pad_sequences([token_list], maxlen=max_len_vec-1, padding='pre')
  prediction = model.predict(token_list)
  predictions=np.argmax(prediction)
  predicted_word=tokenizer_obj.sequences_to_texts([[predictions]])[0]
  return seed_text+' '+predicted_word


if st.button('predict'):
    predictions = predict_next_word(sentence)
    st.write(predictions)

