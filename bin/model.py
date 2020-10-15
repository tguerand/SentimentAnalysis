# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 10:21:59 2020

@author: trist
"""

import tensorflow as tf
from tensorflow import keras

def my_model(max_words=5000, max_len=150, embedd_size=50):
    
    """Model to classify text
    
    Arguments:
        
        max_word: int
            Maximum number of words for the tokenizer
        max_len : int
            Maximum length of a sequence
    
    Returns:
        
        model : keras.Model
            Our custom model"""
    
    
    inputs = keras.layers.Input(name='inputs',shape=[max_len])
    
    x = keras.layers.Embedding(max_words,embedd_size,input_length=max_len)(inputs)
    x = keras.layers.LSTM(64)(x)
    x = keras.layers.Dense(256,name='FC1')(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(1,name='out_layer')(x)
    x = keras.layers.Activation('sigmoid')(x)
    model = keras.Model(inputs=inputs,outputs=x)
    
    return model