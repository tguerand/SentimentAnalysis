## Requirements
```
pip install numpy
pip install pandas
pip install tensorflow
```

## Import Data

First you need to download the training set :
https://drive.google.com/uc?export=download&id=1dO4aubOro159awBPzD675c22qtXMOVp0

Next, you should copy the file to you local git-cloned folder "Sentiment Analysis/DATA"


In case you want to work directly on your Google Drive follow the instructions on the ipynb file.

Then you are set up to run the code :)

## Data Analysis
Visualisation and stuff

## Preparing Data

```
sentences = df['text']
tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(sentences)
word_index=tokenizer.word_index
sequences=tokenizer.texts_to_sequences(sentences)
sequences
```
A tokenizer is a built-in function in keras which helps with encoding a text vector v.

tokenizer.word_index allows us to encode in a dictionary the most n frequent words in the vector v (n being a parameter defined when calling the tokenizer). 

tokenizer.texts_to_sequences allows us to encode a vector v based on the dictionary word_index
