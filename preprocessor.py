# Importing packages

import re
from nltk.stem import WordNetLemmatizer
from gensim.parsing.preprocessing import remove_stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize

def preprocess_text(text):
    # Remove mentions (@username) and hashtags (#tech)
    text = re.sub(r'@\w+|#', '', text)
    
    # Remove special characters and punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # remove extra whitespace
    text = ' '.join(text.split())

    # Removing stopwords and convert the text to lowercase
    text = remove_stopwords(text.lower())

    # Tokenization
    text = word_tokenize(text)

    #lemmatization
    # lemma = EnglishStemmer()
    lemma = WordNetLemmatizer()
    # text = ' '.join([lemma.stem(word) for word in text])
    text = ' '.join([lemma.lemmatize(word) for word in text])
    return text