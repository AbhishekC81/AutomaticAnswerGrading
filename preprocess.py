import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def stop_words_removal(tokens):
    stopwords = nltk.corpus.stopwords.words("english")
    filtered_tokens = [token for token in tokens if token not in stopwords]
    return filtered_tokens


def lemmatization(tokens):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens


def preprocess_text(text):
    # Tokeinzation
    tokens = nltk.word_tokenize(text)

    # Lowercasing
    tokens = [token.lower() for token in tokens]

    # Stop Words Removal
    tokens = stop_words_removal(tokens)

    # Lemmatization
    tokens = lemmatization(tokens)

    # Reconstructing text
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text




