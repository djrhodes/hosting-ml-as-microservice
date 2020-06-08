import nltk
from nltk import data
from string import punctuation
from nltk.stem import WordNetLemmatizer
import re
from nltk.util import everygrams
import pickle
from nltk.tokenize import word_tokenize
import json
from nltk.corpus import stopwords

nltk.data.path.append("/tmp")
from nltk import download
download('punkt',download_dir="/tmp")
download('stopwords',download_dir="/tmp")
download('wordnet',download_dir="/tmp")
# #THEN remove the zip files!

print('getting sdtop words')
stopwords_eng = stopwords.words("english")
print('lemmatizing')
lemmatizer = WordNetLemmatizer()


def clean_words(words):
    return [w for w in words if w not in stopwords.words("english") and w not in punctuation]


def extract_features(document):
    words = word_tokenize(document)
    lemmas = [str(lemmatizer.lemmatize(w)) for w in words if w not in stopwords_eng and w not in punctuation]
    document = " ".join(lemmas)
    document = document.lower()
    document = re.sub(r'[^a-zA-Z0-9\s]', ' ', document)
    words = [w for w in document.split(" ") if w!="" and w not in stopwords_eng and w not in punctuation]
    return [str('_'.join(ngram)) for ngram in list(everygrams(words, max_len=3))]


def get_word_dict(words):
    words = clean_words(words)
    return dict([(w, True) for w in words])

print('opening model file')
model_file = open("sa_classifier.pickle", "rb")
print('model file opened')
model = pickle.load(model_file)
print('model file loaded')
model_file.close()


def get_sentiment(review):
    words = extract_features(review)
    words = get_word_dict(words)
    return model.classify(words)


def lambda_handler(event,context):
    review = event['body']
    print(review)
    sentiment = get_sentiment(review)
    print(sentiment)
    return { 'statusCode': 200, 'body': json.dumps(get_sentiment(review)) }

lambda_handler({"body":"This movie is terrible."}, None)
#lambda_handler({"body":"This movie is amazing, with witty dialog and beautiful shots."}, None)
