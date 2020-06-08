#!/usr/bin/env python
# coding: utf-8

# ## Part 3: Deploying as a FaaS
# 
# <a href="https://colab.research.google.com/github/peckjon/hosting-ml-as-microservice/blob/master/part3/predict_sentiment_analysis.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# ### Download corpuses
# 
# Since we won't be doing any model-training in this step, we don't need the 'movie_reviews' corpus. However, we will still need to extract features from our input before each prediction, so we make sure 'punkt' and 'stopwords' are available for tokenization and stopword-removal. If you added any other corpuses in Part 2, consider whether they'll be needed in the prediction step.

# In[1]:


from nltk import download

download('punkt')
download('stopwords')


# ### Define feature extractor and bag-of-words converter
# 
# IMPORTANT: your predictions will only work properly if you use the same feature extractor that you trained your model with, so copy your updated `extract_features` method over from Part 2, replacing the method below. 

# In[2]:


from nltk.corpus import stopwords
from string import punctuation

stopwords_eng = stopwords.words("english")
def extract_features(words):
    return [w for w in words if w not in stopwords_eng and w not in punctuation]

def bag_of_words(words):
    bag = {}
    for w in words:
        bag[w] = bag.get(w,0)+1
    return bag


# ### Import your pickled model file (non-Colab version)
# 
# In Part 2, we saved the trained model as "sa_classifier.pickle". Now we'll unpickle that file to get it back into memory. Either copy that file into the same folder as this notebook ("part3"), or adjust the path below to "../part2/sa_classifier.pickle" so it reads the file from the folder where it was saved.

# In[3]:


import pickle
import sys

if not 'google.colab' in sys.modules:
    model_file = open('sa_classifier.pickle', 'rb')
    model = pickle.load(model_file)
    model_file.close()


# ### Import your pickled model file (Colab version)
# 
# If you're running this notebook on Colab, we need to retrieve the pickled model from [Google Drive](https://drive.google.com) before we can unpickle it. This code looks for "sa_classifier.pickle" in a folder called "Colab Output"; if you have moved the file elsewhere, change the path below.

# ### Define a method for prediction
# 
# In the prediction step, we'll be taking a single piece of text input and asking the model to classify it. Models need the input for the prediction step to have the same format as the data provided during training -- so we must tokenize the input, run the same `extract_features` method that we used during training, and convert it to a bag of words before sending it to the model's `classify` method.
# 
# Note: if you have (from Part 2) changed your `extract_features` method to accept the full text instead of a tokenized list, then you can omit the tokenization step here.

# In[4]:


import pickle
import sys

if 'google.colab' in sys.modules:
    from google.colab import drive
    drive.mount('/content/gdrive')
    get_ipython().system("ls '/content/gdrive/My Drive/Colab Output'")
    model_file = open('/content/gdrive/My Drive/Colab Output/sa_classifier.pickle','rb')
    model = pickle.load(model_file)
    model_file.close()
    print('Model loaded from /content/gdrive/My Drive/Colab Output')


# In[7]:


from nltk.tokenize import word_tokenize

def get_sentiment(review):
    words = word_tokenize(review)
    words = extract_features(words)
    words = bag_of_words(words)
    return model.classify(words)


# ### Run a prediction
# 
# Test out your `get_sentiment` method on some sample inputs of your own devising: try altering the two reviews below and see how your model performs. It won't be 100% correct, and we're mostly just looking to see that it is able to run at all, but if it sems to *always* be wrong, that may indicate you've missed a critical step above (e.g. you haven't copied over all the changes to your feature extractor from Part 2, or you've loaded the wrong model file, or provided un-tokenized text when a list of words was expected).

# In[8]:


review = event['inputText']
sentimentOfReview = get_sentiment(positive_review)
print('sentiment of review: '+sentimentOfReview)

# In[ ]:




