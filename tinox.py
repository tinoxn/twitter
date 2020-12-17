import streamlit as st
import pickle 
from sklearn.feature_extraction.text import CountVectorizer
import preprocessor as p
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split


#set up punctuations we want to be replaced
REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\|)|(\()|(\))|(\[)|(\])|(\%)|(\$)|(\>)|(\<)|(\{)|(\})")
REPLACE_WITH_SPACE = re.compile("(<br\s/><br\s/?)|(-)|(/)|(:).")

def clean_tweets(df):
    tempArr = []
    for line in df:
        # clean using tweet_preprocessor
        tmpL = p.clean(line)
        # remove all punctuation
        tmpL = REPLACE_NO_SPACE.sub("", tmpL.lower())
        tmpL = REPLACE_WITH_SPACE.sub(" ", tmpL)
        tempArr.append(tmpL)
    return tempArr

pickle_in = pickle_in = open("moody_sentiment_model.sav", "rb")
model = pickle.load(pickle_in)

# datasets
train = pd.read_csv("SentimentDataset_train.csv")
test = pd.read_csv("SentimentDataset_test.csv")

train_tweet = clean_tweets(train["tweet"])
train_tweet = pd.DataFrame(train_tweet)

# append cleaned tweets to the training dataset
train["clean_tweet"] = train_tweet

test_tweet = clean_tweets(test["tweet"])
test_tweet = pd.DataFrame(test_tweet)

test["clean_tweet"] = test_tweet

y = train.label.values

x_train, x_test, y_train, y_test = train_test_split(train.clean_tweet.values, y,
                                                   stratify = y,
                                                   random_state = 1,
                                                   test_size = 0.3,
                                                   shuffle = True)


# initilizing the vectorizer
vectorizer = CountVectorizer(binary = True, stop_words = "english")
vectorizer.fit(list(x_train) + list(x_test))

def classify_tweet(user_text):
    clean = clean_tweets([user_text])
    text_vec = vectorizer.transform(clean)
    predicted = model.predict(text_vec)
    return predicted

st.title("Welcome to Kay website")
st.header("Enter the tweet text")
user_text = st.text_input("Enter the tweet text")
result = ""
r = ""
if st.button("Classify Tweet"):
    result = classify_tweet(user_text)
    if result == [0]:
        r = "Negative"
    elif result == [1]:
        r = "Positive"
st.success('The sentiment for this tweet is : {}'.format(r))
