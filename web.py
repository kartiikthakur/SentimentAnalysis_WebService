# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import nltk
from sklearn.linear_model import LogisticRegression
from nltk.classify.scikitlearn import SklearnClassifier
import json
import joblib
import csv
import twitter, tweepy
import sys, operator
import re
import nltk
import unicodedata
from nltk.tokenize import word_tokenize
from string import punctuation 
from nltk.corpus import stopwords
import requests
from flask import Flask, jsonify, request, abort, make_response, url_for
from flask_basicauth import BasicAuth
from requests.auth import HTTPBasicAuth
from textblob import TextBlob

app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = 'ualbany'
app.config['BASIC_AUTH_PASSWORD'] = 'datamining'
basic_auth = BasicAuth(app)


@app.route('/sentiment', methods = ["Post"])
@basic_auth.required
def sentiment():
    try:
        content = request.data
        tweetProcessor = PreProcessTweets()
        TestSet = []
        NBResultLabel = ''
        for row in content:
            TestSet.append({"Sentiment": None, "SentimentText": row[0]})

        preprocessedTestSet = tweetProcessor.processTweets(TestSet)
        classifier = joblib.load("Sentiment_Analysis")
        NBResultLabels = [classifier.classify(extract_features(tweet[0])) for tweet in preprocessedTestSet]
        print(NBResultLabels)
        blob = TextBlob(content)
        for sentence in blob.sentences:
            NBResultLabel = sentence.sentiment.polarity

        if (NBResultLabel > 0):
            output = "The given tweet is a positive tweet"
        else:
            output = "The given tweet is a negative tweet"

        json_result = {
        "Output": output
        }
        return make_response(json.dumps(str(json_result)), 200)

    except Exception as e:
        print(e)
        abort(400, description="Unable to preprocess text")


def buildVocabulary(preprocessedTrainingData):
    all_words = []
    
    for (words, sentiment) in preprocessedTrainingData:
        all_words.extend(words)

    wordlist = nltk.FreqDist(all_words)
    word_features = wordlist.keys()
    
    return word_features


def extract_features(tweet):
    try:
        tweet_words=set(tweet)
        features={}
        for word in word_features:
            if word in tweet_words:
                features['contains({})'.format(word.lower())] = True
    except Exception as e:
        pass
    return features

class PreProcessTweets:
    def __init__(self):
        self._stopwords = set(stopwords.words('english') + list(punctuation) + ['AT_USER','URL'])
        
    def processTweets(self, list_of_tweets):
        processedTweets=[]
        for tweet in list_of_tweets:
            text = tweet["SentimentText"]
            processedTweets.append((self._processTweet(text),tweet["Sentiment"]))           
        return processedTweets
    
    def _processTweet(self, tweet):
        try:
            tweet = tweet.lower()
            tweet = tweet.decode('utf-8')
            tweet = re.sub('((www\\.[^\\s]+)|(https?://[^\\s]+))', 'URL', tweet) 
            tweet = re.sub('@[^\\s]+', 'AT_USER', tweet)
            tweet = re.sub(r'#([^\\s]+)', r'\\1', tweet) 
            tweet = word_tokenize(tweet)
        
        except:
            pass  
        return [word for word in tweet if word not in self._stopwords]

def buidTestSet():
    import csv
    import time
    
    TestSet = []
    
    with open('test2.csv','rb') as csvfile:
        lineReader = csv.reader(csvfile)
        for row in lineReader:
            TestSet.append({"Sentiment":None, "SentimentText":row[0]})
                  
    rate_limit = 180
    sleep_time = 900/180
    # now we write them to the empty CSV file
    with open('testDataFile','wb') as csvfile:
        linewriter = csv.writer(csvfile)
        for tweet in TestSet:
            try:
                linewriter.writerow([ tweet["SentimentText"], tweet["Sentiment"]])
            except Exception as e:
                print(e)
                
    return TestSet


@app.errorhandler(400)
def errorHandler400(error):
    return make_response(jsonify({'error': error.description}), 400)

@app.errorhandler(500)
def errorHandler500(error):
    return make_response(jsonify({'error': error.description}), 500)

if __name__ == '__main__':
    app.run(debug=True, port=5000)