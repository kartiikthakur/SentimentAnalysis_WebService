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
import matplotlib.pyplot as plt

def buidTrainingSet():
    import csv
    import time
    
    TrainingSet = []
    
    with open('train_500.csv','rb') as csvfile:
        lineReader = csv.reader(csvfile)
        for row in lineReader:
            TrainingSet.append({ "SentimentText":row[0], "Sentiment":row[1]})
                  
    rate_limit = 180
    sleep_time = 900/180
    # now we write them to the empty CSV file
    with open('train','wb') as csvfile:
        linewriter = csv.writer(csvfile)
        for tweet in TrainingSet:
            try:
                linewriter.writerow([ tweet["SentimentText"], tweet["Sentiment"]])
            except Exception as e:
                print(e)

    return TrainingSet

def buidTestSet():
    import csv
    import time
    
    TestSet = []
    
    with open('trumptweet.csv','rb') as csvfile:
        lineReader = csv.reader(csvfile)
        for row in lineReader:
            TestSet.append({"Sentiment":None, "SentimentText":row[0]})
                  
    rate_limit = 180
    sleep_time = 900/180
    with open('testDataFile','wb') as csvfile:
        linewriter = csv.writer(csvfile)
        for tweet in TestSet:
            try:
                linewriter.writerow([ tweet["SentimentText"], tweet["Sentiment"]])
            except Exception as e:
                print(e)
                
    return TestSet

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


testDataSet = buidTestSet()
trainingData = buidTrainingSet()

    
tweetProcessor = PreProcessTweets()
preprocessedTrainingSet = tweetProcessor.processTweets(trainingData)
preprocessedTestSet = tweetProcessor.processTweets(testDataSet)




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


word_features = buildVocabulary(preprocessedTrainingSet)
trainingFeatures=nltk.classify.apply_features(extract_features,preprocessedTrainingSet)
classifier=nltk.NaiveBayesClassifier.train(trainingFeatures)
joblib.dump(classifier, "Sentiment_Analysis_nb")
NBResultLabels = [classifier.classify(extract_features(tweet[0])) for tweet in preprocessedTestSet]



total_tweets = len(NBResultLabels)
positive_tweets = NBResultLabels.count('1')
negative_tweets = NBResultLabels.count('0')
labels = 'Positive tweets', 'Negative tweets'
sizes = [positive_tweets, negative_tweets]
colors = ['gold', 'yellowgreen']
explode = (0.1, 0)
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.show()



