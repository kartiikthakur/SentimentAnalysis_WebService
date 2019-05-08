# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import pylab as pl
import numpy as np
import csv
from sklearn.model_selection import cross_validate
#from sklearn.grid_search import GridSearchCV
import joblib
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as pyplot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
from os import path
import random
from wordcloud import WordCloud, STOPWORDS
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import json

stopwords = ["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"]

tweets = []
with open('train_500.csv','rb') as csvfile:
    lineReader = csv.reader(csvfile)
    for row in lineReader:
        tweets.append([int(row[1]), row[0].lower().strip()])
    

# Extract the vocabulary of keywords
vocab = dict()
for class_label, text in tweets:
    for term in text.split():
        term = term.lower()
        if len(term) > 2 and term not in stopwords:
            if vocab.has_key(term):
                vocab[term] = vocab[term] + 1
            else:
                vocab[term] = 1

# Remove terms whose frequencies are less than a threshold (e.g., 10)
vocab = {term: freq for term, freq in vocab.items() if freq > 13}
# Generate an id (starting from 0) for each term in vocab
vocab = {term: idx for idx, (term, freq) in enumerate(vocab.items())}
print(vocab)
print("The number of keywords used for generating features (frequencies): ", len(vocab))

# Generate X and y
X = []
y = []
for class_label, text in tweets:
    x = [0] * len(vocab)
    terms = [term for term in text.split()]
    for term in terms:
        if vocab.has_key(term):
            x[vocab[term]] += 1
    y.append(class_label)
    X.append(x)

print("The total number of training tweets: {} ({} positives, {}: negatives)".format(len(y), sum(y), len(y) - sum(y)))

# 10 folder cross validation to estimate the best w and b
svc = svm.SVC(kernel='linear')
Cs = range(1, 20)
clf = GridSearchCV(estimator=svc, param_grid=dict(C=Cs), cv = 10)
clf.fit(X, y)
joblib.dump(clf, "Sentiment_Analysis_svm")


# predict the class labels of new tweets
tweets = []
with open('trumptweet.csv','rb') as csvfile:
    lineReader = csv.reader(csvfile)
    for row in lineReader:
        tweets.append(row[0].lower())

# Generate X for testing tweets
test_X = []
for text in tweets:
    x = [0] * len(vocab)
    terms = [term for term in text.split() if len(term) > 2]
    for term in terms:
        if vocab.has_key(term):
            x[vocab[term]] += 1
    test_X.append(x)
test_y = clf.predict(test_X)
print("The total number of testing tweets: {} ({} are predicted as positives, {} are predicted as negatives)".format(len(test_y), sum(test_y), len(test_y) - sum(test_y)))


# Pie Chart
total_tweets = len(test_y)
positive_tweets = sum(test_y)
negative_tweets = len(test_y) - sum(test_y)
labels = 'Positive tweets', 'Negative tweets'
sizes = [positive_tweets, negative_tweets]
colors = ['gold', 'yellowgreen']
explode = (0.1, 0)
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.show()


#Histogram
alphab = ['Positive tweets', 'Negative tweets']
pos = np.arange(len(alphab))
width = 0.1
ax = plt.axes()
ax.set_xticks(pos + (width / 2))
ax.set_xticklabels(alphab)
plt.bar(pos, sizes, width, color='r')
plt.show()



# Word Cloud 
files = open("tweets.txt",'w')
with open('trumptweet.csv','rb') as csvfile:
    lineReader = csv.reader(csvfile)
    for row in lineReader:
        files.write(row[0])

file = open("tweets.txt",'r')
text = file.read()
file.close()
wordcloud = WordCloud(font_path='C:/Windows/Fonts/Calibri.ttf', relative_scaling = 0.25 ).generate(text)
plt.imshow(wordcloud)
plt.savefig('wordcloud.png')
plt.show()

# my_list = ["trump's", 'right', 'says', 'trump', 'just', 'people', "he's", 'donald', 'immigration', '@realdonaldtrump', 'american', 'election', 'plan', 'policy', 'president', '&amp;', '#trump', 'think', 'gop', 'like']
# unique_string=(" ").join(my_list)
# wordcloud = WordCloud(width = 1000, height = 500).generate(unique_string)
# plt.figure(figsize=(15,8))
# plt.imshow(wordcloud)
# plt.axis("off")
# plt.savefig("your_file_name"+".png", bbox_inches='tight')
# plt.show()
# plt.close()


km = KMeans(n_clusters = 2, n_init = 100) # try 100 different initial centroids
km.fit(test_X)
y_kmeans = km.predict(test_X)


cluster = []
cluster_stat = dict()
# Print tweets that belong to cluster 2
for idx, cls in enumerate(km.labels_):
    if cluster_stat.has_key(cls):
        cluster_stat[cls] += 1
    else:
        cluster_stat[cls] = 1
    #open('Cluster-{0}.txt'.format(cls), 'a').write(json.dumps(tweets[idx]) + '\r\n')

print 'basic information about the clusters that are generated by the k-means clustering algorithm: \r\n'
print 'total number of clusters: {0}\r\n'.format(len(cluster_stat))
for cls, count in cluster_stat.items():
    print 'cluster {0} has {1} tweets'.format(cls, count)


