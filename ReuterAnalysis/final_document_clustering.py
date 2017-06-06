from __future__ import print_function

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.cluster import KMeans, MiniBatchKMeans

import logging
from optparse import OptionParser
import sys
from time import time

import numpy as np

categories =[
    'alt.atheism',
    'rec.sport.baseball',
    'comp.graphics',
    'sci.space',
    'rec.sport.hockey',
    'comp.windows.x'
]

categories=['comp.sys.mac.hardware',
    'misc.forsale', 'talk.politics.mideast', 'rec.autos']

dataset = fetch_20newsgroups(subset='train', categories=categories,
                             shuffle=True, random_state=42)

labels = dataset.target
true_k = np.unique(labels).shape[0]
print(true_k)

print("Extracting features from the training dataset using a sparse vectorizer")
print("Running tfidf")
vectorizer = TfidfVectorizer(max_df=0.5,max_features=10000,
                             min_df=2, stop_words='english',
                             use_idf=False,norm='l2')
X = vectorizer.fit_transform(dataset.data)
print(X.shape)
print("n_samples: %d, n_features: %d" % X.shape)
print()

km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                     init_size=1000, batch_size=1000, verbose=True)

km=KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)

print("Clustering sparse data with %s" % km)
km.fit(X)

print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, km.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, km.labels_, sample_size=1000))

print()

if True:
    print("Top terms per cluster:")
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    terms = vectorizer.get_feature_names()
    for i in range(true_k):
        print("Cluster %d:" % i, end='')
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind], end='')
        print()
