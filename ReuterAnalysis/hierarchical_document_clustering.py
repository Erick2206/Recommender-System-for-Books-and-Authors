from __future__ import print_function

from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics

import logging
from optparse import OptionParser
import sys
from time import time

import numpy as np


categories = [
    'alt.atheism',
    'rec.sport.baseball',
    'comp.graphics',
    'sci.space',
]
dataset = fetch_20newsgroups(subset='all', categories=categories,
                             shuffle=True, random_state=42)

labels = dataset.target
true_k = np.unique(labels).shape[0]
print(true_k)

print("Extracting features from the training dataset using a sparse vectorizer")
print("Running tfidf")
vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000,
                             min_df=2, stop_words='english',
                             use_idf=True)
X = vectorizer.fit_transform(dataset.data)
print(X.shape)

print("n_samples: %d, n_features: %d" % X.shape)
print()

hc=AgglomerativeClustering(n_clusters=20, linkage='complete', affinity='cosine')
hc.fit(X.toarray())

print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, hc.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, hc.labels_, sample_size=1000))
