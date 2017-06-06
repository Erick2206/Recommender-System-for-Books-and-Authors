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

outfile="hierarchical_out.txt"

categories =[
    'alt.atheism',
    'rec.sport.baseball',
    'comp.graphics',
    'sci.space',
    'rec.sport.hockey',
    'comp.windows.x'
]

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

hc=AgglomerativeClustering(n_clusters=true_k, linkage='average', affinity='euclidean')
hc.fit(X.toarray())

ari=metrics.adjusted_rand_score(labels, hc.labels_)
silh_coeff=metrics.silhouette_score(X, hc.labels_, sample_size=1000)

print("Adjusted Rand-Index: %0.3f" % metrics.adjusted_rand_score(labels, hc.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, hc.labels_, sample_size=1000))

with open(outfile,'a') as out:
    out.write("ARI: %.3f\n" % metrics.adjusted_rand_score(labels, hc.labels_))
    out.write("Silhouette Co-efficient: %.3f\n" % silh_coeff)
    out.write("\n")
