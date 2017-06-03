from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn import metrics
import logging

logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
categories = ['alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]

dataset=fetch_20newsgroups(subset='all',shuffle=True,random_state=42,categories=categories)
labels=dataset.target
vectorizer=TfidfVectorizer(max_df=1.0,min_df=10,stop_words='english',ngram_range=(1,2))

X=vectorizer.fit_transform(dataset.data)

km=KMeans(verbose=0,n_clusters=50,n_init=5)
km.fit(X)

print "ARI: ", metrics.adjusted_rand_score(labels, km.labels_)
print "Silhouette Coefficient: ", metrics.silhouette_score(X, km.labels_, sample_size=1000)
