from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn import metrics
import logging

logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
dataset=fetch_20newsgroups(subset='all',shuffle=True,random_state=42)
labels=dataset.target
vectorizer=CountVectorizer(max_df=0.5,min_df=2,stop_words='english',ngram_range=(2,2))

X=vectorizer.fit_transform(dataset.data)

km=KMeans(verbose=0,n_clusters=50,n_init=1)
km.fit(X)

print "ARI: ", metrics.adjusted_rand_score(labels, km.labels_)
print "Silhouette Coefficient: ", metrics.silhouette_score(X, km.labels_, sample_size=1000)
