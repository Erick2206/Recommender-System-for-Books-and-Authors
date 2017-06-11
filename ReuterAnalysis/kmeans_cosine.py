import random
from numpy import asarray,squeeze,mean
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups

class KMeans:
    def __init__(self,n_cluster,n_points):
        self.n_cluster=n_cluster
        self.predicted_label=[0 for i in range(n_points)]

    def get_cluster_points(self,X,mu):
        clusters={}
        clusters_with_point={}
        for j,x in enumerate(X):
            bestmukey=max([(i[0], cosine_similarity(x.reshape(1,-1), mu[i[0]].reshape(1,-1)))\
            for i in enumerate(mu)], key=lambda t: t[1])[0]

        self.predicted_label[j]=bestmukey
        clusters_with_point.setdefault(bestmukey,[])
        clusters_with_point[bestmukey].append(j)
        clusters.setdefault(bestmukey,[])
        clusters[bestmukey].append(x)

        return clusters,clusters_with_point

    def get_new_centers(self,mu,clusters):
        returnmu=[]
        keys=sorted(clusters.keys())

        for k in keys:
            returnmu.append(mean(clusters[k],axis=0))

        return returnmu

    def has_converged(self,mu,oldmu):
        mu=squeeze(asarray(mu))
        oldmu=squeeze(asarray(oldmu))
        print mu
        print "1"
        print type(mu)
        t1=set([tuple(x) for x in mu])
        t2=set([tuple(x) for x in oldmu])
        return t1==t2
#        return (set([tuple(a) for a in mu])==set([tuple(a) for a in oldmu]))

    def fit(self,X,n_cluster):
        oldmu=random.sample(X,n_cluster)
        mu=random.sample(X,n_cluster)
        while not self.has_converged(mu,oldmu):
            oldmu=mu
            clusters,clusters_with_point=self.get_cluster_points(X,mu)
            mu=self.get_new_centers(oldmu,clusters)

        return mu,clusters,clusters_with_point


if __name__=="__main__":
    categories=['comp.sys.mac.hardware',
        'misc.forsale', 'talk.politics.mideast', 'rec.autos']
    dataset=fetch_20newsgroups(subset='train', categories=categories,
                                 shuffle=True, random_state=42)
    vectorizer=TfidfVectorizer(min_df=5,max_df=0.5,stop_words='english')
    matrix=vectorizer.fit_transform(dataset.data)
    dense_matrix=matrix.todense()
    n_cluster=4
    km=KMeans(n_cluster,len(dense_matrix))
    mu,clusters,clusters_with_point=km.fit(dense_matrix,n_cluster)
