from numpy import asarray,squeeze,mean
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups

class KMeans:

    def get_new_centers():
        TODO

    def get_cluster_points():
        TODO

    def has_converged(mu,oldmu):
        TODO    

    def fit(X,n_cluster):
        oldmu=random.sample(X,n_cluster)

        while not has_converged(mu,oldmu):
            oldmu=mu
            clusters,clusters_with_point=self.get_cluster_points(X,mu)
            mu=self.get_new_centers(oldmu,clusters)

        return (mu,clusters,clusters_with_point)


if __name__="__main__":
    dataset=fetch_20newsgroups()
    vectorizer=TfidfVectorizer(min_df=5,max_df=0.5,stop_words='english')
    matrix=vectorizer.fit_transform(dataset.data)
    dense_matrix=matrix.todense()

    output=[0 for i in range(len(dense_matrix))]
    km=KMeans(n_cluster)
    mu,clusters,clusters_with_point=km.fit(dense_matrix)
