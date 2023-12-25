# Text-Clustering-with-K-Means
Cluster texts into groups using the K-Means clustering algorithm.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# Assume you have a list of documents in 'corpus'

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(corpus)

true_k = 3  # Number of clusters
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()

for i in range(true_k):
    print(f"Cluster {i + 1}:")
    for ind in order_centroids[i, :10]:
        print(f"  {terms[ind]}")

print(f"\nAdjusted Rand Index: {adjusted_rand_score(labels, model.labels_)}")
