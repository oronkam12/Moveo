import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import io
import base64
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import json

class Kmeans:
    def __init__(self, num_clusters, all_claims_lower, all_claims_flat):
        self.num_clusters = num_clusters
        self.all_claims_lower = all_claims_lower
        self.all_claims_flat = all_claims_flat
        self.cluster_centers = None
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
    def cluster(self):
        # Convert claims to TF-IDF vectors
        tfidf_matrix = self.vectorizer.fit_transform(self.all_claims_lower)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=self.num_clusters)
        kmeans.fit(tfidf_matrix)
        
        # Get cluster labels
        cluster_labels = kmeans.labels_
        
        # Get cluster centers
        self.cluster_centers = kmeans.cluster_centers_
        
        # Assign claims to clusters
        clustered_claims = [[] for _ in range(self.num_clusters)]
        for i, claim in enumerate(self.all_claims_flat):
            clustered_claims[cluster_labels[i]].append(claim)
        
        return clustered_claims
    
    def print_clustered_claims(self):
        clustered_claims = self.cluster()
        for i, cluster in enumerate(clustered_claims):
            print(f"Cluster {i+1}:")
            for claim in cluster:
                print(claim)
            print("\n")

    def tf_idf(self):
        tfidf_matrix = self.vectorizer.fit_transform(self.all_claims_lower)
        return tfidf_matrix
    
    def plot(self):
        # Reduce dimensions
        svd_tfidf = TruncatedSVD(n_components=2)
        tfidf_matrix = self.tf_idf()
        tfidf_matrix_2d = svd_tfidf.fit_transform(tfidf_matrix)
        
        # Plot clusters
        fig, ax = plt.subplots(figsize=(15, 7))
        kmeans = KMeans(n_clusters=self.num_clusters)
        kmeans.fit(tfidf_matrix)
        cluster_labels = kmeans.labels_
        
        for cluster_num in range(self.num_clusters):
            cluster_points = tfidf_matrix_2d[cluster_labels == cluster_num]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_num+1}')

        ax.set_title('K-means Clustering of Claims (TF-IDF)')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.legend()
        
        # Save plot as PNG image in memory
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.read()).decode()
        plt.close(fig)
        
        return img_str
    
    def cluster_density(self):
        clustered_claims = self.cluster()
        cluster_sizes = [len(cluster) for cluster in clustered_claims]
        return cluster_sizes
    
    def distance_between_clusters(self):
        if self.cluster_centers is None:
            self.cluster()
        distances = pairwise_distances(self.cluster_centers, metric='euclidean')
        np.fill_diagonal(distances, np.nan)
        return distances
    
    def get_clusters_titles(self):
        # Initialize a dictionary to store cluster titles
        cluster_titles = {}
        
        # Extract features corresponding to cluster centers
        terms = self.vectorizer.get_feature_names_out()
        sorted_centroids = self.cluster_centers.argsort()[:, ::-1]
        
        for i in range(self.num_clusters):
            # Get the index of the feature with the highest frequency in the cluster
            top_feature_idx = sorted_centroids[i, 0]
            # Get the term associated with the top feature
            top_term = terms[top_feature_idx]
            # Extract the first few words from the term as the title
            first_few_words = ' '.join(top_term.split()[:3])
            # Store the cluster title in the dictionary
            cluster_titles[f"Cluster {i+1}"] = first_few_words
        
        # Convert the dictionary to JSON format
        return json.dumps(cluster_titles)



