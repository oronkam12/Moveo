import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.mixture import GaussianMixture
from sklearn.feature_extraction.text import TfidfVectorizer
import io
import base64
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json

class GMMClustering:
    def __init__(self, num_clusters, all_claims_lower, all_claims_flat):
        self.num_clusters = num_clusters
        self.all_claims_lower = all_claims_lower
        self.all_claims_flat = all_claims_flat
        self.tokenizer = AutoTokenizer.from_pretrained("czearing/article-title-generator")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("czearing/article-title-generator")

    def cluster(self):
        # Convert claims to TF-IDF vectors
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(self.all_claims_lower)
        
        # Reduce the dimensions of TF-IDF matrix for clustering
        svd_tfidf = TruncatedSVD(n_components=2)
        tfidf_matrix_2d = svd_tfidf.fit_transform(tfidf_matrix)
        
        # Perform Gaussian Mixture Model (GMM) clustering
        gmm = GaussianMixture(n_components=self.num_clusters)
        cluster_labels = gmm.fit_predict(tfidf_matrix_2d)
        self.cluster_centers = gmm.means_  # Store cluster centers
        
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
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(self.all_claims_lower)
        return tfidf_matrix
    
    def cluster_density(self):
        # Calculate number of points in each cluster
        clustered_claims = self.cluster()
        cluster_sizes = [len(cluster) for cluster in clustered_claims]
        return cluster_sizes
    
    def distance_between_clusters(self):
        # Calculate pairwise distances between cluster centers
        if self.cluster_centers is None:
            self.cluster()  # Ensure clustering is performed
        distances = pairwise_distances(self.cluster_centers, metric='euclidean')
        np.fill_diagonal(distances, np.nan)  # Set diagonal elements to nan
        return distances


    def plot(self):
        # Reduce the dimensions of TF-IDF matrix for plotting
        svd_tfidf = TruncatedSVD(n_components=2)
        
        # Get TF-IDF matrix
        tfidf_matrix = self.tf_idf()
        
        # Reduce dimensions
        tfidf_matrix_2d = svd_tfidf.fit_transform(tfidf_matrix)
        
        # Perform Gaussian Mixture Model (GMM) clustering
        gmm = GaussianMixture(n_components=self.num_clusters)
        cluster_labels = gmm.fit_predict(tfidf_matrix_2d)
        
        # Plot clusters for TF-IDF clustering
        fig, ax = plt.subplots(figsize=(15, 7))
        
        for cluster_num in range(self.num_clusters):
            cluster_points = tfidf_matrix_2d[cluster_labels == cluster_num]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_num+1}')

        ax.set_title('Gaussian Mixture Model Clustering of Claims (TF-IDF)')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.legend()
        
        # Save the plot as a PNG image in memory
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        
        # Encode the image in base64 format
        img_str = base64.b64encode(img_buffer.read()).decode()
        
        # Close the figure to prevent memory leaks
        plt.close(fig)
        
        return img_str
    
    def get_clusters_titles(self):
        # Cluster the claims
        clustered_claims = self.cluster()

        # Initialize a dictionary to store titles for each cluster
        cluster_titles = {}

        for i, cluster in enumerate(clustered_claims, start=1):
            # Concatenate all claims in the cluster to form a single string
            cluster_text = ". ".join(cluster)

            # Tokenize the cluster text
            inputs = self.tokenizer.encode("summarize: " + cluster_text, return_tensors="pt", max_length=1024, truncation=True)

            # Generate title using the model
            summary_ids = self.model.generate(inputs, max_length=80, min_length=10, length_penalty=5.0, num_beams=4, early_stopping=True)
            cluster_title = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            # Store the cluster title in the dictionary
            cluster_titles[f"Cluster {i}"] = cluster_title

        # Convert the dictionary to JSON format and return
        return json.dumps(cluster_titles)

