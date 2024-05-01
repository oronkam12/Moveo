import torch
from transformers import GPT2Tokenizer, GPT2Model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
import io
import base64
from sklearn.feature_extraction.text import TfidfVectorizer
import json

class GPT2Kmeans:
    def __init__(self, num_clusters, all_claims_flat):
        self.num_clusters = num_clusters
        self.all_claims_flat = all_claims_flat
        self.sentence_embeddings = self.generate_gpt_embeddings(all_claims_flat)
        self.cluster_centers = None
    
    def generate_gpt_embeddings(self, sentences):
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2Model.from_pretrained('gpt2')
        
        def generate_gpt_embedding(sentence):
            inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            return embeddings
        
        return np.array([generate_gpt_embedding(sentence) for sentence in sentences])
    
    def cluster(self):
        kmeans = KMeans(n_clusters=self.num_clusters)
        kmeans.fit(self.sentence_embeddings)
        self.cluster_centers = kmeans.cluster_centers_
        cluster_labels = kmeans.labels_
        
        clustered_claims = [[] for _ in range(self.num_clusters)]
        for i, claim in enumerate(self.all_claims_flat):
            clustered_claims[cluster_labels[i]].append(claim)
        
        self.claims_labels = cluster_labels  # Store the labels for further use
        return clustered_claims
    
    def print_clustered_claims(self):
        clustered_claims = self.cluster()
        for i, cluster in enumerate(clustered_claims):
            print(f"Cluster {i+1}:")
            for claim in cluster:
                print(claim)
            print("\n")
    
    def cluster_density(self):
        clustered_claims = self.cluster()
        return [len(cluster) for cluster in clustered_claims]
    
    def distance_between_clusters(self):
        if self.cluster_centers is None:
            self.cluster()
        distances = pairwise_distances(self.cluster_centers)
        np.fill_diagonal(distances, np.nan)
        return distances

    def get_clusters_titles(self):
        if self.cluster_centers is None:
            self.cluster()  # Ensure clusters are ready

        kmeans_embeddings = KMeans(n_clusters=self.num_clusters)
        kmeans_embeddings.fit(self.sentence_embeddings)
        cluster_labels_embeddings = kmeans_embeddings.labels_
        cluster_centers_embeddings = kmeans_embeddings.cluster_centers_

        cluster_titles = {}  # Initialize a dictionary to store cluster titles

        for i in range(self.num_clusters):
            # Find closest sentence to cluster center
            distances_to_center = pairwise_distances(self.sentence_embeddings[cluster_labels_embeddings == i], 
                                                    [cluster_centers_embeddings[i]])
            closest_sentence_idx = np.argmin(distances_to_center)
            closest_sentence = self.all_claims_flat[np.where(cluster_labels_embeddings == i)[0][closest_sentence_idx]]
            
            # Extract the first few words as the title
            first_few_words = ' '.join(closest_sentence.split()[:3])
            
            # Store the cluster title in the dictionary
            cluster_titles[f"Cluster {i+1}"] = first_few_words

        # Convert the dictionary to JSON format and return
        return json.dumps(cluster_titles)



    def plot(self):
        svd_embeddings = TruncatedSVD(n_components=2)
        embeddings_2d = svd_embeddings.fit_transform(self.sentence_embeddings)
        
        fig, ax = plt.subplots(figsize=(15, 7))
        kmeans = KMeans(n_clusters=self.num_clusters)
        kmeans.fit(self.sentence_embeddings)
        cluster_labels = kmeans.labels_
        
        for cluster_num in range(self.num_clusters):
            cluster_points = embeddings_2d[cluster_labels == cluster_num]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_num+1}')

        ax.set_title('K-means Clustering of Claims (GPT-2 Embeddings)')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.legend()
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.read()).decode()
        plt.close(fig)
        
        return img_str
