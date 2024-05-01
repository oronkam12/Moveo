import requests
from bs4 import BeautifulSoup
import re
# Flask app
from flask import Flask, render_template, request
from GMMClustering import GMMClustering
from GPT2Kmeans import GPT2Kmeans
from Kmeans import Kmeans
import json

def extract_claims_from_url(url):
    # Make an HTTP GET request to the URL
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML using Beautiful Soup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all claim elements
        claims = soup.find_all(class_='claim-text')
        
        # Extract text from each claim and remove numbering part
        claim_texts = []
        current_claim_number = None
        current_claim_text = ""
        for claim in claims:
            text = claim.get_text(strip=True)  # Use strip=True to remove leading and trailing whitespace
            # Check if the claim is not a dependent claim
            if 'dependent' not in claim.get('class', []):
                claim_number_match = re.match(r'^(\d+)\.', text)
                if claim_number_match:
                    # If current claim text is not empty, append it to claim_texts
                    if current_claim_text:
                        # Remove numbers from the start until the first "."
                        current_claim_text = re.sub(r'^\d+(\.)?', '', current_claim_text)
                        claim_texts.append(current_claim_text.strip())
                    # Update current claim number and reset current claim text
                    current_claim_number = int(claim_number_match.group(1))
                    current_claim_text = text
                else:
                    # Concatenate the text to the current claim text
                    current_claim_text += " " + text
        # Append the last claim text
        if current_claim_text:
            # Remove numbers from the start until the first "."
            current_claim_text = re.sub(r'^\d+(\.)?', '', current_claim_text)
            claim_texts.append(current_claim_text.strip())
        
        # Return the extracted claim texts
        return claim_texts
    else:
        # Print an error message if the request was not successful
        print(f"Failed to fetch HTML content from {url}. Status code: {response.status_code}")
        return None

def extract_claims_from_multiple_urls(urls):
    all_claims = []
    for url in urls:
        claims = extract_claims_from_url(url)
        if claims:
            all_claims.extend(claims)  # Extend the list with extracted claims
    return all_claims



app = Flask(__name__)

# Example URLs
urls = [
    "https://patents.google.com/patent/GB2478972A/en?q=(phone)&oq=phone",
    "https://patents.google.com/patent/US9634864B2/en?oq=US9634864B2",
    "https://patents.google.com/patent/US9980046B2/en?oq=US9980046B2"
]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        num_clusters = int(request.form['num_clusters'])
        method = request.form['method']
        
        # Extract claims from URLs
        all_claims = extract_claims_from_multiple_urls(urls)
        
        # Initialize variables to store clustered claims, image string, and cluster titles
        clustered_claims = None
        img_str = None
        cluster_titles = None

        if method == 'kmeans':
            kmeans = Kmeans(num_clusters, all_claims, all_claims)
            clustered_claims = kmeans.cluster()
            img_str = kmeans.plot()
            cluster_titles = kmeans.get_clusters_titles()  # Get cluster titles as JSON
        elif method == 'gmm':
            gmm = GMMClustering(num_clusters, all_claims, all_claims)
            clustered_claims = gmm.cluster()
            img_str = gmm.plot()
            cluster_titles = gmm.get_clusters_titles()  # Get cluster titles as JSON
        elif method == 'gpt-kmeans':
            gpt_kmeans = GPT2Kmeans(num_clusters, all_claims)
            clustered_claims = gpt_kmeans.cluster()
            img_str = gpt_kmeans.plot()
            cluster_titles = gpt_kmeans.get_clusters_titles()  # Get cluster titles as JSON
        
        # Convert cluster_titles to dictionary if it's a string
        if isinstance(cluster_titles, str):
            cluster_titles = json.loads(cluster_titles)
        
        return render_template('index.html', clustered_claims=clustered_claims, img_str=img_str, cluster_titles=cluster_titles)
    
    return render_template('index.html', clustered_claims=None)

if __name__ == '__main__':
    app.run(debug=False)
