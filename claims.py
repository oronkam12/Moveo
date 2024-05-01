import requests
from bs4 import BeautifulSoup
import re
class Claims():

    def extract_claims_from_url(self,url):
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

    def extract_claims_from_multiple_urls(self,urls):
        all_claims = []
        for url in urls:
            claims = self.extract_claims_from_url(url)
            if claims:
                all_claims.append(claims)
        return all_claims

    # Example usage
    urls = [
        "https://patents.google.com/patent/GB2478972A/en?q=(phone)&oq=phone",
        "https://patents.google.com/patent/US9634864B2/en?oq=US9634864B2",
        "https://patents.google.com/patent/US9980046B2/en?oq=US9980046B2"
    ]  # Replace "URL1", "URL2", etc. with your actual URLs
    def get_claims(self):
        all_claims = self.extract_claims_from_multiple_urls(self.urls)
        return all_claims