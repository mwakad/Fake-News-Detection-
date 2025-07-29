import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
import os

# Optional: Set user-agent to mimic a browser
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

def fetch_article_text(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            paragraphs = soup.find_all('p')
            article_text = " ".join([p.get_text() for p in paragraphs])
            return article_text
        else:
            print(f"[{response.status_code}] Failed to retrieve: {url}")
            return None
    except Exception as e:
        print(f"Exception for {url}: {e}")
        return None

def scrape_dataset(file_path, output_file=None):
    df = pd.read_csv(file_path)
    tqdm.pandas(desc=f"Scraping {os.path.basename(file_path)}")
    df['extracted_article_text'] = df['url'].progress_apply(fetch_article_text)
    
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"Saved scraped data to: {output_file}")
    return df

def main():
    datasets = [
        ("./dataset/gossipcop_fake.csv", "./dataset/gossipcop_fake_scraped.csv"),
        ("./dataset/gossipcop_real.csv", "./dataset/gossipcop_real_scraped.csv"),
        ("./dataset/politifact_fake.csv", "./dataset/politifact_fake_scraped.csv"),
        ("./dataset/politifact_real.csv", "./dataset/politifact_real_scraped.csv"),
    ]

    for input_file, output_file in datasets:
        if os.path.exists(input_file):
            print(f"\nProcessing {input_file}...")
            scrape_dataset(input_file, output_file)
            time.sleep(2)  # Politeness delay
        else:
            print(f"{input_file} not found.")

if __name__ == "__main__":
    main()
