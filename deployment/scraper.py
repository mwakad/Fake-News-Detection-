# scraper.py
import requests
from bs4 import BeautifulSoup
import spacy
import re

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Please install spaCy English model: python -m spacy download en_core_web_sm")
    raise

def clean_text(text):
    if not text:
        return ""
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_article(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'lxml')

        # Extract title
        title_tag = soup.find('title')
        title = title_tag.get_text(strip=True) if title_tag else "No Title Found"

        # Remove scripts, styles
        for script in soup(["script", "style"]):
            script.decompose()

        # Try common article tags
        article_tag = soup.find('article')
        if article_tag:
            text = article_tag.get_text(separator=' ', strip=True)
        else:
            # Fallback: use body
            text = soup.body.get_text(separator=' ', strip=True) if soup.body else ""

        # Clean text
        cleaned_text = clean_text(text)
        cleaned_title = clean_text(title)

        # Use spaCy for sentence filtering (optional preprocessing like in notebook)
        doc = nlp(cleaned_text)
        sentences = [sent.text for sent in doc.sents if len(sent.text.strip()) > 10]
        spacy_text = " ".join(sentences)

        return {
            "title": cleaned_title,
            "body": cleaned_text,
            "spacy_text": spacy_text
        }

    except Exception as e:
        return {
            "error": f"Error scraping article: {str(e)}"
        }