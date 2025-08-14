# app.py (Frontend)

# Imports
import streamlit as st
import requests
import json
import matplotlib.pyplot as plt
import os

# Use an environment variable for flexibility
# API_URL = os.getenv("API_URL", "http://127.0.0.1:8000") # Localhost
API_URL = os.getenv("API_URL", "https://fri-professionals-cooler-track.trycloudflare.com") #cloudflared

# Format Frontend app UI
st.set_page_config(page_title="Transformers in Fake News Detector", layout="centered")

st.title("Fake News Detection")
st.markdown(
    """
    This application leverages a fine-tuned **RoBERTa-base** 
    transformer to predict the credibility of a news article.
    
    - **Prediction:** Predicted class.
        - **Fake Article (Label 0):** An article is likelier to contain **_mis-information_**. 
        - **Real Article (Label 1):** An article is likelier to contain **_credible-information_**.
    - **Confidence Scores:** Raw prediction probabilities (percentage).   
    - **LIME Explanation:** Visualize top-10 important features for the predicted class by the model.
    """
)
st.markdown("<br>", unsafe_allow_html=True)

# Input form
url = st.text_input("News Article URL", placeholder="https://example.com/news-article")

if st.button("Analyze Article"):
    if not url:
        st.warning("Please enter a valid URL.")
    else:
        with st.spinner("Scraping article and analyzing..."):
            try:
                # Call FastAPI backend
                response = requests.post(f"{API_URL}/predict", json={"url": url})
                if response.status_code == 200:
                    result = response.json()
                    st.session_state.result = result
                    if "explanation" in st.session_state:
                        del st.session_state.explanation
                else:
                    st.error(f"Error: {response.json()['detail']}")
            except requests.ConnectionError:
                st.error("Could not connect to the FastAPI server. Make sure it's running on `http://127.0.0.1:8000`.")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

if "result" in st.session_state:
    result = st.session_state.result
    st.success(f"**Article Title:** *{result['title']}*")
    st.markdown(f"**Prediction:** `{result['prediction']}`")
    st.markdown(f"**Confidence:** `{result['confidence']:.2%}`")

    # Show probabilities with progress bar
    st.progress(result["probabilities"]["real"])
    st.caption(f"🟥 Fake: {result['probabilities']['fake']:.2%} | 🟩 Real: {result['probabilities']['real']:.2%}")
    st.markdown("<br>", unsafe_allow_html=True)

    if result["pred_label"] is not None and result["pred_label"] in [0, 1]:
        if st.button("Generate LIME Explanation"):
            with st.spinner("Generating explanation..."):
                try:
                    explain_response = requests.post(
                        f"{API_URL}/explain",
                        json={
                            "processed_text": result["processed_text"],
                            "pred_label": result["pred_label"]
                        }
                    )
                    if explain_response.status_code == 200:
                        st.session_state.explanation = explain_response.json()["explanation"]
                    else:
                        st.error(f"Error: {explain_response.json()['detail']}")
                except Exception as e:
                    st.error(f"Error generating explanation: {e}")

    if "explanation" in st.session_state:
        explanation = st.session_state.explanation
        if isinstance(explanation, list) and explanation:
            st.markdown("### LIME Explanation")

            # Unzip words and weights
            words, weights = zip(*explanation)

            # Color positive weights green and negative weights red
            colors = ["green" if w > 0 else "red" for w in weights]

            # Create horizontal bar plot
            fig, ax = plt.subplots(figsize=(10, 8))
            y_pos = range(len(words))
            ax.barh(y_pos, weights, color=colors)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(words)
            ax.invert_yaxis() 
            ax.set_xlabel("Weight")
            ax.set_title(f"Top-10 Important Features for {result['prediction']} Prediction")
            ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
            st.pyplot(fig)

        else:
            st.info("No explanation available.")