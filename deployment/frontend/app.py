# app.py (Frontend)

import streamlit as st
import requests
import matplotlib.pyplot as plt
import os

# set API_URL liknk
# API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")  # localhost
API_URL = os.getenv("API_URL", "https://f4cd6746181f.ngrok-free.app") # ngrok tunnel

# App configuration
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

# Input
url = st.text_input("News Article URL", placeholder="https://example.com/news-article")

if st.button("Analyze Article"):
    if not url:
        st.warning("Please enter a valid URL.")
    else:
        with st.spinner("Scraping article and analyzing..."):
            try:
                response = requests.post(f"{API_URL}/predict", json={"url": url})
                if response.status_code == 200:
                    result = response.json()
                    st.session_state.result = result
                    if "explanation" in st.session_state:
                        del st.session_state.explanation
                else:
                    error_detail = response.json().get("detail", "Unknown error")
                    st.error(f"Error: {error_detail}")
            except requests.ConnectionError:
                st.error("Could not connect to the FastAPI server. Is it running?")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

# Display prediction
if "result" in st.session_state:
    result = st.session_state.result
    st.success(f"**Article Title:** *{result['title']}*")
    st.markdown(f"**Prediction:** `{result['prediction']}`")
    st.markdown(f"**Confidence:** `{result['confidence']:.2%}`")

    # Progress bar
    st.progress(result["probabilities"]["real"])
    st.caption(f"🟥 Fake: {result['probabilities']['fake']:.2%} | 🟩 Real: {result['probabilities']['real']:.2%}")
    st.markdown("<br>", unsafe_allow_html=True)

    # Generate explanation only if valid label
    if result["pred_label"] in [0, 1]:
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
                        error_detail = explain_response.json().get("detail", "Unknown error")
                        st.error(f"Error: {error_detail}")
                except Exception as e:
                    st.error(f"Error generating explanation: {e}")
            
    # Display explanation
    if "explanation" in st.session_state:
        explanation = st.session_state.explanation
        if isinstance(explanation, list) and len(explanation) > 0:
            st.markdown("### LIME Explanation")

            if isinstance(explanation[0], dict):
                words = [item["word"] for item in explanation]
                weights = [item["weight"] for item in explanation]
            elif isinstance(explanation[0], (list, tuple)):
                words = [item[0] for item in explanation]
                weights = [item[1] for item in explanation]
            else:
                st.error("Unexpected explanation format.")
                st.json(explanation)
                st.stop()

            colors = ["green" if w > 0 else "red" for w in weights]

            fig, ax = plt.subplots(figsize=(10, 6))
            y_pos = range(len(words))
            ax.barh(y_pos, weights, color=colors)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(words)
            ax.invert_yaxis()
            ax.set_xlabel("Contribution to Prediction")
            ax.set_title(f"Top-10 Features for '{result['prediction']}'")
            ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
            st.pyplot(fig)
        else:
            st.info("No explanation available.")