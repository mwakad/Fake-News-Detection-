# streamlit_app.py
import streamlit as st
import requests
import json

st.set_page_config(page_title="🔍 Fake News Detector", layout="centered")

st.title("Fake News Detector")
st.markdown("This application leverages transfer learning to finetune the **RoBERTa-base** transformer for detecting whether a **News Article** contains **_misinformation_** (**class 0**) or it is **_credible_** (**class 1**)")
st.markdown("Enter a news article URL")

# Input form
url = st.text_input("News Article URL", placeholder="https://example.com/news-article")

if st.button("Analyze Article"):
    if not url:
        st.warning("Please enter a valid URL.")
    else:
        with st.spinner("Scraping article and analyzing..."):
            try:
                # Call FastAPI backend
                response = requests.post("http://127.0.0.1:8000/predict", json={"url": url})
                if response.status_code == 200:
                    result = response.json()

                    st.success(f" Article analyzed: *{result['title']}*")
                    st.write(f"**Prediction**: {result['prediction']}")
                    st.write(f"**Confidence**: {result['confidence']:.2%}")

                    # Show probabilities
                    st.progress(result["probabilities"]["real"])
                    st.caption(f"Fake: {result['probabilities']['fake']:.2%} | Real: {result['probabilities']['real']:.2%}")

                else:
                    error = response.json().get("detail", "Unknown error")
                    st.error(f" Error: {error}")

            except requests.ConnectionError:
                st.error(" Could not connect to the FastAPI server. Make sure it's running on `http://127.0.0.1:8000`.")
            except Exception as e:
                st.error(f" Unexpected error: {e}")
                

                
# import streamlit as st
# import requests
# import json

# st.set_page_config(page_title=" Fake News Detector", layout="centered")

# st.title("Fake News Detector")
# st.markdown("This application leverages transfer learning to finetune the **RoBERTa-base** transformer for detecting whether a **News Article** contains **_misinformation_** (**class 0**) or it is **_credible_** (**class 1**)")
# st.markdown("Enter a news article URL")

# # Input form
# url = st.text_input("News Article URL", placeholder="https://example.com/news-article")

# if st.button("Analyze Article"):
#     if not url:
#         st.warning("Please enter a valid URL.")
#     else:
#         with st.spinner("Scraping article and analyzing..."):
#             try:
#                 # Call FastAPI backend
#                 response = requests.post("http://127.0.0.1:8000/predict", json={"url": url})
#                 if response.status_code == 200:
#                     result = response.json()

#                     st.success(f" Article analyzed: *{result['title']}*")
                    
#                     # Display prediction only if confidence is above 80%
#                     if result['probabilities']['real'] > 0.8:
#                         st.write(f"**Prediction**: Real (Credible)")
#                         st.write(f"**Confidence**: {result['probabilities']['real']:.2%}")
#                     else:
#                         st.write("**Prediction**: Not confident enough to classify as Real (Credible).")
#                         st.write(f"**Confidence**: {result['probabilities']['real']:.2%} (must be above 80%)")

#                     # Show probabilities
#                     st.progress(result["probabilities"]["real"])
#                     st.caption(f"Fake: {result['probabilities']['fake']:.2%} | Real: {result['probabilities']['real']:.2%}")

#                 else:
#                     error = response.json().get("detail", "Unknown error")
#                     st.error(f" Error: {error}")

#             except requests.ConnectionError:
#                 st.error("Could not connect to the FastAPI server. Make sure it's running on `http://127.0.0.1:8000`.")
#             except Exception as e:
#                 st.error(f"Unexpected error: {e}")