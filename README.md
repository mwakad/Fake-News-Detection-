# Fake News Detection

## Project Summary
This project applies advanced Natural Language Processing (NLP) and machine learning techniques to classify news articles as either **fake** (`Class 0`) or **real** (`Class 1`). It incorporates traditional ML (**_Logistic regression_**), ensemble learning (**_XGBoost_**), neural networks (**_LSTM_**), and transformer-based deep learning (**_RoBERTa-base_**) models. **Macro-averaged recall** is set as the success criteria because the training dataset is imbalanced and the project prioritizes to categorize fake news articles based on respective text.

Built with **scikit-learn**, **XGBoost**, **TensorFlow/Keras**, **Torch**, **Hugging Face Transformers**, **Streamlit**, **FastAPI**, and **Docker**.

| Task                         | Model Used                   | Deployment File |
|------------------------------|------------------------------|-----------------|
| Fake News Classification     | Finetuned RoBERTa (SMOTE)    | main.py         |
| Interface                    | Streamlit + FastAPI          | streamlit_app.py|


## Data Understanding
The dataset consists of **15,116 news articles** with labels indicating whether each article is *fake* (`Class 0`) or *real* (`Class 1`).  

Preprocessing steps included:
- Tokenization and lemmatization using spaCy.
- Stopword removal and punctuation stripping.
- Domain extraction.

The dataset is **imbalanced**, with a higher proportion of real articles compared to fake ones. This informed the choice of macro-averaged recall as the primary evaluation metric.
<img width="2370" height="1765" alt="class-distribuctions" src="https://github.com/user-attachments/assets/4c0d0fef-a71e-4fba-9a0d-ef8731eee539" />


## Problem Statement
The rapid spread of fake news online erodes trust, misinforms the public, and can have serious political, social, and health consequences.  
Manual fact-checking is slow and infeasible at scale. The goal is to build an **automated, accurate, and interpretable** detection system to identify fake news in real time.

## Project Objectives
### **Objective 1:** Perform Extraploratory Data Analysis (EDA)

**Top-5 Domains for Fake and Real News Articles**
<img width="3590" height="1169" alt="top-5-domains-for-fake-and-real-news" src="https://github.com/user-attachments/assets/80ecdace-432b-4130-8837-f9b4e8fcf64b" />

**Top-10 Bigrams for Fake and Real News Articles**
<img width="4468" height="2365" alt="top-10-bigrams" src="https://github.com/user-attachments/assets/ae007b2f-0a1b-48af-9e14-ad2c229443d2" />

**Top-10 Trigrams for Fake and Real News Articles**
<img width="4468" height="2365" alt="top-10-trigrams" src="https://github.com/user-attachments/assets/53622ac7-236e-46ac-b6fb-f5623bd501a2" />


### **Objective 2:** Build a baseline Logistic Regression model
A baseline Logistic Regression model was built with imbalanced data and two balancing strategies:
- **SMOTE** oversampling for the minority class
- **Random Undersampling (RUS)** for the majority class
  - **Best configuration:** SMOTE-balanced data with **macro-averaged recall = 0.7415**.

### **Objective 3:** Build an **XGBoost** model, **LSTM** neural network, and Finetune the **RoBERTa-base Transformer**
- An XGBoost classifier:  
  - **Best configuration:** RUS-balanced data with **macro-averaged recall = 0.7427**.
- An LSTM neural network:
  - **Best configuration:** SMOTE-balanced data with **macro-averaged recall = 0.7104**.
- Finetune the RoBERTa-base transformer:  
  - **Best configuration:** SMOTE-balanced data with **macro-averaged recall = 0.7660**

### **Objective 4:** Interpret best performing model using the LIME library

| MODEL                            | MACRO-AVG RECALL | RECALL (Fake) | AUC   |
|----------------------------------|------------------|---------------|-------|
| Logistic Regression-SMOTE        | 0.7145           | 0.7029        | 0.815 |
| XGBoost-RUS                      | 0.7427           | 0.7088        | 0.822 |
| LSTM-SMOTE                       | 0.7104           | 0.5784        | 0.792 |
| Finetuned RoBERTa-SMOTE          | **0.7660**       | **0.6686**    | **0.843** |

- The Finetuned RoBERTa model using SMOTE-balanced dataset is the best performing alternative because it achieves the highest scores for Macro-averaged recall (0.7660).

- This is corroborated by the confusion matrix, which reports the highest true positives for accurate fake news predictions (682).
<img width="3570" height="1466" alt="roberta-confusion_matrices" src="https://github.com/user-attachments/assets/c547198d-e201-4a96-b88b-676a9a0b1398" />
  
- The ROC-AUC curve plots reinforces this decision since with the finetuned transformer with SMOTE-balanced data achieves the highest AUC (0.843). 

<img width="2970" height="2066" alt="roberta-roc-auc" src="https://github.com/user-attachments/assets/efc7e878-2292-4a74-a56f-79cee8a3ff07" />

  - **LIME** was used to explain the **_finetuned RoBERTa-base Transformer_** model's predictions at the feature level.
  - Visualizations highlight the top 10 features supporting the prediction (green = `predicted class`, red = `alternative prediction`).
<img width="2969" height="1765" alt="lime_explanation_barplot" src="https://github.com/user-attachments/assets/663c9b2c-09da-4307-a1fc-3562c9bae87e" />

### **Objective 5:** Deploy selected model using FastAPI and Streamlit
The **Finetuned RoBERTa-base transformer with SMOTE-balanced dataset** is deployed using FastAPI, Streamlit, and Docker.
  - **Back-end :** The FastAPI app (**_main.py_**) is responsible for model inference. 
  - **Front-end :** The Streamlit app (**_app.py_**) provides an interactive frontend web interface.
  - **Docker :** Containerizes the **_Front-end_** and the **_Back-end_**.

## Conclusion
The **Finetuned RoBERTa with SMOTE balancing** achieved the highest macro-averaged recall (0.7660) and demonstrated strong capability in detecting fake news articles.  
Its contextual language understanding from pretraining, combined with balanced fine-tuning, improved recall on the minority class without sacrificing generalization.

## Recommendations
- Incoporate more training data expecially class 0 (Fake News) entries to improve recall.
- Leverage pretrained embeddings when finetuning the RoBERTa transformer to reduce training time.
- Experiment finetuning other NLP transformers such as DistilBERT and DeBERTa.

## Next Steps
- Deploy the containerized finetuned RoBERTa model via AWS SageMaker for low-latency inference.
- Develop browser extensions and social media plugins using the model’s API for real-time credibility scoring.
- Incorporate automation frameworks to flag low-confidence predictions for human review.

## Installation & Running the App
1. Clone the repository  
`git clone https://github.com/yourusername/fake-news-detector.git`
2. Install dependencies
`pip install -r requirements.txt`
3. Run FastAPI backend
`uvicorn main:app --reload`
4. Run Streamlit frontend
`streamlit run streamlit_app.py`
