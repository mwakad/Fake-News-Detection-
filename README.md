# Fake News Detection

## Project Summary
This project applies advanced Natural Language Processing (NLP) and machine learning techniques to classify news articles as either **fake** (`Class 0`) or **real** (`Class 1`). It incorporates traditional ML (**_Logistic regression_**), ensemble learning (**_XGBoost_**), neural networks (**_LSTM_**), and transformer-based deep learning (**_RoBERTa-base_**) models. **Recall (Fake)** is set as the success criteria because the project prioritizes to categorize fake news articles based on respective text.

Built with **scikit-learn**, **XGBoost**, **TensorFlow/Keras**, **Torch**, **Hugging Face Transformers**, **Streamlit**, **FastAPI**, and **Docker**.

| Task                         | Model Used                   | Deployment File |
|------------------------------|------------------------------|-----------------|
| Fake News Classification     | Finetuned RoBERTa (RUS)    | roberta_rus      |
| Interface                    | FastAPI + Streamlit          | main.py and app.py|


## Data Understanding
The dataset consists of **22,465** news articles with labels indicating whether each article is **fake** (`Class 0`) or **real** (`Class 1`).  

Preprocessing steps included:
- Tokenization and lemmatization using spaCy.
- Stopword removal and punctuation stripping.
- Domain extraction.

The dataset is **imbalanced**, with a higher proportion of real articles (**_11,699_** compared to fake articles (**_10,766_**). 
<img width="2370" height="1766" alt="class-distribuctions" src="https://github.com/user-attachments/assets/fa5ddef5-9825-48ff-bf38-c3e8650fbeac" />



## Problem Statement
The rapid spread of fake news online erodes trust, misinforms the public, and can have serious political, social, and health consequences.  
Manual fact-checking is slow and infeasible at scale. The goal is to build an **automated, accurate, and interpretable** detection system to identify fake news in real time.

## Project Objectives
### **Objective 1:** Perform Extraploratory Data Analysis (EDA)

**Number of Sentences Distribuction**
<img width="2370" height="1765" alt="num_sentences_distribution_class0_filtered" src="https://github.com/user-attachments/assets/926c8356-5bee-4711-9854-39be009b3608" />

<img width="2370" height="1765" alt="num_sentences_distribution_class1_filtered" src="https://github.com/user-attachments/assets/0c42bce7-cf14-4292-91c1-e747ab58c848" />

**Text Length Distribuction**
<img width="2370" height="1765" alt="text_length_distribution_class0_filtered" src="https://github.com/user-attachments/assets/e1c4efc4-e509-4ddc-a9b5-be4f6991a279" />

<img width="2370" height="1765" alt="text_length_distribution_class1_filtered" src="https://github.com/user-attachments/assets/cea74697-c904-4aef-aa6c-8c3379879c3b" />


**Top-10 Bigrams**
<img width="4468" height="2365" alt="top-10-bigrams" src="https://github.com/user-attachments/assets/e08e1d1f-4e29-488f-8307-b52af07e44f6" />

**Top-10 Trigrams**
<img width="4468" height="2365" alt="top-10-trigrams" src="https://github.com/user-attachments/assets/cc100f04-1f62-4f05-8405-23b9d4d8dce7" />


### **Objective 2:** Build a baseline Logistic Regression model
A baseline Logistic Regression model was built with imbalanced data and two balancing strategies:
- **SMOTE** oversampling for the minority class
- **Random Undersampling (RUS)** for the majority class
  - **Best configuration:** RUS-balanced data: **Recall (Fake) = 0.8214**

### **Objective 3:** Build an **XGBoost** model, **LSTM** neural network, and Finetune the **RoBERTa-base Transformer**
- An XGBoost classifier:  
  - **Best configuration:** RUS-balanced data: **Recall (Fake) = 0.8427**
- An LSTM neural network:
  - **Best configuration:** Imbalanced data: **Recall (Fake) = 0.7130**
- Finetune the RoBERTa-base transformer:  
  - **Best configuration:** RUS-balanced data: **Recall (Fake) = 0.8712**

### **Objective 4:** Interpret best performing model using the LIME library

| MODEL           | Recall (Fake) | Recall (Real) | Precision (Fake) | Precision (Real) | AUC | Accuracy |
|-----------------|--------|--------|--------|--------|-------|-------|
| LR-RUS          | 0.8214 | 0.9014 | 0.8846 | 0.8458 | 0.930 | 0.8631 | 
| XGBoost-RUS     | 0.8427 | 0.9054 | 0.8913 | 0.8622 | 0.939 | 0.8754 | 
| LSTM-Imbalanced | 0.7130 | 0.9425 | 0.9194 | 0.7811 | 0.867 | 0.8325 | 
| RoBERTa-RUS     | **0.8712** | 0.8920 | 0.8813 | **0.8827** | 0.951 | 0.8820 | 

The Finetuned RoBERTa model using SMOTE-balanced dataset is the best performing alternative because: 
- It achieves the highest **Recall (Fake)** (**_0.8712_**) **:** Highest **true positives for fake (`class 0`) predicitions (**_2814_**).
- It achieves the highest **Precision (Real)** (**_0.8872_**) **:** Highest certainity (*88.27%*) for `class 1` predicitions when an article is actually **real**.
  

- This is corroborated by the confusion matrix, which reports the highest true positives for accurate fake news predictions (**_2814_**).
<img width="2970" height="2066" alt="roberta-roc-auc" src="https://github.com/user-attachments/assets/f7d68f78-1b17-4c07-bcd4-c9ab998fd466" />


  - **LIME** was used to explain the **_finetuned RoBERTa-base Transformer_** model's predictions at the feature level.
  - Visualizations highlight the top 10 features supporting the prediction (green = `predicted class`, red = `alternative prediction`).
<img width="2969" height="1765" alt="lime_explanation_barplot" src="https://github.com/user-attachments/assets/3fd97b6e-7222-4787-90f9-e839585ad398" />


### **Objective 5:** Deploy selected model using FastAPI and Streamlit
The **Finetuned RoBERTa-base transformer with SMOTE-balanced dataset** is deployed using FastAPI, Streamlit, and Docker.
  - **Back-end :** The FastAPI app (**_main.py_**) is responsible for model inference. 
  - **Front-end :** The Streamlit app (**_app.py_**) provides an interactive frontend web interface.
  - **Docker :** Containerizes the **_Front-end_** and the **_Back-end_**.

## Conclusion
 - The **Finetuned RoBERTa with RUS balancing** achieved the highest **Recall (0.8712) and demonstrated strong capability in detecting fake news articles.
 - Its contextual language understanding from pretraining and finetuning using RUS-balanced data improved recall on the minority class without sacrificing generalization.

## Recommendations
- **Collaborate with Media Outlets :** Work alongsode news dissemination orgarnizations and social media platforms to incorporate the finetuned RoBERTa model for real-time inference.
- **Educate the Public :** Raise awareness about fake news detection tools, and encourage the public to leverage these technologies for more informed decision-making.
- **Expand into Multiple Languages :** Train the model with multilingual datasets to optimize robustness in catching fake news articles written/ published across multiple languages/ dialects.  

## Next Steps
- Deploy the containerized finetuned RoBERTa model via AWS SageMaker for low-latency inference.
- Develop browser extensions and social media plugins using the model’s API for real-time credibility scoring.
- Incorporate automation frameworks to flag low-confidence predictions for human review.

## Installation & Running the App
1. Clone the repository  
`git clone https://github.com/mwakad/fake-news-detector.git`
2. Install dependencies
`pip install -r requirements.txt` (cd deployment/backend)
3. Run FastAPI backend
`uvicorn main:app --reload`
4. Run Streamlit frontend
`streamlit run streamlit_app.py`
