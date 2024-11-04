import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch 
from torch import nn
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import nltk
import re

nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

N_LABELS = 3
MAX_LEN = 128
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

roberta_tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base", do_lower_case=True)
roberta_model = RobertaForSequenceClassification.from_pretrained("FacebookAI/roberta-base", num_labels=N_LABELS)
roberta_model.load_state_dict(torch.load("/kaggle/working/roberta.pth", map_location=device))
roberta_model.to(device)

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower() 
    
    # Replace weird characters and links with no character
    text = re.sub(r'\$[A-Za-z]+', '', text) 
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text) 
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    
    # Tokenize text
    words = text.strip().split()
    
    # Remove stopwords
    words = [word for word in words if word not in stop_words]
    text = ' '.join(words)
    return text

def predictForStreamlit(input_text, roberta_model, roberta_tokenizer):
    roberta_model.eval()

    encoding = roberta_tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
    )
    
    input_ids, attention_mask = encoding['input_ids'].to(device), encoding['attention_mask'].to(device)
    label_predict_idx = None
    label_name_lst = ['Bearish','Bullish', 'Neutral']
    prob_lst = None
    with torch.no_grad():
        test_output = roberta_model(input_ids=input_ids,attention_mask=attention_mask)
        logits = test_output.logits
        prob_lst = logits.softmax(dim=1)
        logits = logits.detach().cpu().numpy()
        label_predict_idx = np.argmax(logits) 

    
    return prob_lst, label_predict_idx, label_name_lst[label_predict_idx], prob_lst[0][label_predict_idx].item()

# Streamlit app interface
st.title("Finance-related Tweets Sentiment Analyzer")

# Text input from user
user_input = st.text_area("Enter a finance-related tweet:", "")

# Predict sentiment when the user submits the input
if st.button("Predict"):
    if len(user_input) != 0:
        input_text_preprocessed = preprocess_text(user_input)
        prob_lst, label_predict_idx, sentiment, prob = predictForStreamlit(input_text_preprocessed, roberta_model, roberta_tokenizer)
        st.subheader(":green[**Result**]")
        prob_real_lst = [prob_lst[0][0].item(), prob_lst[0][1].item(), prob_lst[0][2].item()]
        # Visualization
        fig, ax = plt.subplots()
        ax.pie(prob_real_lst, labels=['Bearish','Bullish', 'Neutral'], autopct='%.2f')
        col1, col2 = st.columns(2)
        col1.pyplot(fig)
        
        # Predict label
        col2.write(f"Predicted label: **{label_predict_idx}**")
        col2.write(f"Predicted sentiment: **{sentiment}**")
        col2.write(f"Probability (round 4 digits after dot): **{round(prob,4)} ({round(prob,4)*100}%)**")
        
    else:
        st.write("Please enter a valid tweet!")