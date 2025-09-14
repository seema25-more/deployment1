import streamlit as st
import os
import torch
from transformers import pipeline
import boto3

# AWS S3 setup
bucket_name = "final-practise"
s3_prefix = 'ml-models/tinybert-sentiment-analysis/'
local_path = 'new-tinybert-sentiment-analysis'

s3 = boto3.client('s3')

# Function to download S3 folder locally
def download_dir(local_path, s3_prefix):
    os.makedirs(local_path, exist_ok=True)
    paginator = s3.get_paginator('list_objects_v2')
    for result in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
        if 'Contents' in result:
            for key in result['Contents']:
                s3_key = key['Key']
                local_file = os.path.join(local_path, os.path.relpath(s3_key, s3_prefix))
                os.makedirs(os.path.dirname(local_file), exist_ok=True)
                s3.download_file(bucket_name, s3_key, local_file)

# Streamlit UI
st.title("Machine Learning Model Deployment at the Server!!!")

# Download button
if st.button("Download Model"):
    with st.spinner("Downloading... Please wait!"):
        download_dir(local_path, s3_prefix)
        st.success("Model downloaded successfully!")

# Input text
text = st.text_area("Enter Your Review", "Type here...")

# Predict button
if st.button("Predict"):
    # Check if model exists locally
    if not os.path.exists(local_path) or not os.listdir(local_path):
        st.error("Model not found locally. Please download it first!")
    else:
        with st.spinner("Predicting..."):
            # Load pipeline from local folder
            device = 0 if torch.cuda.is_available() else -1  # GPU or CPU
            classifier = pipeline('text-classification', model=local_path, device=device)
            output = classifier(text)
            st.write(output)
