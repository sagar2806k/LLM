# app.py

import streamlit as st
from model import analyze_pdf, query_model,retrieve_relevent_text

st.title("LangChain RAG Pipeline with Llama 3")

# PDF Upload
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf", accept_multiple_files=True)
if uploaded_file:
    pdf_text = ""
    for file in uploaded_file:
        pdf_text += analyze_pdf(file)
    st.session_state['pdf_text'] = pdf_text
    st.success("PDF data analyzed successfully!")

# User Prompt
if 'pdf_text' in st.session_state:
    user_prompt = st.text_input("Enter your query:")
    if st.button("Submit Query"):
        with st.spinner('Processing...'):
            relevant_text = retrieve_relevent_text(user_prompt, st.session_state['pdf_text'])
            response = query_model(user_prompt, relevant_text)
            st.write(response)
