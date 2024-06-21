import os
from langchain_community.document_loaders import PyPDFLoader
pdf = PyPDFLoader("https://uou.ac.in/sites/default/files/slm/DHA-101.pdf")

pdfpages = pdf.load_and_split()

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings

my_books = pdf.load()
print("pdf loaded")

text_splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=5)
split_text = text_splitter.split_documents(my_books)

print("document splitting done")

llm = Ollama(model="llama2")
embeddings = OllamaEmbeddings(model="llama2")
vectorstore = FAISS.from_documents(split_text,embeddings)

print("vectorstore Done")

vector_retriver = vectorstore.as_retriever()

from langchain.agents.agent_toolkits import create_retriever_tool

tool = create_retriever_tool(
    vector_retriver,
    "Hospitality_Topic_Search",
    "Retrive Detailed information on the Hospitality Services"
    )

tools = [tool]

from langchain.agents.agent_toolkits import create_conversational_retrieval_agent

myagent = create_conversational_retrieval_agent(llm,tools,verbose=True)

contex = "The user is conducting research on the hospitality services"
question = "Types of meal plan?"

prompt = f"""You need to answer the question in the sentence as same as in the pdf content.
Given below is the contex and question of the user.
contex = {contex}
question = {question}
"""

result = myagent.invoke({"input":prompt})

import streamlit as st
import os
st.title('Hospitality Services Q&A')
contex = "The user is conducting research on hospitality services"
question = st.text_input('Enter your question:', 'Types of meal plan?')

if st.button('Ask'):
    prompt = f"""You need to answer the question in the sentence as same as in the pdf content.
    Given below is the context and question of the user.
    context = {contex}
    question = {question}
    """
    result = myagent.invoke({"input": prompt})
    st.write(result['output'])
