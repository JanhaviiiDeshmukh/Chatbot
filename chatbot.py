import streamlit as st
import os
from dotenv import load_dotenv

# LangChain Imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="C++ Chatbot", page_icon=":robot_face:")
st.title("C++ Chatbot")
st.write("Ask questions")


load_dotenv()

@st.cache_resource
def load_vectorstore():
    # Load the text file
    loader = TextLoader("C++_Introduction.txt", encoding="utf-8")
    documents = loader.load()

    # Split the text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    finalDocuments = text_splitter.spilt_documents(documents)

    # Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(finalDocuments, embeddings)
    return db

db = load_vectorstore()

# -----------------------------
# User Input
# -----------------------------
query = st.text_input("Enter your question about C++:")

if query:
    docs = db.similarity_search(query, k=3)

    st.subheader("📚 Retrieved Context:")

    for i, doc in enumerate(docs):
        st.markdown(f"*Result {i+1}:*")
        st.write(doc.page_content)
