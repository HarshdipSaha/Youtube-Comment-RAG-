import sys
import os
import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

st.title("YouTube Comment Analyser 🎬 🎥 🔴 ▶")

# Input fields
Url = st.text_input("YouTube video link: ")
btn = st.button("Create Knowledgebase")

# Progress bar and button action
if btn:
    progress_bar = st.progress(0)

    def progress_callback(progress):
        progress_bar.progress(progress)

    with st.spinner("Creating the knowledgebase..."):
        create_vector_db(Url, progress_callback)
        st.success("Knowledgebase created successfully!")

# Question input and response
question = st.text_input("Question: ")

if question:
    chain = get_qa_chain()
    response = chain.invoke({"input": question})
    
    st.header("Answer")
    st.write(response["answer"])
