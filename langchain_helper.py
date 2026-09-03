from langchain_community.document_loaders.csv_loader import CSVLoader
import pandas as pd
import csv
import os
from langchain_core.prompts import PromptTemplate
sec_key = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

if sec_key:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = sec_key
from langchain_huggingface import HuggingFaceEndpoint

from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
apikey="AIzaSyCOll-1nURu72Fv-XMKNx0txTb7J77y5cE"
from langchain.text_splitter import RecursiveCharacterTextSplitter
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest",google_api_key=apikey)

from langchain_core.prompts.prompt import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
vectordb_file_path = "faiss_index"
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")

import time

def create_vector_db(Url, progress_callback=None):
    url = Url  # Replace with your YouTube URL
    downloader = YoutubeCommentDownloader()
    
    if progress_callback:
        progress_callback(10)  # Update progress to 10%

    comments = downloader.get_comments_from_url(url, sort_by=SORT_BY_POPULAR)

    if progress_callback:
        progress_callback(30)  # Update progress to 30%

    # Open a CSV file to write the comments
    try:
        with open('youtube_comments.csv', mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Merged Comment'])  # Single header for the merged column

            for idx, comment in enumerate(comments, start=1):
                text = comment['text']
                likes = comment['votes']
                user_id = comment['author']
                published_at = comment['time']

                # Merge all columns into a single column with the specified format
                merged_comment = f"comment is '{text}' with likes= {likes} with user_id '{user_id}' and published(time)  '{published_at}'"
                writer.writerow([merged_comment])
    except Exception as ex:
        print(ex)

    if progress_callback:
        progress_callback(60)  # Update progress to 60%

    loader = CSVLoader(file_path='youtube_comments.csv', encoding="utf-8")
    data = loader.load()

    if progress_callback:
        progress_callback(80)  # Update progress to 80%

    instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
    vectordb = FAISS.from_documents(documents=data, embedding=instructor_embeddings)
    vectordb.save_local('vectordb_file_path')  # Specify your file path here

    if progress_callback:
        progress_callback(100)  # Update progress to 100%
def get_qa_chain():
    
    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings,allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever(score_threshold=0.7)
    
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=20, temperature=0.7, token=sec_key)
    system_prompt = (
        "Use the given context to answer the question. "
        
        "If you don't know the answer, return the most nearest answer. "
        "In the context , details of content of comment start from 'comment is', details of number of likes(integer type) is starting from 'with likes' then the number of likes(integer),then we have userid(who posted) and its time(date) of publish "
        "Context: {context}"
     )

    # Define the prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return chain


if __name__ == "__main__":
    
    chain = get_qa_chain()
  
 