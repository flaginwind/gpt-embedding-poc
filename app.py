from langchain.vectorstores import FAISS
from typing import cast
import openai
import os
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings

#load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_DEPLOYMENT_ENDPOINT = os.getenv("OPENAI_DEPLOYMENT_ENDPOINT")
OPENAI_DEPLOYMENT_NAME = os.getenv("OPENAI_DEPLOYMENT_NAME")
OPENAI_MODEL_NAME = "gpt-35-turbo"
OPENAI_EMBEDDING_DEPLOYMENT_NAME = "text-embedding-ada-002"
OPENAI_EMBEDDING_MODEL_NAME = "text-embedding-ada-002"
OPENAI_DEPLOYMENT_VERSION = "2023-05-15"

#init Azure OpenAI
openai.api_type = "azure"
openai.api_version = OPENAI_DEPLOYMENT_VERSION
openai.api_base = OPENAI_DEPLOYMENT_ENDPOINT
openai.api_key = OPENAI_API_KEY

def ask_question(qa, question):
    result = qa({"query": question})
    print("Question:", question)
    print("Answer:", result["result"])

if __name__ == "__main__":
    print("OPENAI_API_KEY :" + OPENAI_API_KEY)
    print("OPENAI_DEPLOYMENT_ENDPOINT :" + OPENAI_DEPLOYMENT_ENDPOINT)
    print("OPENAI_DEPLOYMENT_NAME :" + OPENAI_DEPLOYMENT_NAME)
    #init openai
    llm = AzureChatOpenAI(deployment_name=OPENAI_DEPLOYMENT_NAME,
                      model_name=OPENAI_MODEL_NAME,
                      openai_api_base=OPENAI_DEPLOYMENT_ENDPOINT,
                      openai_api_version=OPENAI_DEPLOYMENT_VERSION,
                      openai_api_key=OPENAI_API_KEY)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model=OPENAI_EMBEDDING_MODEL_NAME, chunk_size=1)

    #load the faiss vector store we saved into memory
    vectorStore = FAISS.load_local("D:/Projects/GPT/faiss_index", embeddings)
    print(vectorStore)
    print("vector end")

    #use the faiss vector store we saved to search the local document
    retriever = vectorStore.as_retriever(search_type="similarity", search_kwargs={"k":2})

    #use the vector store as a retriever
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=False)
    while True:
        query = input('you: ')
        if query == 'q':
            break
        ask_question(qa, query)