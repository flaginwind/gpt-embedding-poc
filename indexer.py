import os
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import openai


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

if __name__ == "__main__":
    print("OPENAI_API_KEY :" + OPENAI_API_KEY)
    print("OPENAI_DEPLOYMENT_ENDPOINT :" + OPENAI_DEPLOYMENT_ENDPOINT)
    print("OPENAI_DEPLOYMENT_NAME :" + OPENAI_DEPLOYMENT_NAME)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model=OPENAI_EMBEDDING_DEPLOYMENT_NAME , chunk_size=1)
    dataPath = "D:/Projects/GPT/"
    fileName = dataPath + "1.pdf"

    #use langchain PDF loader
    loader = PyPDFLoader(fileName)

    #split the document into chunks
    pages = loader.load_and_split()

    #Use Langchain to create the embeddings using text-embedding-ada-002
    db = FAISS.from_documents(documents=pages, embedding=embeddings)

    #save the embeddings into FAISS vector store
    db.save_local("D:/Projects/GPT/faiss_index")