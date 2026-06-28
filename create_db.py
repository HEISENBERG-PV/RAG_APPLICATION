#load pdf
#chunnking
#create embedding 
# store in vector DB

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
load_dotenv()

data = PyPDFLoader("document_loader/deeplearning.pdf")
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = splitter.split_documents(docs)
print(len(chunks))

embedding = MistralAIEmbeddings(
    model= "mistral-embed"
)

vector_stores = Chroma.from_documents(
    documents= chunks,
    embedding= embedding,
    persist_directory= "vector_DB"
)