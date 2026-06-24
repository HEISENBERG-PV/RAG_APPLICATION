from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()



template = ChatPromptTemplate.from_messages(
    [("system", "You summarize the data in simple words"),
     ("human", "{data}")
     ]
)

model = init_chat_model(
    "gemini-2.5-flash-lite",
    model_provider="google_genai")

