from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

embedding = MistralAIEmbeddings(
    model="mistral-embed"
)

vector_store = Chroma(
    persist_directory="vector_DB",
    embedding_function= embedding
)

retriever = vector_store.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        "k" : 3,
        "fetch_k": 10,
        "lambda_mult": 0.25
    }
)

# doc = retriever.invoke("what is deep learning")
# print(doc[0].page_content)

template = ChatPromptTemplate.from_messages(
    [
        ("system", """You are powerful AI assistance
        Answer the query only from the given context
        If context doesn't have the answer then just tell user that the answer is not present in the document """
    ),
        ("human", """
            Context: {context}
            Query: {query}"""
    )
     ]
)

model = init_chat_model(
    "gemini-2.5-flash-lite",
    model_provider="google_genai")


while True:
    query = input("You : ")
    if query == "0":
        break

    doc = retriever.invoke(query)
    context = " ".join(
        [i.page_content for i in doc]
    )

    final_prompt = template.invoke({
        "context": {context},
        "query": {query}
    })

    res = model.invoke(final_prompt)

    print("AI : ",res.content)

