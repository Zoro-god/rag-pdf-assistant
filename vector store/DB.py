from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

from langchain_core.documents import Document

docs = [
    Document(
        page_content="Python is widely used in Artificial Intelligence.",
        metadata={"source": "AI_book"}
    ),
    Document(
        page_content="Pandas is used for data analysis in Python.",
        metadata={"source": "DataScience_book"}
    ),
    Document(
        page_content="Neural networks are used in deep learning.",
        metadata={"source": "DL_book"}
    )
]

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
            )

vectorstore = Chroma.from_documents(
    documents=docs, 
    embedding=embedding_model,
    persist_directory="chroma-db",
    )

vectorstore.similarity_search("what is used for data analysis?",k=2)

for r in results:
    print(r)
