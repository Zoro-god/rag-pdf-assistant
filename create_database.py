
import shutil
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# Configuration
pdf_path = "document loaders/F.pdf"
persist_directory = "chroma-db"

# 1. Load PDF
data = PyPDFLoader(pdf_path)
docs = data.load()

print("PDF pages loaded:", len(docs))

# 2. Split PDF into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)

print("Chunks created:", len(chunks))

# 3. Fix invalid Unicode in extracted text
for chunk in chunks:
    chunk.page_content = chunk.page_content.encode(
        "utf-8", errors="replace"
    ).decode("utf-8")

# 4. Clear the old database to prevent duplicate chunks
db_path = Path(persist_directory)

if db_path.exists():
    shutil.rmtree(db_path)
    print("Old database deleted.")

# 5. Configure Gemini embeddings
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

# 6. Create a fresh ChromaDB database
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=persist_directory
)

# 7. Verify database creation
count = vectorstore._collection.count()

print("Chunks stored:", count)

if count == len(chunks):
    print("Database created successfully!")
else:
    print("Warning: Stored chunk count differs from chunk count.")
