
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# 1. Load the same embedding model used during database creation
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

# 2. Open the existing database
vectorstore = Chroma(
    persist_directory="chroma-db",
    embedding_function=embedding_model
)

print("Database count:", vectorstore._collection.count())

# 3. Create the retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# 4. Create the prompt template
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
    ),
    (
        "human",
        """Context:
{context}

Question:
{question}
"""
    )
])

# 5. Configure your existing Gemini chat model
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0.7
)

print("RAG system created")
print("Enter 0 to stop")

# 6. Ask questions
while True:
    user_input = input("\nEnter your question: ").strip()

    if user_input == "0":
        break

    if not user_input:
        continue

    # Retrieve relevant chunks
    docs = retriever.invoke(user_input)

    print("\n--- Retrieved Documents ---")
    print("Number of documents:", len(docs))

    for i, doc in enumerate(docs, start=1):
        print(f"\nDocument {i}")
        print("Metadata:", doc.metadata)
        print(doc.page_content[:700])
        print("-" * 40)

    # Handle empty retrieval
    if not docs:
        print("No documents were retrieved.")
        continue

    # Combine retrieved text into context
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    # Fill the prompt template
    final_prompt = prompt.invoke({
        "context": context,
        "question": user_input
    })

    # Generate the answer
    result = llm.invoke(final_prompt)

    print("\nAI:", result.text)
