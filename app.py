
import os
import shutil
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="DocuMind | AI PDF Assistant",
    page_icon="📚",
    layout="wide",
)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "source_files" not in st.session_state:
    st.session_state.source_files = []

if "work_dir" not in st.session_state:
    st.session_state.work_dir = None

# --------------------------------------------------
# MODELS - SAME MODELS AS YOUR PROJECT
# --------------------------------------------------

@st.cache_resource
def get_embedding_model():
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )


@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        temperature=0.7,
    )


# --------------------------------------------------
# PROMPT TEMPLATE
# --------------------------------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."

Do not invent facts. Be clear and explain concepts
in simple language when appropriate."""
    ),
    (
        "human",
        """Context:
{context}

Question:
{question}
"""
    ),
])


# --------------------------------------------------
# PDF PROCESSING
# --------------------------------------------------

def clean_unicode(text):
    return text.encode(
        "utf-8", errors="replace"
    ).decode("utf-8")


def process_pdfs(uploaded_files):
    work_dir = tempfile.mkdtemp(prefix="documind_")
    db_path = os.path.join(work_dir, "chroma-db")

    try:
        all_docs = []

        # 1. Save and load uploaded PDFs
        for index, uploaded_file in enumerate(uploaded_files):
            safe_name = Path(uploaded_file.name).name
            pdf_path = os.path.join(
                work_dir,
                f"{index}_{safe_name}"
            )

            with open(pdf_path, "wb") as file:
                file.write(uploaded_file.getbuffer())

            loader = PyPDFLoader(pdf_path)
            docs = loader.load()

            # Preserve the original uploaded filename
            for doc in docs:
                doc.metadata["source"] = uploaded_file.name

                # Clean extracted text and string metadata
                doc.page_content = clean_unicode(
                    doc.page_content
                )

                for key, value in doc.metadata.items():
                    if isinstance(value, str):
                        doc.metadata[key] = clean_unicode(value)

            all_docs.extend(docs)

        if not all_docs:
            raise ValueError(
                "No readable pages were found in the uploaded PDFs."
            )

        # 2. Split documents into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

        chunks = splitter.split_documents(all_docs)

        if not chunks:
            raise ValueError(
                "No text chunks could be created from these PDFs."
            )

        # 3. Create a fresh vector database for these uploads
        embedding_model = get_embedding_model()

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=db_path,
        )

        count = vectorstore._collection.count()

        if count == 0:
            raise ValueError(
                "No chunks were stored in the vector database."
            )

        return vectorstore, count, work_dir

    except Exception:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise


# --------------------------------------------------
# SIDEBAR: UPLOAD DOCUMENTS
# --------------------------------------------------

with st.sidebar:
    st.title("📚 DocuMind")
    st.caption("Your AI-powered PDF assistant")

    st.divider()

    st.subheader("Upload your PDFs")

    uploaded_files = st.file_uploader(
        "Choose one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload textbooks, notes, reports, or other PDF documents.",
    )

    if uploaded_files:
        st.caption(f"{len(uploaded_files)} PDF file(s) selected")

        for uploaded_file in uploaded_files:
            st.write(f"📄 {uploaded_file.name}")

    process_button = st.button(
        "Process Documents",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_files,
    )

    if process_button:
        try:
            with st.spinner(
                "Reading PDFs, creating embeddings, and building your database..."
            ):
                new_vectorstore, count, new_work_dir = process_pdfs(
                    uploaded_files
                )

            # Replace the previous database only after success
            old_work_dir = st.session_state.work_dir

            st.session_state.vectorstore = new_vectorstore
            st.session_state.work_dir = new_work_dir
            st.session_state.source_files = [
                file.name for file in uploaded_files
            ]
            st.session_state.chat_history = []

            if old_work_dir:
                shutil.rmtree(
                    old_work_dir,
                    ignore_errors=True,
                )

            st.success(
                f"Processed {len(uploaded_files)} PDF(s) "
                f"into {count} chunks!"
            )

        except Exception as error:
            st.error(f"Could not process the PDFs: {error}")

    st.divider()

    if st.session_state.vectorstore is not None:
        st.success("Documents are ready!")

        st.write("**Loaded files**")
        for name in st.session_state.source_files:
            st.write(f"📄 {name}")

        if st.button("Clear chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()


# --------------------------------------------------
# MAIN CHAT INTERFACE
# --------------------------------------------------

st.title("Chat with your Documents")
st.write(
    "Upload your PDFs and ask questions. "
    "Answers are generated using information retrieved from your documents."
)

if st.session_state.vectorstore is None:
    st.info(
        "👈 Upload one or more PDFs in the sidebar, "
        "then click **Process Documents** to begin."
    )

    st.markdown("### How it works")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📄 Upload")
        st.write("Select your PDF files.")

    with col2:
        st.markdown("### 🔍 Retrieve")
        st.write("Find relevant document chunks.")

    with col3:
        st.markdown("### 🤖 Ask")
        st.write("Get answers grounded in your documents.")

else:
    # Display previous messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("sources"):
                with st.expander("View retrieved sources"):
                    for source in message["sources"]:
                        st.markdown(
                            f"**{source['file']} — "
                            f"Page {source['page']}**"
                        )
                        st.write(source["text"])

    # Accept a new question
    user_input = st.chat_input(
        "Ask a question about your PDFs..."
    )

    if user_input:
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input,
        })

        with st.chat_message("user"):
            st.markdown(user_input)

        try:
            with st.chat_message("assistant"):
                with st.spinner("Searching your documents..."):
                    # 1. Retrieve relevant chunks
                    retriever = (
                        st.session_state.vectorstore.as_retriever(
                            search_type="similarity",
                            search_kwargs={"k": 4},
                        )
                    )

                    docs = retriever.invoke(user_input)

                    if not docs:
                        answer = (
                            "I could not find relevant information "
                            "in the uploaded documents."
                        )
                        sources = []

                    else:
                        # 2. Prepare context
                        context = "\n\n".join(
                            doc.page_content for doc in docs
                        )

                        # 3. Format prompt
                        final_prompt = prompt.invoke({
                            "context": context,
                            "question": user_input,
                        })

                        # 4. Generate answer
                        result = get_llm().invoke(final_prompt)
                        answer = result.text

                        # 5. Capture source information
                        sources = []

                        for doc in docs:
                            page = doc.metadata.get("page_label")

                            if page is None:
                                page = doc.metadata.get("page")

                                if page is not None:
                                    page = int(page) + 1

                            sources.append({
                                "file": doc.metadata.get(
                                    "source", "Unknown file"
                                ),
                                "page": (
                                    str(page)
                                    if page is not None
                                    else "Unknown"
                                ),
                                "text": doc.page_content[:1000],
                            })

                st.markdown(answer)

                if sources:
                    with st.expander("View retrieved sources"):
                        for source in sources:
                            st.markdown(
                                f"**{source['file']} — "
                                f"Page {source['page']}**"
                            )
                            st.write(source["text"])

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })

        except Exception as error:
            st.error(f"Something went wrong: {error}")
