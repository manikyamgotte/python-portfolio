import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# -----------------------------
# Streamlit App UI
# -----------------------------
st.set_page_config(page_title="AI PDF Chat", layout="wide")
st.title("📄 AI PDF Chat (No Quota Errors)")

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF uploaded successfully!")

    # -----------------------------
    # Load and split PDF
    # -----------------------------
    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    # -----------------------------
    # FREE EMBEDDINGS (No API Key)
    # -----------------------------
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vector_db = FAISS.from_documents(chunks, embeddings)

    st.success("Document processed successfully! Ask your question below 👇")

    # -----------------------------
    # Ask Question
    # -----------------------------
    question = st.text_input("Ask a question about the PDF")

    if question:
        # Search relevant text
        matched_docs = vector_db.similarity_search(question, k=3)

        # -----------------------------
        # AI Model (OpenAI Optional)
        # -----------------------------
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        answer = llm.invoke(
            f"Answer the question using ONLY the following context:\n{matched_docs}\n\nQuestion: {question}"
        )

        st.write("### 🧠 Answer:")
        st.write(answer.content)
