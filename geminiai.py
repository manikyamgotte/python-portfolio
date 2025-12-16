import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="AI PDF Chat (Gemini)", layout="wide")
st.title("📘 Gemini AI PDF Chat")

uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_pdf:
    # Save the uploaded PDF
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_pdf.read())

    # Load PDF
    loader = PyPDFLoader("temp.pdf")
    pages = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(pages)

    # Convert to embeddings using Gemini
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

    vector_db = FAISS.from_documents(docs, embeddings)

    # LLM Model (Gemini Flash)
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        google_api_key=GOOGLE_API_KEY
    )

    st.success("PDF processed successfully!")

    query = st.text_input("Ask a question:")

    if st.button("Get Answer"):
        with st.spinner("Searching..."):
            retrieved_docs = vector_db.similarity_search(query, k=3)
            context = "\n\n".join([d.page_content for d in retrieved_docs])

            prompt = (
                "You are an assistant. Answer using ONLY the context below.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {query}\n\n"
                "Answer:"
            )

            result = llm.invoke(prompt)

        st.subheader("Answer:")
        st.write(result)

        st.subheader("Sources:")
        for i, d in enumerate(retrieved_docs):
            st.write(f"📄 Source {i+1}:\n{d.page_content[:300]}...")
