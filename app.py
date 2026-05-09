import streamlit as st
from dotenv import load_dotenv
import os
import shutil

from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ✅ Corrected Imports
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# RAGAS
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

load_dotenv()

st.set_page_config(page_title="RAG Document Q&A", layout="wide")
st.title("🤖 Intelligent Multi-Document Q&A System")
st.markdown("**Upload PDFs • Ask Questions**")

# ====================== SESSION STATE ======================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("📄 Document Management")
    
    uploaded_files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Process Documents", type="primary"):
            if uploaded_files:
                with st.spinner("Processing documents..."):
                    try:
                        os.makedirs("uploaded_docs", exist_ok=True)
                        for file in uploaded_files:
                            with open(f"uploaded_docs/{file.name}", "wb") as f:
                                f.write(file.getbuffer())
                        
                        loader = PyPDFDirectoryLoader("uploaded_docs")
                        docs = loader.load()
                        
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                        chunks = text_splitter.split_documents(docs)
                        
                        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                        
                        st.session_state.vectorstore = Chroma.from_documents(
                            documents=chunks,
                            embedding=embeddings,
                            persist_directory="./vectorstore_db"
                        )
                        
                        st.success(f"✅ Processed {len(uploaded_files)} document(s)")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.warning("Please upload PDFs first.")

    with col2:
        if st.button("🗑️ Clear All"):
            if os.path.exists("uploaded_docs"):
                shutil.rmtree("uploaded_docs")
            if os.path.exists("vectorstore_db"):
                shutil.rmtree("vectorstore_db")
            st.session_state.vectorstore = None
            st.session_state.messages = []
            st.success("✅ All data cleared!")
            st.rerun()

# ====================== MAIN CHAT AREA ======================
if st.session_state.vectorstore is not None:
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0.3
    )

    prompt = ChatPromptTemplate.from_template("""
    Answer the question based only on the provided documents.
    If you cannot find the answer, say "I don't have enough information from the documents."

    Context: {context}
    Question: {input}
    Answer:
    """)

    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 6})
    document_chain = create_stuff_documents_chain(llm, prompt)
    qa_chain = create_retrieval_chain(retriever, document_chain)

    st.header("💬 Ask Questions")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_question := st.chat_input("Ask anything about your documents..."):
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = qa_chain.invoke({"input": user_question})
                answer = response["answer"]
                st.markdown(answer)
                
        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("👈 Upload PDFs from sidebar and click **Process Documents** to begin.")

st.caption("Built with LangChain + Groq + Chroma")

st.caption("RAG System with RAGAS Evaluation")
