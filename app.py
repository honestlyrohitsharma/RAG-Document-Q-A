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
st.title("🤖 Multi-Document RAG System + RAGAS")
st.markdown("**Upload • Process • Ask • Evaluate**")

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
        process_btn = st.button("🔄 Process Documents", type="primary")
    with col2:
        clear_btn = st.button("🗑️ Clear All")

    # RAGAS Evaluation Button
    st.header("📊 RAGAS Evaluation")
    if st.button("Run RAGAS Evaluation", type="secondary"):
        if st.session_state.vectorstore is None:
            st.error("Please process documents first!")
        else:
            with st.spinner("Running RAGAS Evaluation..."):
                try:
                    test_questions = [
                        "What is the main topic of the documents?",
                        "Summarize the key points from the uploaded PDFs.",
                        "What are the important findings or conclusions?"
                    ]
                    
                    # Ragas requires 'reference' (ground truth) for context_precision/recall
                    references = [
                        "The documents discuss the main topic in detail.",
                        "The key points include the core arguments of the text.",
                        "The findings highlight the primary conclusions."
                    ]
                    
                    answers = []
                    contexts = []
                    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 6})
                    
                    for q in test_questions:
                        docs = retriever.invoke(q)
                        context = [doc.page_content for doc in docs]
                        contexts.append(context)
                        
                        llm_temp = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"),
                                           model_name="llama-3.1-8b-instant", temperature=0.3)
                        prompt = ChatPromptTemplate.from_template("Answer based on context:\nContext: {context}\nQuestion: {question}")
                        chain = create_stuff_documents_chain(llm_temp, prompt)
                        answer = chain.invoke({"context": docs, "question": q})
                        answers.append(answer)
                    
                    dataset = Dataset.from_dict({
                        "question": test_questions,
                        "answer": answers,
                        "contexts": contexts,
                        "reference": references,
                    })
                    
                    result = evaluate(
                        dataset=dataset,
                        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                    )
                    
                    st.success("✅ RAGAS Evaluation Done!")
                    st.dataframe(result.scores)
                    
                    avg = result.scores.mean()
                    for metric, score in avg.items():
                        st.metric(metric.replace("_", " ").title(), f"{score:.3f}")
                        
                except Exception as e:
                    st.error(f"Evaluation Error: {str(e)}")

    if clear_btn:
        try:
            if st.session_state.get("vectorstore"):
                st.session_state.vectorstore.delete_collection()
        except Exception:
            pass
        st.session_state.vectorstore = None
        st.session_state.messages = []
        
        # Force garbage collection to release file locks on Windows
        import gc
        gc.collect()
        
        if os.path.exists("uploaded_docs"): shutil.rmtree("uploaded_docs", ignore_errors=True)
        if os.path.exists("vectorstore_db"): shutil.rmtree("vectorstore_db", ignore_errors=True)
        
        st.success("✅ Everything cleared!")
        st.rerun()

    if process_btn and uploaded_files:
        with st.spinner("Processing..."):
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
                
                st.success(f"✅ Processed {len(uploaded_files)} document(s) | Chunks: {len(chunks)}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# ====================== CHAT AREA ======================
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
    st.info("👈 Upload PDFs and click **Process Documents**")

st.caption("RAG System with RAGAS Evaluation")