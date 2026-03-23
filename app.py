import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOpenAI

load_dotenv()

st.set_page_config(page_title="Indecimal Assistant", page_icon="🏗️", layout="wide")

# Custom CSS for a Premium UI compatible with Dark/Light Mode
st.markdown("""
<style>
    /* Global font */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    /* Header styling */
    .custom-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
        margin-bottom: 2rem;
    }
    .custom-header h1 {
        font-weight: 800;
        margin: 0;
        font-size: 2.5rem;
    }
    .custom-header p {
        margin-top: 0.5rem;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    /* Chat message aesthetic leveraging native theme */
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    /* Sidebar aesthetic */
    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.markdown("""
<div class="custom-header">
    <h1>🏗️ Indecimal Construction AI</h1>
    <p>Your transparent, grounded, and intelligent marketplace assistant.</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading Vectorstore...")
def load_vectorstore():
    # Load same embeddings model used in ingest
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    if os.path.exists("vectorstore"):
        vs = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
        return vs
    else:
        st.error("Vectorstore not found! Please run 'python ingest.py' first.")
        st.stop()
        
def get_llm():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        st.sidebar.warning("⚠️ Please add your OPENROUTER_API_KEY to the .env file.")
        st.stop()
    
    llm = ChatOpenAI(
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=api_key,
        model_name="google/gemini-2.5-flash", # Free tier model
        max_tokens=1024,
        default_headers={"HTTP-Referer": "http://localhost:8501", "X-Title": "Mini RAG"}
    )
    return llm

vectorstore = load_vectorstore()
llm = get_llm()

# --- Sidebar Layout ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2237/2237305.png", width=60)
    st.title("Transparency Panel")
    st.markdown("___")
    st.markdown("""
    **🛠️ Mini RAG Architecture**:
    - **Chunking**: Recursive (500 chars)
    - **Embeddings**: `all-MiniLM-L6-v2`
    - **Vector Store**: FAISS Index
    - **Generation**: Gemini-2.5 Flash
    
    *Strictly grounded against the Indecimal policy documents.*
    """)
    st.markdown("___")
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- Main Chat Area ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am the Indecimal assistant. Here is an example query you can try: *What are the operating principles of Indecimal?*", "context": None}]

# Render messages
for msg in st.session_state.messages:
    avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("context"):
            with st.expander("🔍 Show Retrieved Context", expanded=False):
                for idx, doc in enumerate(msg["context"]):
                    st.info(f"**Chunk {idx+1} [Source: {doc.metadata.get('source', 'Unknown')}]:**\n\n{doc.page_content}")

if prompt := st.chat_input("Ask about construction (e.g., 'What are the operating principles of Indecimal?')"):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt, "context": None})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    # Retrieve and generate
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Searching internal documents..."):
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            retrieved_docs = retriever.invoke(prompt)
            
            context_texts = "\n\n".join([f"[Doc {i+1}]: {doc.page_content}" for i, doc in enumerate(retrieved_docs)])
            
            prompt_template = ChatPromptTemplate.from_template("""
            You are a highly professional assistant for a construction marketplace called Indecimal.
            Answer the question explicitly and ONLY using the provided retrieved context.
            If the answer is not contained in the context, graciously say "I don't know based on the provided documents."
            Do not hallucinate or use external knowledge under any circumstances.

            Context:
            {context}

            Question:
            {question}
            """)
            
            chain = prompt_template | llm | StrOutputParser()
            
            try:
                response = chain.invoke({"context": context_texts, "question": prompt})
                st.markdown(response)
                
                with st.expander("🔍 Show Retrieved Context", expanded=False):
                    for idx, doc in enumerate(retrieved_docs):
                        st.success(f"**Chunk {idx+1} [Source: {doc.metadata.get('source', 'Unknown')}]:**\n\n{doc.page_content}")
                        
                st.session_state.messages.append({"role": "assistant", "content": response, "context": retrieved_docs})
            except Exception as e:
                st.error(f"API Error encountered: {str(e)}")
