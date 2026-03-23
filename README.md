# Mini RAG Application 🏗️

A complete, locally-indexed Retrieval-Augmented Generation (RAG) assistant for the Indecimal construction marketplace. It answers user questions strictly based on the provided internal company documents without hallucinating.

## 🚀 Features & Assignment Fulfillment
- **Document Chunking & Embedding**: Automatically chunks the provided internal policies and embeds them locally using `sentence-transformers/all-MiniLM-L6-v2`.
- **Local Vector Search**: Stores embeddings in a lightweight, localized `FAISS` vector index completely avoiding paid managed services like Pinecone.
- **LLM Grounding (No Hallucinations)**: Uses OpenRouter (Gemini-2.5-Flash) driven by a LangChain prompt template explicitly forbidding out-of-context answers.
- **Transparency & Explainability**: The Streamlit user interface explicitly surfaces all document chunks pulled from the Vector Store under a "Show Retrieved Context" expander for every answer.
- **Bonus Implementation included**: Includes a complete comparative evaluation (`eval_script.py`) running a local `Qwen/Qwen2.5-3B-Instruct` model (approx. 3 Billion parameters) natively on the CPU to contrast Answer Quality and Latency against the OpenRouter model!

## 📂 Repository Structure
- `download_docs.py`: Automates downloading the source documents from the provided Google Drive links securely.
- `ingest.py`: Parses the `.md` documents, recursively chunks them, and generates embeddings to construct the FAISS vector database.
- `app.py`: The Main Chatbot interface built with Streamlit. Handles user queries, fetches context, and safely generates the answers.
- `eval_script.py`: The bonus evaluation module. Tests 10 domain-specific queries on both the API-hosted LLM and a locally downloaded 3B Open-Source LLM and compiles `evaluation_results.csv`.
- `requirements.txt`: Master package list.

## ⚙️ Setup Instructions

### 1. Environment & Packages
We heavily recommend using a Python Virtual Environment (`venv`):
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys
Create a `.env` file in the root directory and add your OpenRouter API Key:
```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### 3. Initialize the Vector Store
Download the necessary text context and index it:
```bash
python download_docs.py
python ingest.py
```
*(You should now see a `vectorstore/` folder and a `docs/` folder populated with the assignment's data).*

### 4. Start the Chat Assistant
Launch the visual frontend:
```bash
streamlit run app.py
```

### 5. Start the Local LLM Bonus Evaluation
Compare the API model against a local 3B Parameter model running strictly on your hardware:
```bash
python eval_script.py
```
*(Note: Because this downloads a 3 Billion parameter LLM to your machine and runs it on CPU/GPU, this script may take several minutes to generate all 10 answers!)*
