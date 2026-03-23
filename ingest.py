import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def main():
    docs_dir = "docs"
    vectorstore_path = "vectorstore"
    
    # Load documents
    print("Loading documents...")
    documents = []
    if os.path.exists(docs_dir):
        for file in os.listdir(docs_dir):
            if file.endswith(".md"):
                file_path = os.path.join(docs_dir, file)
                print(f"Loading {file_path}")
                try:
                    loader = TextLoader(file_path, encoding='utf-8')
                    documents.extend(loader.load())
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    else:
        print(f"Directory {docs_dir} not found.")
        return

    if not documents:
        print("No documents found. Please run download_docs.py first.")
        return
        
    print(f"Loaded {len(documents)} pages.")
    
    # Chunk documents
    print("Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    
    # Generate embeddings and create vector store
    print("Generating embeddings and building FAISS index...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Save locally
    print(f"Saving FAISS index to {vectorstore_path}...")
    vectorstore.save_local(vectorstore_path)
    print("Ingestion complete.")

if __name__ == "__main__":
    main()
