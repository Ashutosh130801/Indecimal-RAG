import time
import os
from dotenv import load_dotenv
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOpenAI
from transformers import pipeline

load_dotenv()

# Test queries based on the provided Indecimal documents
test_queries = [
    "What are the operating principles of Indecimal?",
    "How many quality checks does Indecimal perform during construction?",
    "What happens right after a customer raises a request?",
    "Does Indecimal provide guidance for home financing?",
    "What is Indecimal's policy on real-time construction progress tracking?",
    "How does Indecimal reduce hidden surprises in pricing?",
    "When exactly are contractor payments released?",
    "What core value does Indecimal promise to its customers?",
    "Are there any penalties for construction project delays?",
    "Whatkind of support is provided after the house is handed over?"
]

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)

def get_openrouter_llm():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        print("Warning: OPENROUTER_API_KEY not set. Skipping OpenRouter.")
        return None
    return ChatOpenAI(
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=api_key,
        model_name="google/gemini-2.5-flash", 
        max_tokens=1024,
    )

def get_local_llm():
    print("Loading Local LLM (Qwen/Qwen2.5-3B-Instruct). This may take a while...")
    try:
        hf_pipeline = pipeline(
            "text-generation",
            model="Qwen/Qwen2.5-3B-Instruct",
            max_new_tokens=256,
            device_map="auto" # uses GPU if available, else CPU
        )
        return HuggingFacePipeline(pipeline=hf_pipeline)
    except Exception as e:
        print(f"Failed to load local LLM: {e}")
        return None

def evaluate():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    openrouter_llm = get_openrouter_llm()
    local_llm = get_local_llm()
    
    prompt_template = ChatPromptTemplate.from_template("""
    You are a helpful assistant. Answer the question explicitly and ONLY using the provided retrieved context.
    If the answer is not contained in the context, say "I don't know based on the provided documents."

    Context:
    {context}

    Question:
    {question}
    """)

    results = []
    
    for q in test_queries:
        print(f"\nProcessing query: {q}")
        # Retrieve context
        docs = retriever.invoke(q)
        context_texts = "\n\n".join([f"[Doc {i+1}]: {doc.page_content}" for i, doc in enumerate(docs)])
        
        entry = {"Query": q, "OpenRouter Answer": "N/A", "OpenRouter Latency (s)": 0, "Local Answer": "N/A", "Local Latency (s)": 0}
        
        # Test OpenRouter
        if openrouter_llm:
            chain_or = prompt_template | openrouter_llm | StrOutputParser()
            try:
                start = time.time()
                ans = chain_or.invoke({"context": context_texts, "question": q})
                latency = time.time() - start
                entry["OpenRouter Answer"] = ans.replace('\n', ' ')
                entry["OpenRouter Latency (s)"] = round(latency, 2)
            except Exception as e:
                entry["OpenRouter Answer"] = f"Error: {e}"
                
        # Test Local
        if local_llm:
            chain_local = prompt_template | local_llm | StrOutputParser()
            try:
                start = time.time()
                ans = chain_local.invoke({"context": context_texts, "question": q})
                latency = time.time() - start
                # The pipeline may include the prompt in the output, so we need to parse it, but langchain wrapper handles it mostly
                entry["Local Answer"] = ans.replace('\n', ' ')
                entry["Local Latency (s)"] = round(latency, 2)
            except Exception as e:
                entry["Local Answer"] = f"Error: {e}"
                
        results.append(entry)

    df = pd.DataFrame(results)
    df.to_csv("evaluation_results.csv", index=False)
    print("\nEvaluation complete! Results saved to evaluation_results.csv")
    print(df.head())

if __name__ == "__main__":
    evaluate()
