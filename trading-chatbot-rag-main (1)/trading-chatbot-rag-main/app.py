import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from transformers import pipeline

# Page configuration
st.set_page_config(
    page_title="Trading Assistant Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Initialize the embedding model
@st.cache_resource
def load_embedding_model():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

# Initialize the text generation model
@st.cache_resource
def load_qa_model():
    qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")
    return qa_pipeline

# Initialize ChromaDB
@st.cache_resource
def load_vector_db():
    client = chromadb.Client(Settings(
        persist_directory="./chroma_db",
        anonymized_telemetry=False
    ))
    return client

# Function to load documents
def load_documents_to_db(client, embedding_model):
    try:
        collection = client.get_collection(name="trading_docs")
        return collection
    except:
        collection = client.create_collection(name="trading_docs")
        
        doc_path = "documents/trading_basics.txt"
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        chunks = content.strip().split('\n\n')
        
        for i, chunk in enumerate(chunks):
            if chunk.strip():
                embedding = embedding_model.encode(chunk).tolist()
                collection.add(
                    documents=[chunk],
                    embeddings=[embedding],
                    ids=[f"doc_{i}"]
                )
        
        return collection

# Search function
def search_knowledge_base(query, collection, embedding_model, top_k=2):
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results['documents'][0]

# Answer generation
def generate_answer(question, context_docs, qa_model):
    context = "\n\n".join(context_docs)
    
    prompt = f"""Based on the following context, answer the question clearly and concisely. If the answer is not in the context, say "I don't have information about that in my knowledge base."

Context:
{context}

Question: {question}

Answer:"""
    
    answer = qa_model(prompt, max_length=200, do_sample=False)[0]['generated_text']
    return answer

# Main App
st.title("🤖 Trading Assistant Chatbot")
st.caption("Powered by RAG (Retrieval-Augmented Generation) with Hugging Face & ChromaDB")

# Sidebar with info
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This chatbot uses:
    - **Vector Database**: ChromaDB
    - **Embeddings**: Sentence Transformers
    - **LLM**: FLAN-T5 (Hugging Face)
    - **Architecture**: RAG (Retrieval-Augmented Generation)
    """)
    
    st.write("---")
    
    st.header("📊 Stats")
    if 'messages' in st.session_state:
        st.metric("Questions Asked", len([m for m in st.session_state.messages if m["role"] == "user"]))
    
    st.write("---")
    
    st.header("💡 Try asking:")
    st.write("- What is CFD trading?")
    st.write("- What is risk management?")
    st.write("- What is a pip?")

# Load models
with st.spinner("🔄 Loading AI models... (First run may take 1-2 minutes)"):
    embedding_model = load_embedding_model()
    qa_model = load_qa_model()
    chroma_client = load_vector_db()
    collection = load_documents_to_db(chroma_client, embedding_model)

st.success("✅ System ready! Ask me anything about trading.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me about trading..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Searching knowledge base..."):
            relevant_docs = search_knowledge_base(prompt, collection, embedding_model)
            answer = generate_answer(prompt, relevant_docs, qa_model)
            
            st.markdown(answer)
            
            with st.expander("📚 View retrieved sources"):
                for i, doc in enumerate(relevant_docs):
                    st.info(f"**Source {i+1}:**\n\n{doc}")
    
    st.session_state.messages.append({"role": "assistant", "content": answer})

# Footer
st.write("---")
st.caption("Built with Streamlit, Hugging Face, and ChromaDB | RAG-based Trading Assistant")