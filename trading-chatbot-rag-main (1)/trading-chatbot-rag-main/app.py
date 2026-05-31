import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from pypdf import PdfReader
import chromadb
from chromadb.config import Settings
from chromadb import PersistentClient
import uuid
import hashlib
import time

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="TradingGPT - Financial AI Assistant",
    page_icon="📈",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown(
    """
<style>

.block-container {
    padding-top: 2rem;
}

.stChatMessage {
    border-radius: 12px;
    padding: 10px;
}

</style>
""",
    unsafe_allow_html=True
)

# ==========================================
# LOAD MODELS
# ==========================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "BAAI/bge-base-en-v1.5"
    )


@st.cache_resource
def load_qa_pipeline():

    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        framework="pt"
    )


embedding_model = load_embedding_model()

qa_pipeline = load_qa_pipeline()

# ==========================================
# CHROMADB PERSISTENT DATABASE
# ==========================================

client = PersistentClient(
    path="./chroma_db",
    settings=Settings(
        anonymized_telemetry=False
    )
)

collection_name = "trading_docs"

try:

    collection = client.get_collection(
        name=collection_name
    )

except:

    collection = client.create_collection(
        name=collection_name
    )

# ==========================================
# SESSION STATE
# ==========================================

if "processed_files" not in st.session_state:

    st.session_state.processed_files = set()

if "messages" not in st.session_state:

    st.session_state.messages = []

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_file_hash(file):

    file_bytes = file.getvalue()

    return hashlib.md5(
        file_bytes
    ).hexdigest()


def chunk_text(
    text,
    chunk_size=300,
    overlap=80
):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():

            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def process_txt_file(file):

    content = file.read().decode(
        "utf-8"
    )

    return content, []


def process_pdf_file(file):

    reader = PdfReader(file)

    full_text = ""

    page_mapping = []

    for page_number, page in enumerate(
        reader.pages
    ):

        text = page.extract_text()

        if text:

            full_text += text + "\n"

            page_mapping.append(
                {
                    "page": page_number + 1,
                    "text": text
                }
            )

    return full_text, page_mapping


def store_chunks(
    chunks,
    source_name,
    page_mapping=None
):

    for idx, chunk in enumerate(chunks):

        embedding = embedding_model.encode(
            chunk
        ).tolist()

        unique_id = str(
            uuid.uuid4()
        )

        page_number = None

        if page_mapping:

            for mapping in page_mapping:

                if chunk[:50] in mapping["text"]:

                    page_number = mapping["page"]

                    break

        metadata = {
            "source": source_name
        }

        if page_number:

            metadata["page"] = page_number

        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[unique_id]
        )


def rewrite_query(query):

    prompt = f"""
Rewrite this trading-related question
into a more detailed search query.

Question:
{query}

Rewritten Query:
"""

    response = qa_pipeline(
        prompt,
        max_new_tokens=50,
        do_sample=False
    )

    return response[0]["generated_text"]


def retrieve_documents(
    query,
    top_k=5
):

    query_embedding = embedding_model.encode(
        query
    ).tolist()

    results = collection.query(
        query_embeddings=[
            query_embedding
        ],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    documents = results["documents"][0]

    metadatas = results["metadatas"][0]

    distances = results["distances"][0]

    filtered_docs = []
    filtered_metadata = []
    filtered_scores = []

    for doc, meta, score in zip(
        documents,
        metadatas,
        distances
    ):

        if score < 1.2:

            filtered_docs.append(doc)

            filtered_metadata.append(meta)

            filtered_scores.append(score)

    return (
        filtered_docs,
        filtered_metadata,
        filtered_scores
    )


def generate_answer(
    question,
    context_docs,
    chat_history=""
):

    context = "\n\n".join(
        context_docs
    )

    prompt = f"""
You are TradingGPT,
an expert financial AI assistant.

Answer ONLY using the provided context.

If the answer is unavailable,
say:
"I could not find enough information in the uploaded documents."

Provide:
- concise answers
- professional explanations
- accurate financial reasoning

Previous Conversation:
{chat_history}

Context:
{context}

Question:
{question}

Answer:
"""

    response = qa_pipeline(
        prompt,
        max_new_tokens=200,
        truncation=True,
        do_sample=False
    )

    return response[0]["generated_text"]

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.title("📚 TradingGPT")

    uploaded_files = st.file_uploader(
        "Upload TXT or PDF Files",
        type=["txt", "pdf"],
        accept_multiple_files=True
    )

    if st.button("Process Documents"):

        if uploaded_files:

            with st.spinner(
                "Processing documents..."
            ):

                for uploaded_file in uploaded_files:

                    file_hash = get_file_hash(
                        uploaded_file
                    )

                    if file_hash in st.session_state.processed_files:

                        st.warning(
                            f"{uploaded_file.name} already uploaded."
                        )

                        continue

                    file_name = uploaded_file.name

                    # ======================
                    # TXT
                    # ======================

                    if file_name.endswith(".txt"):

                        content, page_mapping = process_txt_file(
                            uploaded_file
                        )

                    # ======================
                    # PDF
                    # ======================

                    elif file_name.endswith(".pdf"):

                        content, page_mapping = process_pdf_file(
                            uploaded_file
                        )

                    else:

                        continue

                    chunks = chunk_text(
                        content,
                        chunk_size=300,
                        overlap=80
                    )

                    store_chunks(
                        chunks,
                        file_name,
                        page_mapping
                    )

                    st.session_state.processed_files.add(
                        file_hash
                    )

                st.success(
                    "Documents processed successfully!"
                )

        else:

            st.warning(
                "Please upload files."
            )

    # ==========================================
    # CLEAR DATABASE
    # ==========================================

    if st.button("Clear Database"):

        client.delete_collection(
            collection_name
        )

        collection = client.create_collection(
            name=collection_name
        )

        st.session_state.processed_files.clear()

        st.success(
            "Database cleared successfully!"
        )

    st.markdown("---")

    st.subheader("📊 Database Statistics")

    st.metric(
        "Stored Chunks",
        collection.count()
    )

    st.markdown("---")

    st.subheader("💡 Example Questions")

    st.write(
        "- What are liquidity zones?"
    )

    st.write(
        "- Explain leverage"
    )

    st.write(
        "- Summarize chapter 2"
    )

    st.write(
        "- Explain market structure"
    )

# ==========================================
# MAIN UI
# ==========================================

st.title(
    "📈 TradingGPT - Financial AI Assistant"
)

st.caption(
    """
Advanced RAG System using:
- BGE Embeddings
- ChromaDB
- FLAN-T5
- Semantic Retrieval
"""
)

# ==========================================
# CHAT HISTORY
# ==========================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

# ==========================================
# USER INPUT
# ==========================================

prompt = st.chat_input(
    "Ask questions about your trading books..."
)

if prompt:

    # ======================================
    # SAVE USER MESSAGE
    # ======================================

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):

        st.markdown(prompt)

    # ======================================
    # CHAT HISTORY
    # ======================================

    history = ""

    for msg in st.session_state.messages[-6:]:

        history += f"""
{msg['role']}:
{msg['content']}
"""

    # ======================================
    # QUERY REWRITING
    # ======================================

    rewritten_query = rewrite_query(
        prompt
    )

    # ======================================
    # RETRIEVAL
    # ======================================

    relevant_docs, metadatas, scores = retrieve_documents(
        rewritten_query,
        top_k=5
    )

    # ======================================
    # GENERATION
    # ======================================

    answer = generate_answer(
        prompt,
        relevant_docs,
        history
    )

    # ======================================
    # STREAMING RESPONSE
    # ======================================

    with st.chat_message("assistant"):

        response_placeholder = st.empty()

        streamed_text = ""

        for word in answer.split():

            streamed_text += word + " "

            response_placeholder.markdown(
                streamed_text
            )

            time.sleep(0.02)

        # ==================================
        # SOURCES
        # ==================================

        with st.expander(
            "📚 Retrieved Sources"
        ):

            for i, doc in enumerate(
                relevant_docs
            ):

                source = metadatas[i].get(
                    "source",
                    "Unknown"
                )

                page = metadatas[i].get(
                    "page",
                    "N/A"
                )

                score = round(
                    scores[i],
                    4
                )

                st.info(
                    f"""
SOURCE:
{source}

PAGE:
{page}

SIMILARITY SCORE:
{score}

CONTENT:
{doc[:1000]}
"""
                )

    # ======================================
    # SAVE ASSISTANT MESSAGE
    # ======================================

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.caption(
    """
TradingGPT v2

Features:
- Persistent ChromaDB
- Query Rewriting
- Semantic Search
- Similarity Filtering
- Streaming Responses
- Source Grounding
- Multi-PDF Support
"""
)
