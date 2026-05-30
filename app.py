import streamlit as st
import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer
from extract_chunks import pdf_text
from llm_connect import generate_answer

# ---------------- CONFIG ----------------
INDEX_PATH = "faiss_index.bin"
SENTENCE_PATH = "sentence.pkl"

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ---------------- FAISS FUNCTIONS ----------------
def create_embeddings(sentences):
    embeddings = model.encode(sentences).astype("float32")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)
    with open(SENTENCE_PATH, "wb") as f:
        pickle.dump(sentences, f)
    return index

def load_embeddings():
    index = faiss.read_index(INDEX_PATH)
    with open(SENTENCE_PATH, "rb") as f:
        sentences = pickle.load(f)
    return index, sentences

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📄")
st.title("📄 PDF Insight")

# ---------------- PDF UPLOAD ----------------
uploaded_file = st.file_uploader("Upload a PDF to get started", type=["pdf"])

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    if "indexed_file" not in st.session_state or st.session_state.indexed_file != uploaded_file.name:
        with st.spinner("Processing PDF..."):
            sentences = pdf_text("temp.pdf")
            index = create_embeddings(sentences)
            st.session_state.indexed_file = uploaded_file.name
            st.session_state.sentences = sentences
            st.session_state.index = index
            st.session_state.messages = []  # clear chat on new PDF
        st.success(f"Ready! {len(sentences)} chunks indexed.")

    else:
        if "index" not in st.session_state:
            st.session_state.index, st.session_state.sentences = load_embeddings()

# ---------------- CHAT HISTORY ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------- CHAT INPUT ----------------
query = st.chat_input("Ask something about the PDF")

if query:
    # Guard: no PDF uploaded yet
    if "index" not in st.session_state:
        st.warning("Please upload a PDF first.")
        st.stop()

    index = st.session_state.index
    sentences = st.session_state.sentences

    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Retrieve chunks
    with st.spinner("Searching..."):
        query_embedding = model.encode([query]).astype("float32")
        faiss.normalize_L2(query_embedding)
        k = min(3, len(sentences))
        distances, indices = index.search(query_embedding, k)
        context = [sentences[i] for i in indices[0]]

    # Build prompt with chat history
    history = "\n".join(
        f"{msg['role']}: {msg['content']}"
        for msg in st.session_state.messages[-6:]
    )

    rag_prompt = f"""
Chat History:
{history}

Context:
{' '.join(context)}

User Question:
{query}
"""

    # Generate answer
    with st.spinner("Thinking..."):
        answer = generate_answer(rag_prompt, context)

    # Show assistant response
    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Debug section
    with st.expander("Retrieved Chunks"):
        for i, chunk in enumerate(context):
            st.caption(f"Chunk {i+1} — score: {distances[0][i]:.3f}")
            st.write(chunk)
            st.divider()
else:
    if "index" not in st.session_state:
        st.info("👆 Upload a PDF above to start chatting.")
