import streamlit as st
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from extract_chunks import pdf_text
from llm_connect import generate_answer

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="NeuralPDF AI",
    page_icon="🤖",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

/* ---------- APP BACKGROUND ---------- */

.stApp{
    background: linear-gradient(
        135deg,
        #0f172a,
        #1e293b,
        #0f172a
    );
    color:white;
}

/* Hide Streamlit */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* Title */

.title{
    text-align:center;
    font-size:60px;
    font-weight:900;
    background:linear-gradient(
        90deg,
        #00f5ff,
        #7c3aed,
        #ff0080
    );
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

/* Subtitle */

.subtitle{
    text-align:center;
    color:#94a3b8;
    margin-bottom:30px;
}

/* Upload Card */

.upload-card{
    background:rgba(255,255,255,0.05);
    border:1px solid rgba(255,255,255,0.1);
    backdrop-filter:blur(20px);
    border-radius:20px;
    padding:20px;
}

/* Source Cards */

.source-card{
    background:rgba(255,255,255,0.05);
    border-radius:15px;
    padding:15px;
    margin-bottom:10px;
    border:1px solid rgba(255,255,255,0.1);
    transition:0.3s;
}

.source-card:hover{
    transform:translateY(-5px);
    box-shadow:0px 0px 20px #7c3aed;
}

/* Buttons */

.stButton > button{
    border-radius:12px;
    border:none;
    background:linear-gradient(
        90deg,
        #2563eb,
        #7c3aed
    );
    color:white;
}

/* Sidebar */

section[data-testid="stSidebar"]{
    background:#111827;
}

/* Metrics */

[data-testid="metric-container"]{
    background:rgba(255,255,255,0.05);
    border:1px solid rgba(255,255,255,0.1);
    padding:15px;
    border-radius:15px;
}

/* Chat */

[data-testid="stChatMessage"]{
    border-radius:15px;
    padding:10px;
}

/* Scrollbar */

::-webkit-scrollbar{
    width:8px;
}

::-webkit-scrollbar-thumb{
    background:#7c3aed;
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# CONFIG
# =====================================================

INDEX_PATH = "faiss_index.bin"
SENTENCE_PATH = "sentence.pkl"

# =====================================================
# MODEL
# =====================================================

@st.cache_resource
def load_model():
    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

model = load_model()

# =====================================================
# FAISS FUNCTIONS
# =====================================================

def create_embeddings(sentences):

    embeddings = model.encode(
        sentences
    ).astype("float32")

    faiss.normalize_L2(
        embeddings
    )

    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(embeddings)

    faiss.write_index(
        index,
        INDEX_PATH
    )

    with open(
        SENTENCE_PATH,
        "wb"
    ) as f:

        pickle.dump(
            sentences,
            f
        )

    return index


def load_embeddings():

    index = faiss.read_index(
        INDEX_PATH
    )

    with open(
        SENTENCE_PATH,
        "rb"
    ) as f:

        sentences = pickle.load(f)

    return index, sentences

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.title("🤖 NeuralPDF AI")

    st.markdown("---")

    st.success("FAISS Search")

    st.success("Semantic Retrieval")

    st.success("LLM Responses")

    st.success("PDF Intelligence")

    st.markdown("---")

    st.info(
        "Upload a PDF and chat with it."
    )

# =====================================================
# HEADER
# =====================================================

st.markdown(
    """
    <div class='title'>
    NeuralPDF AI
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class='subtitle'>
    Upload • Search • Understand • Chat
    </div>
    """,
    unsafe_allow_html=True
)

# =====================================================
# UPLOAD SECTION
# =====================================================

st.markdown(
    """
    <div class='upload-card'>
    <h3>📄 Upload PDF</h3>
    <p>Upload a document and start chatting instantly.</p>
    </div>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "",
    type=["pdf"]
)

# =====================================================
# PDF PROCESSING
# =====================================================

if uploaded_file:

    with open(
        "temp.pdf",
        "wb"
    ) as f:

        f.write(
            uploaded_file.getbuffer()
        )

    if (
        "indexed_file"
        not in st.session_state
        or
        st.session_state.indexed_file
        != uploaded_file.name
    ):

        with st.spinner(
            "🧠 Creating embeddings..."
        ):

            sentences = pdf_text(
                "temp.pdf"
            )

            index = create_embeddings(
                sentences
            )

            st.session_state.index = index
            st.session_state.sentences = sentences
            st.session_state.indexed_file = uploaded_file.name
            st.session_state.messages = []

        st.success(
            f"Indexed {len(sentences)} chunks"
        )

# =====================================================
# METRICS
# =====================================================

if "sentences" in st.session_state:

    c1,c2,c3 = st.columns(3)

    with c1:
        st.metric(
            "Chunks",
            len(
                st.session_state.sentences
            )
        )

    with c2:
        st.metric(
            "Model",
            "MiniLM"
        )

    with c3:
        st.metric(
            "Status",
            "Ready"
        )

# =====================================================
# QUICK ACTIONS
# =====================================================

st.markdown("### ⚡ Quick Prompts")

col1,col2,col3,col4 = st.columns(4)

with col1:
    st.button("Summarize PDF")

with col2:
    st.button("Key Insights")

with col3:
    st.button("Important Dates")

with col4:
    st.button("Generate Notes")

# =====================================================
# CHAT HISTORY
# =====================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:

    with st.chat_message(
        msg["role"]
    ):
        st.markdown(
            msg["content"]
        )

# =====================================================
# CHAT INPUT
# =====================================================

query = st.chat_input(
    "Ask anything about the PDF..."
)

if query:

    if "index" not in st.session_state:

        st.warning(
            "Upload PDF first."
        )

        st.stop()

    st.session_state.messages.append(
        {
            "role":"user",
            "content":query
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    index = st.session_state.index
    sentences = st.session_state.sentences

    with st.spinner(
        "🔍 Searching..."
    ):

        query_embedding = model.encode(
            [query]
        ).astype("float32")

        faiss.normalize_L2(
            query_embedding
        )

        k = min(
            3,
            len(sentences)
        )

        distances,indices = index.search(
            query_embedding,
            k
        )

        context = [
            sentences[i]
            for i in indices[0]
        ]

    history = "\\n".join(
        [
            f"{m['role']}:{m['content']}"
            for m in st.session_state.messages[-6:]
        ]
    )

    rag_prompt = f'''
Chat History:
{history}

Context:
{' '.join(context)}

Question:
{query}
'''

    with st.spinner(
        "🤖 Thinking..."
    ):

        answer = generate_answer(
            rag_prompt,
            context
        )

    with st.chat_message(
        "assistant"
    ):
        st.markdown(answer)

    st.session_state.messages.append(
        {
            "role":"assistant",
            "content":answer
        }
    )

    st.markdown(
        "## 📚 Retrieved Sources"
    )

    for i,chunk in enumerate(context):

        st.markdown(
            f"""
            <div class='source-card'>
            <h4>📄 Chunk {i+1}</h4>

            <b>Score:</b>
            {distances[0][i]:.3f}

            <br><br>

            {chunk}

            </div>
            """,
            unsafe_allow_html=True
        )

else:

    if "index" not in st.session_state:

        st.info(
            "👆 Upload a PDF to start chatting."
        )
