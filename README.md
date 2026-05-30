# RAG-PDF-Question-Answering-System

A Retrieval-Augmented Generation (RAG) application that lets you upload any PDF and ask questions about it in a chat interface. Built with FAISS for semantic search and Groq (LLaMA 3.3 70B) for answer generation.

---

## How it works

1. Upload a PDF — the text is extracted, chunked into sentences, and embedded using `all-MiniLM-L6-v2`
2. Embeddings are stored in a FAISS index with cosine similarity
3. When you ask a question, the top 3 most relevant chunks are retrieved
4. Those chunks are sent to the LLM as context to generate a grounded answer

```
PDF → Extract text → Sentence chunks → Embeddings → FAISS index
                                                          ↓
Answer ← LLM (LLaMA 3.3 70B) ← Top-k chunks ← Query embedding
```

---

## Features

- Upload any PDF and chat with it instantly
- Sentence-level chunking with overlap for better context
- Cosine similarity search with FAISS
- Chat history included in every prompt
- Retrieved chunks visible in debug expander with similarity scores
- Index persists across sessions — no re-embedding on reload

---

## Tech stack

| Layer | Tool |
|---|---|
| Frontend | Streamlit |
| Embeddings | SentenceTransformers (`all-MiniLM-L6-v2`) |
| Vector search | FAISS (`IndexFlatIP`) |
| PDF extraction | pdfplumber |
| LLM | Groq API (LLaMA 3.3 70B) |
| Text chunking | NLTK sentence tokenizer |

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/your-username/rag-demo.git
cd rag-demo
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your API key

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free Groq API key at [console.groq.com](https://console.groq.com)

### 4. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Project structure

```
rag_demo/
├── app.py              # Streamlit UI + session state management
├── extract_chunks.py   # PDF text extraction and sentence chunking
├── llm_connect.py      # Groq API call and prompt construction
├── main_engine.py      # CLI version for testing
├── requirements.txt
├── .env                # API key (not committed)
└── .gitignore
```

---

## Requirements

```
streamlit
sentence-transformers
faiss-cpu
pdfplumber
nltk
groq
python-dotenv
```

---

## Notes

- `faiss_index.bin` and `sentence.pkl` are auto-generated on first upload — not committed to the repo
- Uploading a new PDF automatically clears the previous index and chat history
- For scanned PDFs (image-based), text extraction may be incomplete — OCR support not included



IMAGES:
<img width="1916" height="964" alt="Screenshot 2026-05-30 192301" src="https://github.com/user-attachments/assets/1fecc313-373b-4a03-aac3-413116ff8ebb" />

<img width="1912" height="961" alt="Screenshot 2026-05-30 192320" src="https://github.com/user-attachments/assets/16a0e659-7c55-47cd-9f25-60868baae4c8" />

<img width="1915" height="962" alt="Screenshot 2026-05-30 192343" src="https://github.com/user-attachments/assets/b4f26f3b-0c12-4bd9-877b-644c8a66ab4c" />


