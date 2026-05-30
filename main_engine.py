from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle
import os

from extract_chunks import pdf_text
from llm_connect import generate_answer

model=SentenceTransformer("all-MiniLM-L6-v2")

INDEX_PATH="faiss_index.bin"
SENTENCE_PATH="sentence.pkl"

sentences=pdf_text("C:\\Users\\ajayg\\Downloads\\Python_developer_resume.pdf")
print("text extracted")

def create_embeddings(sentences):
    embeddings=model.encode(sentences)
    dimensions=embeddings.shape[1]
    faiss.normalize_L2(embeddings)
    index=faiss.IndexFlatIP(dimensions)
    index.add(np.array(embeddings))

    #save indexes and sentences
    faiss.write_index(index,INDEX_PATH)

    with open(SENTENCE_PATH,"wb") as f:
        pickle.dump(sentences,f)
    print("index saved to disk")

    return index

def load_embeddings():
    index=faiss.read_index(INDEX_PATH)
    with open(SENTENCE_PATH,"rb") as f:
        sentences=pickle.load(f)
    print("index loaded from disk")
    return index,sentences

# ── Load or build ──────────────────────────────────────
if os.path.exists(INDEX_PATH) and os.path.exists(SENTENCE_PATH):
    index, sentences = load_embeddings()
else:
    sentences = pdf_text("C:\\Users\\ajayg\\Downloads\\Python_developer_resume.pdf")
    print(f"Text extracted — {len(sentences)} chunks")
    index = create_embeddings(sentences)

# index=create_embeddings(sentences)
while True:
    input_query=input("ASK: ")
    if input_query.lower() =="exit":
        break
    else:
        query=model.encode([input_query])
        faiss.normalize_L2(query)
        k = 3
        distances, indices = index.search(np.array(query),k)
        context=[]
        for i in range(k):
            idx=indices[0][i]
            context.append(sentences[idx])
            # print("distance:",distances[0][i])
        print("context sent to llm")
        print(generate_answer(input_query,context))
