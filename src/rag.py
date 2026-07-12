"""
rag.py - Medical RAG System using Knowledge Graph + FAISS + Gemini
Run: python src/rag.py
"""

import os
import json
import pickle
import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer
import faiss
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

OUTPUT_DIR = "output"
os.makedirs(f"{OUTPUT_DIR}/faiss_index", exist_ok=True)


# ════════════════════════════════════════════════════════════
# PART 1: Convert Graph to Text Documents
# ════════════════════════════════════════════════════════════

def graph_to_documents(G):
    """
    Turn each disease node into a text passage.
    Example output:
      Disease: Diabetes
      Description: A metabolic disease...
      Symptoms: fatigue (severity 5), polyuria (severity 6)
      Precautions: avoid sugar; exercise daily
    """
    print("Converting graph to documents...")
    docs = []

    for node, data in G.nodes(data=True):
        if data.get("type") != "Disease":
            continue

        disease     = node
        description = data.get("description", "")
        symptoms    = []
        precautions = []

        for _, neighbor, edata in G.out_edges(node, data=True):
            rel = edata.get("relation", "")
            if rel == "HAS_SYMPTOM":
                sev = edata.get("weight", 3)
                symptoms.append(f"{neighbor} (severity {sev})")
            elif rel == "HAS_PRECAUTION":
                precautions.append(neighbor)

        # Build text passage
        text = f"Disease: {disease}\n"
        if description:
            text += f"Description: {description}\n"
        if symptoms:
            text += f"Symptoms: {', '.join(symptoms)}\n"
        if precautions:
            text += f"Precautions: {'; '.join(precautions)}\n"

        docs.append({
            "id":       disease,
            "text":     text,
            "disease":  disease,
            "symptoms": symptoms,
        })

    # Also add symptom → disease docs (reverse lookup)
    for node, data in G.nodes(data=True):
        if data.get("type") != "Symptom":
            continue

        diseases = [
            tgt for _, tgt, ed in G.out_edges(node, data=True)
            if ed.get("relation") == "INDICATES"
        ]
        if diseases:
            text = (
                f"Symptom: {node}\n"
                f"Severity: {data.get('severity', 3)} out of 7\n"
                f"Diseases with this symptom: {', '.join(diseases)}\n"
            )
            docs.append({"id": f"symptom_{node}", "text": text})

    print(f"✓ Created {len(docs)} documents")
    return docs


# ════════════════════════════════════════════════════════════
# PART 2: Build FAISS Index
# ════════════════════════════════════════════════════════════

def build_faiss_index(docs, model_name="all-MiniLM-L6-v2"):
    print(f"\nLoading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)

    texts = [d["text"] for d in docs]
    print(f"Embedding {len(texts)} documents...")

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    # Build FAISS index
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # Inner Product = cosine on normalized vecs
    index.add(embeddings)

    # Save index and docs
    faiss.write_index(index, f"{OUTPUT_DIR}/faiss_index/index.faiss")
    with open(f"{OUTPUT_DIR}/faiss_index/docs.pkl", "wb") as f:
        pickle.dump(docs, f)

    print(f"✓ FAISS index built — {index.ntotal} vectors")
    print(f"✓ Saved to output/faiss_index/")
    return index, docs, model


# ════════════════════════════════════════════════════════════
# PART 3: Retrieve relevant documents
# ════════════════════════════════════════════════════════════

def retrieve(query, index, docs, model, top_k=5):
    q_emb = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    scores, idxs = index.search(q_emb, top_k)

    results = []
    for i in range(top_k):
        results.append({
            "score": float(scores[0][i]),
            "doc":   docs[idxs[0][i]],
        })
    return results


# ════════════════════════════════════════════════════════════
# PART 4: Ask Gemini LLM
# ════════════════════════════════════════════════════════════

def ask_gemini(question, context):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "ERROR: Add GEMINI_API_KEY to your .env file"

    client = genai.Client(api_key=api_key)

    prompt = f"""You are a medical information assistant.
Answer the question using ONLY the context below.
Always recommend consulting a real doctor.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=prompt,
    )
    return response.text


# ════════════════════════════════════════════════════════════
# PART 5: Full RAG Pipeline (WITHOUT GEMINI)
# ════════════════════════════════════════════════════════════

def ask(question, index, docs, model, top_k=5):
    # Step 1: Retrieve top-k docs
    results = retrieve(question, index, docs, model, top_k)

    # Step 2: Build context string
    context = "\n\n---\n\n".join(
        f"[Source {i+1}]\n{r['doc']['text']}"
        for i, r in enumerate(results)
    )

    # Step 3: Return retrieved knowledge directly
    answer = context

    # Step 4: Show sources
    sources = [r["doc"]["id"] for r in results]

    return answer, sources


# ════════════════════════════════════════════════════════════
# MAIN - Run everything
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Load the graph ──────────────────────────────────────
    print("Loading Knowledge Graph...")
    with open(f"{OUTPUT_DIR}/medical_kg.json") as f:
        graph_data = json.load(f)
    G = nx.node_link_graph(graph_data, directed=True)
    print(f"✓ Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # ── Convert graph → documents ───────────────────────────
    docs = graph_to_documents(G)

    # ── Build FAISS index ───────────────────────────────────
    index, docs, embed_model = build_faiss_index(docs)

    # ── Test questions ──────────────────────────────────────
    questions = [
        "What are the symptoms of diabetes?",
        "Which diseases cause fever and fatigue?",
        "What precautions should I take for hypertension?",
    ]

    print("\n" + "="*55)
    print("  RAG SYSTEM READY — Testing Questions")
    print("="*55)

    for q in questions:
        print(f"\nQ: {q}")
        print("-"*50)
        answer, sources = ask(q, index, docs, embed_model)
        print(answer)
        print(f"\nSources: {sources}")
        print("="*55)
