Medical RAG System:
A medical question-answering system that uses a Knowledge Graph, FAISS vector search, and Gemini AI to answer questions about diseases, symptoms, and precautions.

Project Structure:

```
Medical-RAG/
├── src/
│   ├── build_kg.py       # Builds the Knowledge Graph from CSV files
│   └── rag.py            # RAG pipeline: embed → retrieve → generate
├── data/
│   ├── dataset.csv                # Disease → Symptom mappings
│   ├── symptom_Description.csv    # Disease descriptions
│   ├── symptom_precaution.csv     # Precautions per disease
│   └── Symptom-severity.csv       # Severity scores (1-7) per symptom
├── datasets/
│   └── SLAKE/                     # Medical imaging dataset (Phase 2)
│       ├── imgs/                  # CT, MRI, X-Ray scans
│       ├── train.json
│       ├── validation.json
│       └── test.json
├── output/
│   └── medical_kg.json   # Generated knowledge graph
├── output.txt            # Sample run output
└── requirements.txt
```

--> Phase 1 (text RAG) is complete. Phase 2 (multimodal imaging with SLAKE) is in progress.

How It Works:

Step 1 — Build the Knowledge Graph
Reads 4 CSV files and builds a graph with 3 node types: Disease, Symptom, Precaution. Edges are typed (HAS_SYMPTOM, INDICATES, HAS_PRECAUTION) and symptoms carry severity weights (1–7).

Step 2 — Convert to Documents
Each disease becomes a text passage. Symptoms also get reverse-lookup documents so you can search by symptom, not just disease.

Step 3 — FAISS Vector Index
All documents are embedded with sentence-transformers and indexed in FAISS for similarity search.

Step 4 — Retrieve and Answer
Your question gets embedded, top-5 matching documents are retrieved, and Gemini generates an answer using them as context.

Setup:

1. Clone the repo
```
bashgit clone https://github.com/AishwaryaAryan/Medical-RAG.git
cd Medical-RAG
```

3. Install dependencies
```
bashpip install sentence-transformers faiss-cpu google-genai python-dotenv networkx pandas numpy
```
5. Add your Gemini API key
```
Create a .env file:
GEMINI_API_KEY=your_key_here
```

7. Build the Knowledge Graph
```
bashpython src/build_kg.py
```

9. Run the RAG system
```
bashpython src/rag.py
```

Example Output:
Q: What are the symptoms of diabetes?

[Source 1]
Disease: Diabetes
Description: Diabetes is a disease that occurs when your blood glucose is too high...
Symptoms: fatigue (severity 4), polyuria (severity 4), blurred vision (severity 5)
Precautions: have balanced diet; exercise; consult doctor; follow up

[Source 2]
Symptom: irregular sugar level
Severity: 5 out of 7
Diseases with this symptom: Diabetes
