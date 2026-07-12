"""
build_kg.py - Build Medical Knowledge Graph from CSV files
Run: python src/build_kg.py
"""

import pandas as pd
import networkx as nx
import json
import os

# ── Paths ──────────────────────────────────────────────────────────
DATA_DIR   = "data"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Step 1: Load CSV files ──────────────────────────────────────────
print("Loading data...")

# Disease + Symptoms
df = pd.read_csv(f"{DATA_DIR}/dataset.csv")
df.columns = df.columns.str.strip()

# Disease descriptions
desc_df = pd.read_csv(f"{DATA_DIR}/symptom_Description.csv", header=None)
desc_df.columns = ["disease", "description"]

# Precautions
prec_df = pd.read_csv(f"{DATA_DIR}/symptom_precaution.csv", header=None)
prec_df.columns = ["disease", "p1", "p2", "p3", "p4"]

# Symptom severity
sev_df = pd.read_csv(f"{DATA_DIR}/Symptom-severity.csv")
sev_df.columns = sev_df.columns.str.strip()
sev_df.columns = ["symptom", "weight"]
sev_df["symptom"] = sev_df["symptom"].str.strip().str.replace("_", " ")

print(f"✓ Loaded {df['Disease'].nunique()} diseases")

# ── Step 2: Build the Graph ─────────────────────────────────────────
print("\nBuilding Knowledge Graph...")

G = nx.DiGraph()

# Severity lookup
sev_map = dict(zip(sev_df["symptom"], sev_df["weight"]))

# Description lookup
desc_df["disease"] = desc_df["disease"].str.strip()
desc_map = dict(zip(desc_df["disease"], desc_df["description"]))

# Precaution lookup
prec_df["disease"] = prec_df["disease"].str.strip()

# Disease column
disease_col = [c for c in df.columns if "disease" in c.lower()][0]
symptom_cols = [c for c in df.columns if "symptom" in c.lower()]

for _, row in df.iterrows():
    disease = str(row[disease_col]).strip()

    # Add disease node
    G.add_node(disease,
               type="Disease",
               description=desc_map.get(disease, ""))

    # Add symptom nodes and edges
    for col in symptom_cols:
        symptom = str(row[col]).strip().replace("_", " ")
        if symptom and symptom != "nan":
            severity = sev_map.get(symptom, 3)
            G.add_node(symptom, type="Symptom", severity=severity)
            G.add_edge(disease, symptom, relation="HAS_SYMPTOM", weight=severity)
            G.add_edge(symptom, disease, relation="INDICATES")

# Add precautions
for _, row in prec_df.iterrows():
    disease = row["disease"]
    for col in ["p1", "p2", "p3", "p4"]:
        prec = str(row[col]).strip()
        if prec and prec != "nan":
            G.add_node(prec, type="Precaution")
            G.add_edge(disease, prec, relation="HAS_PRECAUTION")

# ── Step 3: Save the Graph ──────────────────────────────────────────
print("\nSaving graph...")

graph_data = nx.node_link_data(G)
with open(f"{OUTPUT_DIR}/medical_kg.json", "w") as f:
    json.dump(graph_data, f, indent=2)

# ── Step 4: Print Stats ─────────────────────────────────────────────
diseases    = [n for n,d in G.nodes(data=True) if d.get("type")=="Disease"]
symptoms    = [n for n,d in G.nodes(data=True) if d.get("type")=="Symptom"]
precautions = [n for n,d in G.nodes(data=True) if d.get("type")=="Precaution"]

print("\n" + "="*40)
print("  KNOWLEDGE GRAPH BUILT ✓")
print("="*40)
print(f"  Diseases    : {len(diseases)}")
print(f"  Symptoms    : {len(symptoms)}")
print(f"  Precautions : {len(precautions)}")
print(f"  Total nodes : {G.number_of_nodes()}")
print(f"  Total edges : {G.number_of_edges()}")
print(f"\n  Saved to: output/medical_kg.json")
print("="*40)

# ── Step 5: Quick lookup test ───────────────────────────────────────
print("\nExample - Symptoms of Diabetes:")
for _, symptom, data in G.out_edges("Diabetes", data=True):
    if data.get("relation") == "HAS_SYMPTOM":
        print(f"  • {symptom} (severity: {data.get('weight', '?')})")
