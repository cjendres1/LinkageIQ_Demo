import re
import pandas as pd
import numpy as np
import streamlit as st
import spacy
import recordlinkage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy lightweight model
@st.cache_resource
def load_nlp():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        # Fetch and load the model directly if not found locally
        import spacy.cli
        spacy.cli.download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

nlp = load_nlp()

# -----------------------------------------------------------------------------
# 1. SYNTHETIC NHANES DATASETS
# -----------------------------------------------------------------------------
@st.cache_data
def load_datasets():
    vt_data = pd.DataFrame([
        {"vt_id": "VT-101", "first_name": "Rebecca", "last_name": "Miller-Smith", "age": 52, "gender": "Female", "notes": "Patient reports mild hypertension and seasonal allergies."},
        {"vt_id": "VT-102", "first_name": "Debra", "last_name": "Johnson", "age": 64, "gender": "Female", "notes": "History of type 2 diabetes managed with metformin."},
        {"vt_id": "VT-103", "first_name": "Robert", "last_name": "Davis", "age": 58, "gender": "Male", "notes": "Baseline screening normal, routine annual checkup."},
    ])
    return vt_data

vt_db = load_datasets()

# -----------------------------------------------------------------------------
# 2. ADVANCED TEXT PROCESSING (Regex & spaCy)
# -----------------------------------------------------------------------------
def clean_text_regex(text: str) -> str:
    """Regex pipeline for initial text sanitization."""
    text = re.sub(r"[^\w\s]", " ", str(text))  # Strip punctuation
    text = re.sub(r"\d+", "", text)            # Strip digits
    text = re.sub(r"\s+", " ", text)           # Collapse whitespace
    return text.strip().lower()

def spacy_lemmatize(text: str) -> str:
    """spaCy NLP pipeline for tokenization, stop-word removal, and lemmatization."""
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(lemmas)

def preprocess_pipeline(text: str) -> str:
    cleaned = clean_text_regex(text)
    return spacy_lemmatize(cleaned)

# -----------------------------------------------------------------------------
# 3. TF-IDF & COSINE SIMILARITY ENGINE
# -----------------------------------------------------------------------------
def compute_tfidf_similarity(query_text: str, corpus_series: pd.Series):
    """Calculates Cosine Similarity between intake notes and baseline notes."""
    processed_query = preprocess_pipeline(query_text)
    processed_corpus = corpus_series.apply(preprocess_pipeline).tolist()
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([processed_query] + processed_corpus)
    
    # Cosine similarity between query (index 0) and corpus (index 1:)
    cosine_sims = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    return cosine_sims

# -----------------------------------------------------------------------------
# 4. RECORD LINKAGE ENGINE
# -----------------------------------------------------------------------------
def run_record_linkage(fl_record: dict, vt_df: pd.DataFrame):
    """Uses RecordLinkage library for deterministic and probabilistic field comparison."""
    fl_df = pd.DataFrame([fl_record])
    
    indexer = recordlinkage.Index()
    indexer.block("gender")  # Block on gender for optimization
    candidate_pairs = indexer.index(fl_df, vt_df)
    
    compare = recordlinkage.Compare()
    compare.string("first_name", "first_name", method="jarowinkler", threshold=0.85, label="fn_sim")
    compare.string("last_name", "last_name", method="jarowinkler", threshold=0.85, label="ln_sim")
    compare.numeric("age", "age", method="linear", offset=2, scale=5, label="age_sim")
    
    features = compare.compute(candidate_pairs, fl_df, vt_df)
    features["linkage_score"] = features.sum(axis=1) / 3.0  # Normalized 0-1
    return features.reset_index()

# -----------------------------------------------------------------------------
# 5. STREAMLIT INTERFACE
# -----------------------------------------------------------------------------
st.set_page_config(page_title="NHANES Record Linkage Engine", layout="wide")

st.title("NHANES Multi-Modal Record Linkage Engine")
st.caption("Demonstrating Regex, spaCy NLP, TF-IDF + Cosine Similarity, and RecordLinkage indexing.")

st.divider()

col_in, col_out = st.columns([1, 2])

with col_in:
    st.subheader("Florida Intake Record")
    fl_fn = st.text_input("First Name", "Becky")
    fl_ln = st.text_input("Last Name", "Miller")
    fl_age = st.number_input("Age", value=52)
    fl_gender = st.selectbox("Gender", ["Female", "Male"])
    fl_notes = st.text_area("Intake Clinical Notes", "Patient complains of mild elevated BP and seasonal allergies.")
    
    run_btn = st.button("Run Hybrid Linkage")

with col_out:
    st.subheader("Match Candidate Scoring")
    
    if run_btn:
        fl_payload = {"first_name": fl_fn, "last_name": fl_ln, "age": fl_age, "gender": fl_gender}
        
        # 1. Field-based Linkage (RecordLinkage)
        linkage_res = run_record_linkage(fl_payload, vt_db)
        
        # 2. Text-based Linkage (TF-IDF & Cosine Similarity via spaCy + Regex)
        tfidf_scores = compute_tfidf_similarity(fl_notes, vt_db["notes"])
        
        # Combine Scoring
        results_df = vt_db.copy()
        results_df["tfidf_cosine_score"] = tfidf_scores
        
        # Merge linkage score mapped by VT index
        linkage_map = dict(zip(linkage_res["level_1"], linkage_res["linkage_score"]))
        results_df["recordlinkage_score"] = results_df.index.map(lambda x: linkage_map.get(x, 0.0))
        
        # Composite Match Score
        results_df["composite_score"] = (results_df["recordlinkage_score"] * 0.6) + (results_df["tfidf_cosine_score"] * 0.4)
        
        # Sort and Display Results
        display_df = results_df.sort_values(by="composite_score", ascending=False)
        
        for _, row in display_df.iterrows():
            with st.container(border=True):
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(f"**{row['first_name']} {row['last_name']}** (`{row['vt_id']}`)")
                    st.caption(f"Age: {row['age']} | Gender: {row['gender']}")
                    st.text(f"VT Baseline Notes: {row['notes']}")
                with c2:
                    st.metric("Composite Match", f"{row['composite_score']*100:.1f}%")
                    st.caption(f"Field Linkage: {row['recordlinkage_score']*100:.0f}%")
                    st.caption(f"Notes Cosine Sim: {row['tfidf_cosine_score']*100:.0f}%")
