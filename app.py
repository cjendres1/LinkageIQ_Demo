import re
import pandas as pd
import numpy as np
import streamlit as st
import spacy
import recordlinkage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------------------------------------------------
# 0. MODEL LOADER
# -----------------------------------------------------------------------------
@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")

nlp = load_nlp()

# -----------------------------------------------------------------------------
# 1. SYNTHETIC NHANES DATASETS
# -----------------------------------------------------------------------------
@st.cache_data
def load_datasets():
    return pd.DataFrame([
        {
            "vt_id": "VT-101", 
            "first_name": "Rebecca", 
            "last_name": "Miller", 
            "age": 52, 
            "gender": "Female", 
            "sys_bp": 138, 
            "dia_bp": 88, 
            "hba1c": 6.2,
            "notes": "Patient reports mild hypertension and seasonal allergies."
        },
        {
            "vt_id": "VT-102", 
            "first_name": "Debra", 
            "last_name": "Johnson", 
            "age": 64, 
            "gender": "Female", 
            "sys_bp": 142, 
            "dia_bp": 90, 
            "hba1c": 7.1,
            "notes": "History of type 2 diabetes managed with metformin."
        },
        {
            "vt_id": "VT-103", 
            "first_name": "Elizabeth", 
            "last_name": "Johnson", 
            "age": 41, 
            "gender": "Female", 
            "sys_bp": 120, 
            "dia_bp": 78, 
            "hba1c": 5.4,
            "notes": "Normal baseline routine annual checkup."
        },
        {
            "vt_id": "VT-104", 
            "first_name": "Robert", 
            "last_name": "Davis", 
            "age": 58, 
            "gender": "Male", 
            "sys_bp": 130, 
            "dia_bp": 84, 
            "hba1c": 5.8,
            "notes": "Baseline screening normal, routine annual checkup."
        },
    ])

vt_db = load_datasets()

# -----------------------------------------------------------------------------
# 2. ADVANCED TEXT PROCESSING (Regex & spaCy)
# -----------------------------------------------------------------------------
def clean_text_regex(text: str) -> str:
    """Regex pipeline for text sanitization."""
    text = re.sub(r"[^\w\s]", " ", str(text))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

def spacy_lemmatize(text: str) -> str:
    """spaCy NLP pipeline for tokenization and lemmatization."""
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(lemmas)

def preprocess_pipeline(text: str) -> str:
    return spacy_lemmatize(clean_text_regex(text))

# -----------------------------------------------------------------------------
# 3. TF-IDF & COSINE SIMILARITY ENGINE
# -----------------------------------------------------------------------------
def compute_tfidf_similarity(query_text: str, corpus_series: pd.Series):
    """Calculates Cosine Similarity between intake notes and baseline notes."""
    processed_query = preprocess_pipeline(query_text)
    processed_corpus = corpus_series.apply(preprocess_pipeline).tolist()
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([processed_query] + processed_corpus)
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

# -----------------------------------------------------------------------------
# 4. RECORD LINKAGE ENGINE
# -----------------------------------------------------------------------------
def run_record_linkage(fl_record: dict, vt_df: pd.DataFrame):
    """Uses RecordLinkage library for deterministic and probabilistic field comparison."""
    fl_df = pd.DataFrame([fl_record])
    
    indexer = recordlinkage.Index()
    indexer.block("gender")  # Block on gender for execution efficiency
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
st.set_page_config(page_title="LinkageIQ_Demo", layout="wide")

st.title("LinkageIQ_Demo: Multi-Modal Patient Resolver")
st.caption("Demonstrating Regex, spaCy NLP, TF-IDF + Cosine Similarity, and RecordLinkage indexing.")

st.divider()

col_in, col_out = st.columns([1, 2])

with col_in:
    st.subheader("Florida Intake Record")
    fl_fn = st.text_input("First Name", "Becky")
    fl_ln = st.text_input("Last Name", "Miller")
    fl_age = st.number_input("Age", value=52)
    fl_gender = st.selectbox("Gender", ["Female", "Male"])
    
    st.markdown("**Florida Clinical Vitals:**")
    fl_sys_bp = st.number_input("Systolic BP (mmHg)", value=142)
    fl_dia_bp = st.number_input("Diastolic BP (mmHg)", value=92)
    fl_hba1c = st.number_input("HbA1c (%)", value=6.5, step=0.1)
    
    fl_notes = st.text_area(
        "Intake Clinical Notes", 
        "Patient complains of mild elevated BP and seasonal allergies."
    )
    
    st.divider()
    
    # Toggle for including unstructured notes in linkage score
    use_notes_in_matching = st.toggle(
        "Include Clinical Notes in Match Score", 
        value=False,
        help="If disabled, matching relies strictly on demographic fields (name, age, gender). Notes are displayed for reference only."
    )
    
    run_btn = st.button("Run Patient Linkage")

with col_out:
    st.subheader("Match Candidate Scoring")
    
    if run_btn:
        fl_payload = {"first_name": fl_fn, "last_name": fl_ln, "age": fl_age, "gender": fl_gender}
        
        # 1. Demographic Linkage
        linkage_res = run_record_linkage(fl_payload, vt_db)
        results_df = vt_db.copy()
        
        linkage_map = dict(zip(linkage_res["level_1"], linkage_res["linkage_score"]))
        results_df["recordlinkage_score"] = results_df.index.map(lambda x: linkage_map.get(x, 0.0))
        
        # 2. Conditional Unstructured Note Matching
        if use_notes_in_matching and fl_notes.strip():
            tfidf_scores = compute_tfidf_similarity(fl_notes, vt_db["notes"])
            results_df["tfidf_cosine_score"] = tfidf_scores
            results_df["composite_score"] = (results_df["recordlinkage_score"] * 0.6) + (results_df["tfidf_cosine_score"] * 0.4)
        else:
            results_df["tfidf_cosine_score"] = 0.0
            results_df["composite_score"] = results_df["recordlinkage_score"]
        
        st.session_state["search_results"] = results_df.sort_values(by="composite_score", ascending=False).to_dict(orient="records")
        st.session_state["confirmed_patient"] = None

    # Render Search Results
    if "search_results" in st.session_state:
        candidates = st.session_state["search_results"]
        
        for cand in candidates:
            with st.container(border=True):
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(f"**{cand['first_name']} {cand['last_name']}** (`{cand['vt_id']}`)")
                    st.caption(f"Age: {cand['age']} | Gender: {cand['gender']}")
                    st.info(f"**VT Baseline Notes:** {cand['notes']}")
                with c2:
                    st.metric("Composite Match", f"{cand['composite_score']*100:.1f}%")
                    st.caption(f"Demographic Score: {cand['recordlinkage_score']*100:.0f}%")
                    if use_notes_in_matching:
                        st.caption(f"Notes Similarity: {cand['tfidf_cosine_score']*100:.0f}%")
                    
                    if st.button("Confirm Match", key=f"confirm_{cand['vt_id']}"):
                        st.session_state["confirmed_patient"] = cand

    # Side-by-Side Vitals Comparison Table
    if st.session_state.get("confirmed_patient"):
        matched = st.session_state["confirmed_patient"]
        st.divider()
        st.success(f"Matched with Vermont Baseline Record: **{matched['vt_id']}** ({matched['first_name']} {matched['last_name']})")
        
        st.subheader("Clinical Vitals Comparison Table")
        
        comp_df = pd.DataFrame({
            "Metric": ["Systolic BP (mmHg)", "Diastolic BP (mmHg)", "HbA1c (%)"],
            "Vermont Baseline": [matched["sys_bp"], matched["dia_bp"], matched["hba1c"]],
            "Florida Current": [fl_sys_bp, fl_dia_bp, fl_hba1c],
            "Delta": [
                fl_sys_bp - matched["sys_bp"], 
                fl_dia_bp - matched["dia_bp"], 
                round(fl_hba1c - matched["hba1c"], 1)
            ]
        })
        st.dataframe(comp_df, hide_index=True, use_container_width=True)
