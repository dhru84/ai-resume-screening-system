"""Streamlit UI for the AI Resume Screening System.  Run: streamlit run app.py"""
from pathlib import Path

import pandas as pd
import streamlit as st

from src.extractor import EDU_LABELS, parse_jd
from src.parser import extract_text
from src.scorer import DEFAULT_WEIGHTS, screen, to_dataframe

ROOT = Path(__file__).parent
st.set_page_config(page_title="AI Resume Screener", page_icon="📄", layout="wide")
st.title("📄 AI Resume Screening System")
st.caption("Decision-support for recruiters: ranks candidates and explains *why*. Final decisions stay with humans.")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Scoring weights")
    weights = {k: st.slider(k.title(), 0, 100, int(v * 100), 5) for k, v in DEFAULT_WEIGHTS.items()}
    st.caption("Weights are normalised automatically.")
    backend = st.radio("Semantic engine", ["auto", "sbert", "tfidf"], index=0,
                       help="auto = sentence-transformers if installed, else TF-IDF")
    anonymize_on = st.toggle("Anonymise resumes before semantic scoring", value=True,
                             help="Removes name, contact details and gendered words to reduce bias.")
    use_sample = st.button("Load sample data")

# -------------------------------------------------------------------- input
col_jd, col_cv = st.columns(2)
sample_jd = (ROOT / "data/sample_jd.txt").read_text() if use_sample else ""
with col_jd:
    st.subheader("1. Job description")
    jd_file = st.file_uploader("Upload JD (txt/pdf/docx)", type=["txt", "pdf", "docx"], key="jd")
    jd_text = extract_text(jd_file) if jd_file else None
    jd_text = st.text_area("…or paste it here", value=jd_text or sample_jd, height=260)
with col_cv:
    st.subheader("2. Resumes")
    files = st.file_uploader("Upload resumes", type=["pdf", "docx", "txt"], accept_multiple_files=True)

resumes = {f.name: extract_text(f) for f in files} if files else {}
if use_sample and not resumes:
    resumes = {p.name: p.read_text() for p in sorted((ROOT / "data/sample_resumes").glob("*.txt"))}
    st.info(f"Loaded {len(resumes)} sample resumes.")

if not (jd_text and resumes):
    st.stop()

# ---------------------------------------------------------------- JD summary
jd = parse_jd(jd_text)
with st.expander("What the system understood from the JD", expanded=True):
    c1, c2, c3 = st.columns(3)
    c1.markdown("**Required skills**\n\n" + (", ".join(jd["required_skills"]) or "—"))
    c2.markdown("**Preferred skills**\n\n" + (", ".join(jd["preferred_skills"]) or "—"))
    c3.markdown(f"**Experience:** {jd['min_years']:g}+ yrs  \n**Education:** {EDU_LABELS[jd['min_education']]}")

# -------------------------------------------------------------------- score
with st.spinner("Screening…"):
    results = screen(jd_text, resumes, weights=weights, anonymize_text=anonymize_on, embedder=backend)
df = to_dataframe(results)
st.caption(f"Semantic engine used: {results[0]['embedder']}")

st.subheader("Ranked candidates")
st.dataframe(df, use_container_width=True, hide_index=True,
             column_config={"Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.1f")})
st.bar_chart(df.set_index("Candidate")[["Semantic", "Skills", "Experience", "Education"]])
st.download_button("⬇️ Download ranking (CSV)", df.to_csv(index=False), "ranking.csv", "text/csv")

st.subheader("Why each candidate ranked where they did")
for r in results:
    a = r["analysis"]
    with st.expander(f"#{r['rank']}  {r['candidate']} — {r['score']}"):
        st.write(r["why"])
        x, y = st.columns(2)
        x.success("Matched: " + (", ".join(r["matched_required"] + r["matched_preferred"]) or "none"))
        y.error("Missing required: " + (", ".join(r["missing_required"]) or "none"))
        st.progress(min(r["score"] / 100, 1.0))

        st.markdown("##### 🔎 What the system read in this resume")
        c1, c2, c3 = st.columns(3)
        c1.markdown("**Education**  \n" + (f"{r['education']}  \n_“{a['education_evidence']}”_" if a["education"] else "⚠️ not detected"))
        if a["years_source"] == "dates":
            exp = f"{a['years']:g} yrs from: " + "; ".join(a["date_ranges"][:4])
        elif a["years_source"] == "stated":
            exp = f"{a['years']:g} yrs (as stated in text)"
        else:
            exp = "⚠️ no work dates found (0 yrs)"
        c2.markdown("**Experience**  \n" + exp)
        c3.markdown(f"**Skills found ({len(a['skills'])})**  \n" + (", ".join(a["skills"]) or "⚠️ none"))
        if a["other_skills"]:
            st.caption("Other skills listed (not in the taxonomy, so not scored): " + ", ".join(a["other_skills"]))
        if not a["education"] or a["years_source"] == "none" or not a["skills"]:
            st.warning("Something was not detected. Check the extracted text below - scanned or heavily "
                       "formatted PDFs can lose text. A plain-text/DOCX export usually works better.")
        with st.expander("View extracted resume text"):
            st.text(resumes[r["candidate"]][:4000])
