"""Hybrid resume scoring: semantic + skills + experience + education."""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .extractor import EDU_LABELS, parse_jd, parse_resume
from .privacy import anonymize

DEFAULT_WEIGHTS = {"semantic": 0.35, "skills": 0.35, "experience": 0.15, "education": 0.15}
REQUIRED_SKILL_WEIGHT = 2.0   # a required skill counts double a preferred one
PREFERRED_SKILL_WEIGHT = 1.0


# ------------------------------------------------------------- embedders ----

def _chunks(text: str, size: int = 120) -> list[str]:
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)] or [""]


class TfidfEmbedder:
    """Lexical fallback used when sentence-transformers is unavailable."""

    name = "TF-IDF cosine (fallback)"
    # Typical raw cosine range for a good resume<->JD pair; used to rescale to 0-1.
    scale = 0.45

    def similarities(self, jd: str, docs: list[str]) -> np.ndarray:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        matrix = vec.fit_transform([jd] + docs)
        sims = cosine_similarity(matrix[0], matrix[1:])[0]
        return np.clip(sims / self.scale, 0, 1)


class SBertEmbedder:
    """Semantic embedder with chunking.

    MiniLM truncates long inputs, so both JD and resume are split into ~120-word
    chunks. For every JD chunk we take its best-matching resume chunk and
    average these maxima: "how well does the resume cover each part of the JD".
    """

    name = "sentence-transformers/all-MiniLM-L6-v2"
    low, high = 0.15, 0.65  # raw cosine range rescaled to 0-1

    def __init__(self):
        from sentence_transformers import SentenceTransformer

        self.model = _load_sbert()

    def similarities(self, jd: str, docs: list[str]) -> np.ndarray:
        jd_chunks = _chunks(jd)
        jd_emb = self.model.encode(jd_chunks, normalize_embeddings=True)
        out = []
        for doc in docs:
            doc_emb = self.model.encode(_chunks(doc), normalize_embeddings=True)
            sim = jd_emb @ doc_emb.T
            out.append(float(sim.max(axis=1).mean()))
        out = np.array(out)
        return np.clip((out - self.low) / (self.high - self.low), 0, 1)


@lru_cache(maxsize=1)
def _load_sbert():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def get_embedder(prefer: str = "auto"):
    """prefer: 'auto' (try SBERT, else TF-IDF), 'sbert', or 'tfidf'."""
    if prefer != "tfidf":
        try:
            return SBertEmbedder()
        except Exception:
            if prefer == "sbert":
                raise
    return TfidfEmbedder()


# ---------------------------------------------------------- component fns ----

def skill_score(required: list[str], preferred: list[str], have: set[str]):
    req_hit = [s for s in required if s in have]
    pref_hit = [s for s in preferred if s in have]
    total = REQUIRED_SKILL_WEIGHT * len(required) + PREFERRED_SKILL_WEIGHT * len(preferred)
    if total == 0:
        return 1.0, req_hit, pref_hit
    got = REQUIRED_SKILL_WEIGHT * len(req_hit) + PREFERRED_SKILL_WEIGHT * len(pref_hit)
    return got / total, req_hit, pref_hit


def experience_score(have: float, need: float) -> float:
    return 1.0 if need <= 0 else min(have / need, 1.0)


def education_score(have: int, need: int) -> float:
    return 1.0 if need <= 0 else min(have / need, 1.0)


def _normalise(weights: dict | None) -> dict:
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    total = sum(w.values()) or 1.0
    return {k: v / total for k, v in w.items()}


def explain(name, jd, cand, req_hit, pref_hit) -> str:
    parts = []
    if jd["required_skills"]:
        miss = [s for s in jd["required_skills"] if s not in req_hit]
        txt = f"Matches {len(req_hit)}/{len(jd['required_skills'])} required skills"
        parts.append(txt + (f" (missing: {', '.join(miss)})" if miss else ""))
    if jd["preferred_skills"]:
        parts.append(f"{len(pref_hit)}/{len(jd['preferred_skills'])} preferred skills")
    if jd["min_years"]:
        parts.append(f"{cand['years']:g} yrs experience vs {jd['min_years']:g} required")
    if jd["min_education"]:
        parts.append(f"{EDU_LABELS[cand['education']]} vs {EDU_LABELS[jd['min_education']]} required")
    return "; ".join(parts) + "."


# ------------------------------------------------------------------ main ----

def screen(jd_text: str, resumes: dict[str, str], weights: dict | None = None,
           anonymize_text: bool = True, embedder: str = "auto") -> list[dict]:
    """Score and rank resumes against a job description.

    Returns a list of dicts sorted by score (best first).
    """
    w = _normalise(weights)
    jd = parse_jd(jd_text)
    emb = get_embedder(embedder)
    names = list(resumes)
    texts = [anonymize(resumes[n]) if anonymize_text else resumes[n] for n in names]
    sem = emb.similarities(jd_text, texts)

    results = []
    for name, raw, s_sem in zip(names, [resumes[n] for n in names], sem):
        cand = parse_resume(raw)
        s_skill, req_hit, pref_hit = skill_score(jd["required_skills"], jd["preferred_skills"], set(cand["skills"]))
        s_exp = experience_score(cand["years"], jd["min_years"])
        s_edu = education_score(cand["education"], jd["min_education"])
        parts = {"semantic": float(s_sem), "skills": s_skill, "experience": s_exp, "education": s_edu}
        score = 100 * sum(w[k] * v for k, v in parts.items())
        results.append({
            "candidate": name,
            "score": round(score, 1),
            "breakdown": {k: round(100 * v, 1) for k, v in parts.items()},
            "matched_required": req_hit,
            "matched_preferred": pref_hit,
            "missing_required": [s for s in jd["required_skills"] if s not in req_hit],
            "missing_preferred": [s for s in jd["preferred_skills"] if s not in pref_hit],
            "extra_skills": [s for s in cand["skills"] if s not in jd["required_skills"] + jd["preferred_skills"]],
            "years": cand["years"],
            "education": EDU_LABELS[cand["education"]],
            "why": explain(name, jd, cand, req_hit, pref_hit),
            "embedder": emb.name,
            "analysis": cand,
        })
    results.sort(key=lambda r: r["score"], reverse=True)
    for i, r in enumerate(results, 1):
        r["rank"] = i
    return results


def to_dataframe(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "Rank": r["rank"], "Candidate": r["candidate"], "Score": r["score"],
            "Semantic": r["breakdown"]["semantic"], "Skills": r["breakdown"]["skills"],
            "Experience": r["breakdown"]["experience"], "Education": r["breakdown"]["education"],
            "Years": r["years"], "Degree": r["education"],
            "Missing required": ", ".join(r["missing_required"]),
        })
    return pd.DataFrame(rows)


def tfidf_baseline(jd_text: str, resumes: dict[str, str]) -> dict[str, float]:
    """Plain TF-IDF cosine similarity: the baseline the hybrid model must beat."""
    names = list(resumes)
    vec = TfidfVectorizer(stop_words="english")
    m = vec.fit_transform([jd_text] + [resumes[n] for n in names])
    sims = cosine_similarity(m[0], m[1:])[0]
    return dict(zip(names, sims))
