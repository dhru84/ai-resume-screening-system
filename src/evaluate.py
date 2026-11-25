"""Evaluate ranking quality against human relevance labels.

    python -m src.evaluate

Ground truth: eval/ground_truth.csv with columns `resume,relevance` where
relevance is 0 (irrelevant) .. 3 (excellent). Replace the sample data with your
own ~20 hand-ranked resumes for a stronger result.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score

from .parser import extract_text
from .scorer import screen, tfidf_baseline

ROOT = Path(__file__).resolve().parent.parent


def load(jd_path=ROOT / "data/sample_jd.txt", resume_dir=ROOT / "data/sample_resumes",
         truth=ROOT / "eval/ground_truth.csv"):
    jd = extract_text(jd_path)
    truth_df = pd.read_csv(truth)
    resumes = {r: extract_text(Path(resume_dir) / r) for r in truth_df["resume"]}
    return jd, resumes, truth_df.set_index("resume")["relevance"].to_dict()


def metrics(scores: dict[str, float], truth: dict[str, int], k: int = 3) -> dict:
    names = list(truth)
    y_true = np.array([truth[n] for n in names], dtype=float)
    y_score = np.array([scores[n] for n in names], dtype=float)
    top = sorted(names, key=lambda n: scores[n], reverse=True)[:k]
    return {
        f"P@{k}": sum(truth[n] >= 2 for n in top) / k,
        f"NDCG@{k}": ndcg_score([y_true], [y_score], k=k),
        "NDCG@all": ndcg_score([y_true], [y_score]),
        "Spearman": spearmanr(y_score, y_true)[0],
    }


def run(embedder: str = "auto") -> pd.DataFrame:
    jd, resumes, truth = load()
    rows = {"TF-IDF baseline": metrics(tfidf_baseline(jd, resumes), truth)}
    configs = {
        "Semantic only": {"semantic": 1, "skills": 0, "experience": 0, "education": 0},
        "Skills only": {"semantic": 0, "skills": 1, "experience": 0, "education": 0},
        "Hybrid (default weights)": None,
    }
    embedder_name = ""
    for label, w in configs.items():
        res = screen(jd, resumes, weights=w, embedder=embedder)
        embedder_name = res[0]["embedder"]
        rows[label] = metrics({r["candidate"]: r["score"] for r in res}, truth)
    df = pd.DataFrame(rows).T.round(3)
    print(f"Semantic backend: {embedder_name}\n")
    print(df.to_markdown() if hasattr(df, "to_markdown") else df)
    return df


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "auto")
