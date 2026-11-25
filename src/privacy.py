"""Lightweight anonymisation to reduce bias before semantic scoring.

This is a heuristic, not a guarantee. It removes direct identifiers (name line,
contact details, links) and gendered / demographic cues so they cannot
influence the semantic similarity score. Skills, experience and education
scores never use these fields at all.
"""
import re

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(?<!\d)\+?\d[\d\s().-]{8,}\d(?!\d)")
_URL = re.compile(r"(https?://\S+|www\.\S+|linkedin\.com/\S+|github\.com/\S+)", re.I)
_PERSONAL_LINE = re.compile(
    r"^\s*(name|gender|sex|age|dob|date of birth|marital status|nationality|"
    r"religion|father'?s name|address)\s*[:\-].*$",
    re.I | re.M,
)
_GENDER_WORDS = re.compile(
    r"\b(he|she|his|her|hers|him|himself|herself|mr|mrs|ms|miss|male|female)\b\.?", re.I
)


def anonymize(text: str) -> str:
    lines = text.strip().splitlines()
    # First non-empty line is almost always the candidate's name.
    for i, line in enumerate(lines):
        if line.strip():
            words = line.split()
            if len(words) <= 4 and re.fullmatch(r"[A-Za-z.\s'-]+", line.strip()):
                lines[i] = ""
            break
    text = "\n".join(lines)
    text = _PERSONAL_LINE.sub("", text)
    for pattern in (_EMAIL, _PHONE, _URL):
        text = pattern.sub(" ", text)
    text = _GENDER_WORDS.sub(" ", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()
