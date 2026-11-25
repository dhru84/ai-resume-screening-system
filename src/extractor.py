"""Information extraction: skills, experience, education, and JD parsing."""
from __future__ import annotations

import datetime as dt
import re
from functools import lru_cache

from .skills import SKILLS

# ---------------------------------------------------------------- skills ----


@lru_cache(maxsize=1)
def _skill_patterns() -> dict[str, re.Pattern]:
    patterns = {}
    for canon, aliases in SKILLS.items():
        alts = sorted({a.lower() for a in aliases} | {canon.lower()}, key=len, reverse=True)
        body = "|".join(re.escape(a) for a in alts)
        patterns[canon] = re.compile(
            rf"(?<![A-Za-z0-9+#.])(?:{body})(?![A-Za-z0-9+#])", re.I
        )
    return patterns


def extract_skills(text: str) -> set[str]:
    """Return canonical skills found in text."""
    return {canon for canon, pat in _skill_patterns().items() if pat.search(text)}


# -------------------------------------------------------------- sections ----

# A line is a section heading only if EVERY word belongs to this vocabulary.
# This stops job titles such as "Education Consultant" from being read as headings.
_HEADING_WORDS = set("""
education educational academic academics qualification qualifications background details scholastic
experience experiences employment work history professional internship internships training industrial
projects project personal key technical core skills skill technologies tech stack competencies strengths
summary objective profile about me career and of the relevant additional other
certifications certification courses course achievements achievement awards extracurricular activities
hobbies interests references declaration publications
""".split())

_HEADINGS = [  # order matters: first match wins
    ("education", r"educat|academic|qualification|scholastic"),
    ("projects", r"projects?"),
    ("experience", r"experience|employment|work history|work profile|internship|training"),
    ("skills", r"skills?|technologies|tech stack|competenc|strengths"),
    ("summary", r"summary|profile|objective|about me|career"),
    ("certifications", r"certification|courses?|achievement|awards?|extracurricular|activities|hobbies|interests|references|declaration|publications?"),
]
_BULLETS = "-•*▪●◦·➢►–—"


def _heading_of(line: str) -> str | None:
    raw = line.strip()
    if not raw or raw[0] in _BULLETS:
        return None
    s = raw.strip(":#*_=| ").lower().replace("&", " and ")
    words = re.findall(r"[a-z]+", s)
    if not words or len(words) > 4 or re.search(r"[\d@,;]", s):
        return None
    if not all(w in _HEADING_WORDS for w in words):
        return None
    for name, pat in _HEADINGS:
        if re.search(pat, s):
            return name
    return None


def split_sections(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current = "header"
    for line in text.splitlines():
        heading = _heading_of(line)
        if heading:
            current = heading
            continue
        sections.setdefault(current, []).append(line)
    return {k: "\n".join(v) for k, v in sections.items()}


# ------------------------------------------------------------ experience ----

_MON = "jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec"
_MONTHS = {m: i for i, m in enumerate(_MON.split("|"), 1)}
# One date: "Jun 2023", "June'23", "06/2023", "2023".
_POINT = (r"(?:(?P<m{n}>" + _MON + r")[a-z]{{0,6}}\.?[\s,'’\-]*|(?P<nm{n}>\d{{1,2}})\s?[/.\-]\s?)?"
          r"(?P<y{n}>(?:19|20)\d{{2}}|(?<=['’])\d{{2}})")
_RANGE = re.compile(
    _POINT.format(n=1) + r"\s*(?:-|–|—|to|till|until)\s*(?:" + _POINT.format(n=2)
    + r"|(?P<pres>present|current|now|ongoing|date|today))",
    re.I,
)
_STATED = re.compile(
    r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)(?:\s+of)?\s+(?:[\w-]+\s+){0,3}experience", re.I
)
_EDU_WORDS = re.compile(
    r"b\.?\s?tech|bachelor|master|m\.?\s?tech|b\.?\s?sc|m\.?\s?sc|mba|mca|bca|ph\.?d|university|"
    r"college|school|institute|degree|diploma|cgpa|gpa|\bb\.?e\b",
    re.I,
)


def _idx(m: re.Match, n: int, is_end: bool) -> int:
    y = m.group(f"y{n}")
    year = int(y) + 2000 if len(y) == 2 else int(y)
    mon, num = m.group(f"m{n}"), m.group(f"nm{n}")
    if mon:
        month = _MONTHS[mon.lower()[:3]]
    elif num and 1 <= int(num) <= 12:
        month = int(num)
    else:
        month = 12 if is_end else 1
    return year * 12 + month


def experience_details(text: str, today: dt.date | None = None) -> dict:
    """Estimate professional experience and show the evidence used.

    Returns {"years", "source" ('dates' | 'stated' | 'none'), "ranges": [matched date ranges]}.
    Dates are read from the Experience/Internship sections (education and project
    dates are ignored) and merged so overlapping jobs are not double counted.
    """
    today = today or dt.date.today()
    now_idx = today.year * 12 + today.month
    sections = split_sections(text)
    if "experience" in sections:
        scope = sections["experience"]
    else:
        keep = [v for k, v in sections.items() if k not in {"education", "projects", "certifications", "skills"}]
        scope = "\n".join(l for l in "\n".join(keep).splitlines() if not _EDU_WORDS.search(l))

    intervals, ranges = [], []
    for m in _RANGE.finditer(scope):
        start = _idx(m, 1, False)
        end = now_idx if m.group("pres") else min(_idx(m, 2, True), now_idx)
        if end >= start:
            intervals.append((start, end))
            ranges.append(m.group(0).strip())

    intervals.sort()
    merged: list[list[int]] = []
    for s, e in intervals:
        if merged and s <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    months = sum(e - s + 1 for s, e in merged)
    if months > 0:
        return {"years": round(months / 12, 1), "source": "dates", "ranges": ranges}
    stated = [float(x) for x in _STATED.findall(text)]
    if stated:
        return {"years": max(stated), "source": "stated", "ranges": []}
    return {"years": 0.0, "source": "none", "ranges": []}


def experience_years(text: str, today: dt.date | None = None) -> float:
    return experience_details(text, today)["years"]


# ------------------------------------------------------------- education ----

EDU_LABELS = {0: "Not specified", 1: "Diploma", 2: "Bachelor's", 3: "Master's", 4: "PhD"}
# Case-insensitive patterns
_EDU_CI = [
    (4, r"\bph\.?\s?d\b|doctorate|doctoral"),
    (3, r"(?<!scrum )\bmaster'?s?\b|\bm\.?\s?tech\b|\bm\.?\s?sc\b|\bmba\b|\bmca\b|\bpgdm\b|\bm\.?\s?com\b|post[\s-]?graduat"),
    (2, r"\bbachelor'?s?\b|\bb\.?\s?tech\b|\bb\.?\s?sc\b|\bbca\b|\bbba\b|\bb\.?\s?com\b|\bundergraduate\b"),
    (1, r"\bdiploma\b|\bpolytechnic\b"),
]
# Case-SENSITIVE patterns: short abbreviations that are ordinary words in lowercase (be, me, ma, ms)
_EDU_CS = [
    (3, r"\bM\.E\b|\bME(?=\s*(?:in\b|\())|\bM\.A\b|\bM\.S\b"),
    (2, r"\bB\.?E\b|\bB\.A\b|\bB\.S\b"),
]


def _line_level(line: str) -> int:
    best = 0
    for lvl, pat in _EDU_CI:
        if re.search(pat, line, re.I):
            best = max(best, lvl)
    for lvl, pat in _EDU_CS:
        if re.search(pat, line):
            best = max(best, lvl)
    return best


def detect_education(text: str) -> int:
    return max((_line_level(l) for l in text.splitlines()), default=0)


def education_details(text: str) -> tuple[int, str]:
    """Highest degree level plus the line that proves it.

    Looks in the Education section first; if nothing is found there (or the heading
    was not recognised) it scans the whole resume.
    """
    sections = split_sections(text)
    scopes = ([sections["education"]] if "education" in sections else []) + [text]
    for scope in scopes:
        best, evidence = 0, ""
        for line in scope.splitlines():
            lvl = _line_level(line)
            if lvl > best:
                best, evidence = lvl, line.strip()
        if best:
            return best, evidence[:140]
    return 0, ""


def resume_education(text: str) -> int:
    return education_details(text)[0]


# ------------------------------------------------------------ JD parsing ----

_PREFERRED = re.compile(
    r"\b(preferred|nice to have|good to have|a plus|is a plus|bonus|desirable|added advantage|optional)\b", re.I
)
_REQUIRED_HEADING = re.compile(
    r"\b(requirements?|must have|must-have|required|qualifications?|responsibilities|what you.?ll need|skills)\b", re.I
)


def parse_jd(text: str) -> dict:
    """Split a job description into required / preferred skills plus thresholds."""
    required: set[str] = set()
    preferred: set[str] = set()
    edu_levels: list[int] = []
    mode = "required"

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        is_heading = len(stripped) < 60 and (stripped.endswith(":") or stripped.isupper())
        if is_heading and _PREFERRED.search(stripped):
            mode = "preferred"
            continue
        if is_heading and _REQUIRED_HEADING.search(stripped):
            mode = "required"
            continue
        line_mode = "preferred" if (mode == "preferred" or _PREFERRED.search(stripped)) else "required"
        skills = extract_skills(stripped)
        (preferred if line_mode == "preferred" else required).update(skills)
        if line_mode == "required":
            level = detect_education(stripped)
            if level:
                edu_levels.append(level)

    preferred -= required
    years = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)", text, re.I)]
    return {
        "required_skills": sorted(required),
        "preferred_skills": sorted(preferred),
        "min_years": years[0] if years else 0.0,
        "min_education": min(edu_levels) if edu_levels else 0,
    }


_GENERIC_LABELS = {"languages", "programming languages", "frameworks", "tools", "databases", "technologies",
                   "skills", "libraries", "platforms", "soft skills", "others", "other", "technical skills", "web"}


def listed_skills(text: str) -> list[str]:
    """Items from the Skills section that are NOT in the taxonomy (shown, not scored)."""
    section = split_sections(text).get("skills", "")
    out: list[str] = []
    for token in re.split(r"[:,;|•·●▪\n]+", section):
        t = token.strip(" -–—*\t")
        if (2 <= len(t) <= 30 and len(t.split()) <= 4 and re.search(r"[A-Za-z]", t)
                and t.lower() not in _GENERIC_LABELS and not extract_skills(t)
                and t.lower() not in {o.lower() for o in out}):
            out.append(t)
    return out[:25]


def parse_resume(text: str) -> dict:
    exp = experience_details(text)
    level, evidence = education_details(text)
    return {
        "skills": sorted(extract_skills(text)),
        "years": exp["years"],
        "years_source": exp["source"],
        "date_ranges": exp["ranges"],
        "education": level,
        "education_evidence": evidence,
        "other_skills": listed_skills(text),
    }
