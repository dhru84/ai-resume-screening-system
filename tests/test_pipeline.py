import datetime as dt

from src.extractor import detect_education, experience_years, extract_skills, parse_jd
from src.privacy import anonymize
from src.scorer import screen


def test_skill_word_boundaries():
    found = extract_skills("Worked with JavaScript, MySQL and C++ daily.")
    assert "javascript" in found and "mysql" in found and "c++" in found
    assert "java" not in found and "sql" not in found


def test_experience_merges_overlaps():
    text = "Experience\nA Corp Jan 2020 - Dec 2021\nB Corp Jun 2021 - Dec 2022\n"
    assert experience_years(text, today=dt.date(2026, 1, 1)) == 3.0


def test_education_not_counted_as_experience():
    text = "Education\nB.Tech 2016 - 2020\nExperience\nX Jan 2021 - Dec 2021\n"
    assert experience_years(text, today=dt.date(2026, 1, 1)) == 1.0


def test_education_levels():
    assert detect_education("Ph.D. in CS") == 4
    assert detect_education("M.Sc. Statistics") == 3
    assert detect_education("B.Tech CSE") == 2


def test_jd_required_vs_preferred():
    jd = parse_jd("Requirements:\n- Python and SQL\nNice to have:\n- Docker\n3+ years experience")
    assert jd["required_skills"] == ["python", "sql"]
    assert jd["preferred_skills"] == ["docker"]
    assert jd["min_years"] == 3.0


def test_anonymize_removes_identifiers():
    out = anonymize("Jane Doe\njane@x.com +91 98765 43210\nShe built Python apps.")
    assert "jane@x.com" not in out and "Jane Doe" not in out and "She" not in out


def test_ranking_prefers_better_match():
    jd = "Requirements:\n- Python, SQL, machine learning\n2+ years experience"
    good = "Experience\nML Eng Jan 2020 - Dec 2023\nPython SQL machine learning projects"
    bad = "Experience\nChef Jan 2020 - Dec 2023\nCooking and kitchen management"
    res = screen(jd, {"good": good, "bad": bad}, embedder="tfidf")
    assert res[0]["candidate"] == "good"


# ---- robustness: formats common in student / Indian resumes ----
import pytest
from src.extractor import education_details, experience_details, listed_skills, parse_resume


@pytest.mark.parametrize("line,level", [
    ("BE in Computer Science, Anna University", 2),
    ("B.E. (CSE) 2021-2025, CGPA 8.4", 2),
    ("Bachelor of Engineering - Information Technology", 2),
    ("BTech CSE, VIT", 2), ("B.Sc Computer Science", 2), ("BCA, Christ University", 2),
    ("MCA, Bangalore University", 3), ("M.Tech in AI", 3), ("MBA Finance", 3),
    ("Diploma in Computer Engineering", 1), ("PhD in Machine Learning", 4),
    ("Certified Scrum Master", 0), ("Proficient in MS Office and Excel", 0),
    ("Please contact me to be considered", 0),
])
def test_education_formats(line, level):
    assert education_details("Education\n" + line)[0] == level


def test_education_found_when_heading_missing_or_elsewhere():
    assert education_details("John\nI studied B.Tech at NIT\nSkills\nPython")[0] == 2
    # heading exists but degree is written outside it
    assert education_details("Education\nXYZ University, 2021\nProjects\nBE final-year project")[0] == 2


@pytest.mark.parametrize("line,years", [
    ("Intern  Jun'24 - Aug'24", 0.2),
    ("Intern  06/2022 - 08/2022", 0.2),
    ("Dev  Jan 2022 - Dec 2023", 2.0),
    ("Dev  2020 - 2022", 3.0),
    ("Dev  March 2021 – Present", None),
])
def test_experience_date_formats(line, years):
    got = experience_details("Experience\n" + line, today=dt.date(2026, 11, 1))
    if years is None:
        assert got["years"] > 5
    else:
        assert got["years"] == years


def test_project_dates_not_counted_without_experience_heading():
    text = "Education\nB.Tech 2021 - 2025\nProjects\nChat app  Jan 2024 - Apr 2024\nSkills\nPython"
    assert experience_details(text)["years"] == 0.0


def test_job_title_not_mistaken_for_heading():
    text = "Experience\nEducation Consultant  Jan 2020 - Dec 2020\nAnalyst  Jan 2021 - Dec 2021"
    assert experience_details(text)["years"] == 2.0


def test_full_student_resume():
    text = """DHRUTHI K
dhruthi@example.com | +91 90000 00000

EDUCATIONAL QUALIFICATION
B.E. Computer Science and Engineering, XYZ Institute of Technology   2021 - 2025   CGPA 8.6

TECHNICAL SKILLS
Languages: Python, Java, C++
Web: HTML, CSS, JavaScript
Tools: Git, VS Code, Postman, Figma Dev Mode

INTERNSHIPS
Web Developer Intern, ABC Soft   Jun'24 - Aug'24

PROJECTS
Resume Screener (Python, NLP)   Jan 2025 - Mar 2025
"""
    r = parse_resume(text)
    assert r["education"] == 2 and "B.E." in r["education_evidence"]
    assert r["years"] == 0.2 and r["years_source"] == "dates"
    assert {"python", "java", "c++", "html", "css", "javascript", "git", "postman"} <= set(r["skills"])
    assert "VS Code" in r["other_skills"]
