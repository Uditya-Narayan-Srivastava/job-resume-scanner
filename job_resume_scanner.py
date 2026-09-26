import io
import re
from typing import List, Tuple, Set, Dict

import streamlit as st
import spacy
from PyPDF2 import PdfReader
from docx import Document

# Load NLP model
# -----------------------------
nlp = spacy.load("en_core_web_sm")

# Skill Bank
# -----------------------------
SKILL_BANK = [
    # Data/ML/AI
    "machine learning", "deep learning", "natural language processing", "computer vision",
    "data science", "data analysis", "data engineering", "llm", "nlp", "cv",
    "neural networks", "recommendation systems", "time series",
    # Languages
    "python", "java", "c++", "c", "c#", "javascript", "typescript", "sql", "r", "go", "rust", "scala",
    # Python stack
    "numpy", "pandas", "scikit-learn", "tensorflow", "keras", "pytorch", "matplotlib",
    "flask", "django", "fastapi", "streamlit",
    # Cloud/DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "git", "linux", "bash",
    # Web/DB
    "html", "css", "react", "node", "express", "rest api", "graphql", "mongodb", "mysql", "postgresql",
    # BI/ETL/Tools
    "power bi", "tableau", "airflow", "spark", "hadoop", "excel",
]

ALIASES = {
    "ml": "machine learning",
    "dl": "deep learning",
    "cv": "computer vision",
    "llms": "llm",
    "nlp": "natural language processing",
    "tf": "tensorflow",
    "sklearn": "scikit-learn",
    "ts": "typescript",
    "js": "javascript",
    "postgres": "postgresql",
    "rest": "rest api",
}

# Utilities
# -----------------------------
def _to_text_pdf(file) -> str:
    text = []
    reader = PdfReader(file)
    for page in reader.pages:
        try:
            text.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(text)

def _to_text_docx(file) -> str:
    data = file.read()
    file.seek(0)
    bio = io.BytesIO(data)
    doc = Document(bio)
    return "\n".join(p.text for p in doc.paragraphs)

def extract_text(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        return _to_text_pdf(uploaded_file)
    elif name.endswith(".docx"):
        return _to_text_docx(uploaded_file)
    else:
        try:
            return _to_text_pdf(uploaded_file)
        except Exception:
            try:
                uploaded_file.seek(0)
                return _to_text_docx(uploaded_file)
            except Exception:
                return ""

def normalize_token(tok: str) -> str:
    tok = tok.strip().lower()
    tok = ALIASES.get(tok, tok)
    return tok

def clean_text(text: str) -> str:
    text = re.sub(r"[^\w\+\-#/\. ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()

def ngrams(words: List[str], n: int) -> List[str]:
    return [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]

def extract_skills(text: str) -> Set[str]:
    txt = clean_text(text)
    words = txt.split()
    found = set()
    bank = set(normalize_token(s) for s in SKILL_BANK)
    for n in (3, 2, 1):
        for g in ngrams(words, n):
            g_norm = normalize_token(g)
            if g_norm in bank:
                found.add(g_norm)
    for w in words:
        w_norm = normalize_token(w)
        if w_norm in bank:
            found.add(w_norm)
    return found

def extract_experience_years(text: str) -> float:
    txt = clean_text(text)
    vals = []

    ranges = re.findall(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*(?:\+?\s*)?(?:years|yrs|y)", txt)
    vals += [float(a) for a, _ in ranges]

    singles = re.findall(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|year|yrs|yr|y)\b", txt)
    vals += [float(x) for x in singles]

    totals = re.findall(r"total\s+experience\s*[:\-]?\s*(\d+(?:\.\d+)?)", txt)
    vals += [float(x) for x in totals]

    return max(vals) if vals else 0.0

def detect_fresher(text: str) -> bool:
    txt = text.lower()
    keywords = ["fresher", "recent graduate", "entry level", "no experience", "internship"]
    return any(k in txt for k in keywords)

def parse_jd_requirements(jd_text: str) -> Tuple[Set[str], float]:
    jd_skills = extract_skills(jd_text)
    jd_exp = extract_experience_years(jd_text)
    return jd_skills, jd_exp

def score_resume(cand_skills: Set[str], cand_exp: float, jd_skills: Set[str], jd_min_exp: float, is_fresher: bool) -> Dict[str, float]:
    if jd_skills:
        matched = cand_skills.intersection(jd_skills)
        skill_score = (len(matched) / len(jd_skills)) * 100.0
    else:
        matched = set()
        skill_score = 50.0

    if is_fresher:
        exp_score = 100.0 if jd_min_exp <= 1 else 40.0  # Freshers penalized only if JD strictly needs >1 year
    else:
        if jd_min_exp > 0:
            exp_ratio = min(cand_exp / jd_min_exp, 1.0)
            exp_score = exp_ratio * 100.0
        else:
            exp_score = 50.0

    overall = 0.7 * skill_score + 0.3 * exp_score
    missing_skills = jd_skills - cand_skills

    return {
        "overall": round(overall, 2),
        "skill_score": round(skill_score, 2),
        "exp_score": round(exp_score, 2),
        "matched_count": len(matched),
        "required_count": len(jd_skills),
        "missing_count": len(missing_skills),
    }

# Streamlit App
# -----------------------------
st.set_page_config(page_title="AI Resume Scanner (Skills + Experience + Fresher)", page_icon="🤖", layout="wide")
st.title("🤖 AI Resume Scanner — Skills + Experience + Fresher Support")
st.caption("Upload resumes (PDF/DOCX) + paste Job Description. Handles both **Freshers & Experienced** automatically.")

with st.expander("ℹ️ Scoring System"):
    st.markdown(
        "- **Skills (70%)**: JD skills matched.  \n"
        "- **Experience (30%)**: Freshers get fair scoring. If JD asks ≤1 year → full marks, else partial credit.  \n"
        "- Detects **Fresher vs Experienced** automatically.  \n"
        "- Shows **Matched vs Missing** skills and experience extracted."
    )

uploaded = st.file_uploader("📂 Upload Resumes", type=["pdf", "docx"], accept_multiple_files=True)
jd_text = st.text_area("📝 Paste Job Description", height=200,
                       placeholder="e.g., Looking for Python + ML fresher OR 2+ years experience in data science...")

if st.button("Analyze"):
    if not jd_text:
        st.error("Please paste a Job Description.")
    elif not uploaded:
        st.error("Please upload at least one resume.")
    else:
        jd_skills, jd_exp = parse_jd_requirements(jd_text)
        st.subheader("📑 JD Parsed Requirements")
        st.metric("Required Skills", len(jd_skills))
        st.metric("Min Experience (yrs)", jd_exp)
        st.write("**Skills Detected:** " + (", ".join(sorted(jd_skills)) if jd_skills else "—"))

        results, details = [], []

        for file in uploaded:
            try:
                text = extract_text(file)
                if not text.strip():
                    raise ValueError("No text extracted (scanned PDF not supported).")

                cand_skills = extract_skills(text)
                cand_exp = extract_experience_years(text)
                fresher_flag = detect_fresher(text) or (cand_exp == 0)

                scores = score_resume(cand_skills, cand_exp, jd_skills, jd_exp, fresher_flag)
                results.append({
                    "Resume": file.name,
                    "Overall Score": scores["overall"],
                    "Skill Score": scores["skill_score"],
                    "Exp Score": scores["exp_score"],
                    "Candidate Exp (yrs)": "Fresher" if fresher_flag else round(cand_exp, 2),
                    "JD Min Exp (yrs)": jd_exp,
                    "Matched Skills": scores["matched_count"],
                    "Required Skills": scores["required_count"],
                    "Missing Skills": scores["missing_count"],
                })
                details.append((file.name, cand_skills, jd_skills - cand_skills, fresher_flag))

            except Exception as e:
                st.warning(f"⚠️ {file.name}: {e}")

        results_sorted = sorted(results, key=lambda x: x["Overall Score"], reverse=True)

        st.subheader("🏆 Ranking")
        st.dataframe(results_sorted, use_container_width=True)

        st.subheader("🔍 Resume Details")
        for name, sk_found, sk_missing, fresher_flag in details:
            with st.expander(f"{name} — skill breakdown"):
                st.markdown(f"**Candidate Type:** {'Fresher' if fresher_flag else 'Experienced'}")
                st.markdown(f"**Detected Skills:** {', '.join(sorted(sk_found)) if sk_found else '—'}")
                st.markdown(f"**Missing JD Skills:** {', '.join(sorted(sk_missing)) if sk_missing else '—'}")

        st.success("Analysis complete ✅")

st.write("---")
st.caption("Tip: For scanned PDFs (image-only), run OCR before uploading.")
