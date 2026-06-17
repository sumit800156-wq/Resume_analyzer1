import streamlit as st
import PyPDF2
import re
import nltk
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ---------------- NLTK ---------------- #
nltk.download("punkt")
nltk.download("stopwords")

# ---------------- CONFIG ---------------- #
st.set_page_config(page_title="AI Resume Analyzer Pro", layout="wide")

# ---------------- SESSION ---------------- #
if "page" not in st.session_state:
    st.session_state.page = "home"

def go_to(page):
    st.session_state.page = page

# ---------------- CSS ---------------- #
st.markdown("""
<style>

/* Title on Image */
.center-title {
    position: relative;
    text-align: center;
    color: white;
    top: -250px;
    font-size: 42px;
    font-weight: bold;
}

/* Feature Cards */
.feature-box {
    background: white;
    color: black;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
}

/* Buttons */
.stButton>button {
    background-color: #4da6ff;
    color: white;
    border-radius: 8px;
}

/* Section Title */
.section-title {
    text-align: center;
    font-size: 30px;
    font-weight: bold;
    margin-top: 20px;
}

/* Result Boxes */
.green-box {
    background: #d4edda;
    padding: 15px;
    border-radius: 10px;
}

.red-box {
    background: #f8d7da;
    padding: 15px;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- FUNCTIONS ---------------- #
def extract_text_from_pdf(file):
    pdf = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    words = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

def calculate_similarity(resume, jd):
    vec = TfidfVectorizer()
    tfidf = vec.fit_transform([clean_text(resume), clean_text(jd)])
    return round(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0] * 100, 2)

def get_skills():
    return ["python","sql","machine learning","deep learning","pandas","numpy","excel","aws","docker","git"]

def get_skill_match(resume, jd):
    skills = get_skills()
    resume = resume.lower()
    jd = jd.lower()
    matched = [s for s in skills if s in resume and s in jd]
    missing = [s for s in skills if s in jd and s not in resume]
    return matched, missing

def calculate_ats_score(similarity, matched, missing):
    total = len(matched) + len(missing)
    skill_score = (len(matched)/total*100) if total>0 else 0
    ats = similarity*0.6 + skill_score*0.4
    return round(ats,2), round(skill_score,2)

def extract_keywords(text):
    words = word_tokenize(clean_text(text))
    return Counter(words).most_common(10)

# ---------------- HOME ---------------- #
if st.session_state.page == "home":
    st.image("image.png", use_column_width=True)
    st.markdown("<div class='center-title'>🚀 AI Resume Analyzer Pro</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<div class='feature-box'>📊 ATS Score</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='feature-box'>🧠 Skill Match</div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='feature-box'>❌ Missing Keywords</div>", unsafe_allow_html=True)

    if st.button("🚀 Start Analysis"):
        go_to("upload")

# ---------------- UPLOAD ---------------- #
elif st.session_state.page == "upload":
    st.image("image1.png", use_column_width=True)
    st.markdown("<div class='section-title'>📤 Upload Inputs</div>", unsafe_allow_html=True)

    file = st.file_uploader("📄 Upload Resume PDF", type=["pdf"])
    jd = st.text_area("📝 Paste Job Description")

    if st.button("🔍 Analyze"):
        if file and jd:
            st.session_state.file = file
            st.session_state.jd = jd
            go_to("results")
        else:
            st.warning("Upload file & JD")

# ---------------- RESULTS ---------------- #
elif st.session_state.page == "results":
    st.image("image2.png", use_column_width=True)
    st.markdown("<div class='section-title'>📊 Analysis Results</div>", unsafe_allow_html=True)

    resume = extract_text_from_pdf(st.session_state.file)
    jd = st.session_state.jd

    similarity = calculate_similarity(resume, jd)
    matched, missing = get_skill_match(resume, jd)
    ats, skill = calculate_ats_score(similarity, matched, missing)

    st.subheader("🎯 ATS Score")
    st.progress(ats/100)
    st.success(f"{ats}%")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='green-box'><b>Matched Skills</b><br>"+", ".join(matched)+"</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='red-box'><b>Missing Skills</b><br>"+", ".join(missing)+"</div>", unsafe_allow_html=True)

    st.subheader("📊 Skill Chart")
    fig = px.pie(names=["Matched","Missing"], values=[len(matched),len(missing)])
    st.plotly_chart(fig)

    st.subheader("📈 Keywords")
    df = pd.DataFrame(extract_keywords(resume), columns=["Keyword","Count"])
    st.dataframe(df)

    st.subheader("☁ Word Cloud")
    wc = WordCloud(width=800, height=400, background_color="white").generate(resume)
    fig, ax = plt.subplots()
    ax.imshow(wc)
    ax.axis("off")
    st.pyplot(fig)

    if st.button("🔄 Analyze Again"):
        go_to("upload")