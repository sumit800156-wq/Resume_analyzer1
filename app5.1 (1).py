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
import base64
# ---------------- NLTK ---------------- #
nltk.download("punkt")
nltk.download("stopwords")

# ---------------- CONFIG ---------------- #
st.set_page_config(page_title="AI Resume Analyzer Pro", layout="wide")
# ---------------- CSS ---------------- #
st.markdown("""
<style>

/* Streamlit default padding hatao */
.block-container {
    padding-top: 0rem !important;
    padding-bottom: 2rem !important;
}
/* Streamlit ka main container relative banao - YE SABSE ZARURI HAI */
div[data-testid="stAppViewContainer"] > div:first-child {
    position: relative;
}
/* Hero container ko relative banao */
.hero-wrapper {
    position: relative;
}

/* Cards ko image ke upar absolute kar do */
.feature-section{
    position: absolute !important;
    bottom: -100px !important;  /* image ke bottom se 60px neeche */
    left: 50px !important;
    right: 50px !important;
    z-index: 999 !important;
}

/* Cards glass effect */
.feature-box {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    color: black;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.4);
    height: 120px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    font-weight: 600;
    border: 1px solid rgba(255,255,255,0.3);
}

/* Green/Red boxes */
.green-box {
    background: #d4edda;
    color: #155724;
    padding: 15px;
    border-radius: 10px;
    border-left: 5px solid #28a745;
    margin-bottom: 10px;
}
.red-box {
    background: #f8d7da;
    color: #721c24;
    padding: 15px;
    border-radius: 10px;
    border-left: 5px solid #dc3545;
    margin-bottom: 10px;
}

/* Button */
.stButton>button {
    background-color: #4da6ff;
    color: white;
    border-radius: 8px;
    border: none;
    font-weight: 600;
}
.start-btn{
    text-align:center;
    margin-top:80px;  /* cards ke neeche button ke liye space */
}

</style>

 """, unsafe_allow_html=True)
# ---------------- SESSION ---------------- #
if "page" not in st.session_state:
    st.session_state.page = "home"

def go_to(page):
    st.session_state.page = page
def hero_banner(image_path, title):
    try:
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode()
        img_tag = f'<img src="data:image/png;base64,{encoded}" style="width:100%;height:100%;object-fit:cover;">'
    except:
        img_tag = '<div style="width:100%;height:100%;background:linear-gradient(90deg,#4da6ff,#0052cc);"></div>'

    st.markdown('<div class="hero-wrapper">', unsafe_allow_html=True)

    st.markdown(f"""
    <div style="position:relative;width:100%;height:400px;overflow:hidden;border-radius:10px;z-index:1;">
        {img_tag}
        <div style="
            position:absolute;
            top:50%;
            left:50%;
            transform:translate(-50%,-50%);
            color:white;
            font-size:48px;
            font-weight:bold;
            text-shadow:3px 3px 10px rgba(0,0,0,0.8);
            width:100%;
            text-align:center;
            z-index:2;">
            {title}
        </div>
    </div>
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

    hero_banner("image.png", "🚀 AI Resume Analyzer Pro")

    st.markdown("<div class='feature-section'>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<div class='feature-box'>📊 ATS Score</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='feature-box'>🧠 Skill Match</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='feature-box'>❌ Missing Keywords</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,1,1])

    with col2:
        if st.button("🚀 Start Analysis"):
            go_to("upload")
# ---------------- UPLOAD ---------------- #
elif st.session_state.page == "upload":
    hero_banner("image1.png", "📤 Upload Inputs")
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
    hero_banner("image2.png", "📊 Analysis Results")
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