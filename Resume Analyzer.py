import streamlit as st
import numpy as np
import pandas as pd
import pdfplumber
import matplotlib.pyplot as plt
import re
import ollama
from sentence_transformers import SentenceTransformer, util
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="AI Resume Analyzer Pro", layout="wide", initial_sidebar_state="collapsed", page_icon="📄")

# -------------------------
# PURPLE DREAM THEME CSS
# -------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400;14..32,500;14..32,600;14..32,700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%);
    }
    
    .main .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #6b21a5 !important;
        font-weight: 700 !important;
        margin-top: 0 !important;
        margin-bottom: 0.2rem !important;
    }
    
    .stFileUploader > div {
        background: #faf5ffcc !important;
        border: 2px dashed #c084fc !important;
        border-radius: 40px !important;
        margin-top: 0 !important;
    }
    
    .stMarkdown, .stAlert, .stExpander, .stTextArea, .stSelectbox, .stButton > button {
        background: rgba(255, 255, 255, 0.9) !important;
        backdrop-filter: blur(4px);
        border-radius: 28px !important;
        border: 1px solid #d8b4fe;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.05);
        padding: 0.5rem 1rem;
    }
    
    .stMetric {
        background: #ffffffdd;
        backdrop-filter: blur(8px);
        border-radius: 32px;
        padding: 1rem;
        border-top: 4px solid #8b5cf6;
        border-left: 1px solid #e9d5ff;
        border-right: 1px solid #e9d5ff;
        border-bottom: 1px solid #e9d5ff;
        transition: all 0.2s;
    }
    .stMetric:hover {
        transform: translateY(-3px);
        border-top-color: #7c3aed;
    }
    .stMetric label {
        color: #5b21b6 !important;
        font-weight: 600;
        font-size: 0.8rem;
        text-transform: uppercase;
    }
    .stMetric .stMetricValue {
        color: #8b5cf6 !important;
        font-size: 2rem !important;
        font-weight: 800;
    }
    
    .stButton > button {
        background: linear-gradient(95deg, #8b5cf6, #a78bfa) !important;
        color: white !important;
        font-weight: 700 !important;
        border: none !important;
        padding: 0.5rem 1.5rem !important;
        border-radius: 60px !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background: linear-gradient(95deg, #7c3aed, #8b5cf6) !important;
        transform: scale(0.98);
    }
    
    .streamlit-expanderHeader {
        background: #faf5ff !important;
        border-radius: 40px !important;
        color: #6b21a5 !important;
        font-weight: 700;
        border: 1px solid #d8b4fe;
    }
    
    .skill-tag {
        display: inline-block;
        background: #ede9fe;
        color: #5b21b6;
        padding: 0.25rem 0.8rem;
        border-radius: 40px;
        font-weight: 600;
        margin: 0.2rem;
        font-size: 0.75rem;
        border: 1px solid #c084fc;
    }
    
    .stAlert {
        border-left: 5px solid #8b5cf6;
        background: #faf5ff !important;
        color: #4c1d95;
    }
    
    .stTextArea textarea, .stSelectbox select {
        background: #ffffff !important;
        border-radius: 24px !important;
        border: 1px solid #d8b4fe;
    }
    
    .element-container {
        margin-bottom: 0.2rem !important;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------
# SESSION STATE
# -------------------------
if "page" not in st.session_state:
    st.session_state.page = "upload"
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "skills" not in st.session_state:
    st.session_state.skills = []
if "score" not in st.session_state:
    st.session_state.score = 0
if "rewritten_bullet" not in st.session_state:
    st.session_state.rewritten_bullet = ""
if "jd_match" not in st.session_state:
    st.session_state.jd_match = None

# -------------------------
# SKILLS DATABASES
# -------------------------
extended_skills_db = [
    'python', 'sql', 'html', 'css', 'power bi', 'javascript', 'java',
    'react', 'node.js', 'mongodb', 'django', 'flask', 'aws', 'docker',
    'git', 'machine learning', 'data analysis', 'excel', 'tableau', 'c++',
    'c#', 'php', 'laravel', 'spring boot', 'kotlin', 'swift', 'ruby',
    'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy', 'matplotlib'
]
skills_db = ['python', 'sql', 'html', 'css', 'power bi', 'javascript', 'java']

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embed_model = load_embedding_model()
skill_embeddings = embed_model.encode(skills_db, convert_to_tensor=True)

# -------------------------
# HELPER FUNCTIONS (all features)
# -------------------------
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.lower()

def extract_skills_combined(text):
    found = []
    for skill in skills_db:
        if skill in text:
            found.append(skill)
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    if chunks:
        chunk_embs = embed_model.encode(chunks, convert_to_tensor=True)
        for idx, skill_emb in enumerate(skill_embeddings):
            sim = util.cos_sim(skill_emb, chunk_embs).max().item()
            if sim > 0.35 and skills_db[idx] not in found:
                found.append(skills_db[idx])
    return list(set(found))

def fuzzy_score(skills_count, has_degree):
    skills_var = ctrl.Antecedent(np.arange(0, 11, 1), 'skills')
    education = ctrl.Antecedent(np.arange(0, 2, 1), 'education')
    score = ctrl.Consequent(np.arange(0, 101, 1), 'score')
    skills_var['low'] = fuzz.trimf(skills_var.universe, [0, 0, 4])
    skills_var['medium'] = fuzz.trimf(skills_var.universe, [2, 5, 8])
    skills_var['high'] = fuzz.trimf(skills_var.universe, [6, 10, 10])
    education['no_degree'] = fuzz.trimf(education.universe, [0, 0, 0.5])
    education['has_degree'] = fuzz.trimf(education.universe, [0.5, 1, 1])
    score['poor'] = fuzz.trimf(score.universe, [0, 0, 40])
    score['average'] = fuzz.trimf(score.universe, [30, 50, 70])
    score['good'] = fuzz.trimf(score.universe, [60, 75, 90])
    score['excellent'] = fuzz.trimf(score.universe, [80, 100, 100])
    rules = [
        ctrl.Rule(skills_var['low'] & education['no_degree'], score['poor']),
        ctrl.Rule(skills_var['low'] & education['has_degree'], score['average']),
        ctrl.Rule(skills_var['medium'] & education['no_degree'], score['average']),
        ctrl.Rule(skills_var['medium'] & education['has_degree'], score['good']),
        ctrl.Rule(skills_var['high'] & education['no_degree'], score['good']),
        ctrl.Rule(skills_var['high'] & education['has_degree'], score['excellent']),
    ]
    scoring_ctrl = ctrl.ControlSystem(rules)
    scoring = ctrl.ControlSystemSimulation(scoring_ctrl)
    scoring.input['skills'] = min(skills_count, 10)
    scoring.input['education'] = 1 if has_degree else 0
    scoring.compute()
    return round(scoring.output['score'], 2)

def linear_score(skills, text):
    skill_score = min(len(skills) * 12, 60)
    edu_score = 20 if "b.e" in text or "bachelor" in text or "b.tech" in text else 10
    return skill_score + edu_score

def generate_questions_simple(skills, resume_text):
    if not skills:
        return ["📝 Please add more technical skills to your resume."]
    try:
        prompt = f"""
Generate 3 to 5 interview questions for a candidate with these skills: {', '.join(skills[:5])}.
Resume excerpt: {resume_text[:800]}

Output only the questions as a numbered list (1., 2., 3., etc.). Do not include answers.
"""
        response = ollama.chat(model='phi', messages=[{'role': 'user', 'content': prompt}])
        content = response['message']['content']
        lines = content.split('\n')
        questions = []
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                clean = re.sub(r'^[\d\-•]+\.?\s*', '', line)
                if clean and len(clean) > 10:
                    questions.append(clean)
        if len(questions) < 2:
            return [f"❓ What is your experience with {s}?" for s in skills[:3]]
        return questions[:5]
    except:
        return [f"❓ Describe your proficiency with {s}." for s in skills[:3]]

def get_missing_skills(detected_skills, full_db):
    detected_lower = [s.lower() for s in detected_skills]
    missing = [s for s in full_db if s.lower() not in detected_lower]
    return missing[:15]

def compare_with_jd(resume_text, jd_text):
    resume_words = set(re.findall(r'\b[a-z]{3,}\b', resume_text.lower()))
    jd_words = set(re.findall(r'\b[a-z]{3,}\b', jd_text.lower()))
    common = resume_words.intersection(jd_words)
    match_percent = len(common) / len(jd_words) * 100 if jd_words else 0
    missing_keywords = list(jd_words - resume_words)[:20]
    return round(match_percent, 2), missing_keywords

def rewrite_bullet(bullet_text):
    try:
        prompt = f"""
Rewrite the following resume bullet point to be more powerful, action-oriented, and quantifiable. Use strong verbs and metrics if possible.

Original: {bullet_text}

Rewritten version:
"""
        response = ollama.chat(model='phi', messages=[{'role': 'user', 'content': prompt}])
        return response['message']['content'].strip()
    except:
        return "✍️ Could not rewrite."

def ats_check(text):
    issues = []
    suggestions = []
    if len(text.split()) < 300:
        issues.append("📄 Resume is too short (less than 300 words).")
        suggestions.append("Add more details about projects and experience.")
    if "contact" not in text and "phone" not in text and "email" not in text:
        issues.append("📞 Missing contact information.")
        suggestions.append("Add phone/email at the top.")
    if "education" not in text and "degree" not in text:
        issues.append("🎓 Education section missing.")
        suggestions.append("Include degree and university.")
    if "experience" not in text and "work" not in text:
        issues.append("💼 Work experience missing.")
        suggestions.append("Add internships or job entries.")
    if "summary" not in text and "profile" not in text:
        suggestions.append("✨ Add a short professional summary.")
    suggestions.append("⚠️ Avoid tables/images – they confuse ATS.")
    return issues, suggestions

def generate_pdf_report(score, skills, missing, questions, feedback, jd_match=None):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    y = 750
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "📄 AI Resume Analyzer Pro - Report")
    y -= 30
    c.setFont("Helvetica", 12)
    c.drawString(50, y, f"✨ Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    y -= 30
    c.drawString(50, y, f"🎯 Resume Score: {score}/100")
    y -= 25
    c.drawString(50, y, "📊 Detected Skills: " + ", ".join(skills[:10]))
    y -= 25
    c.drawString(50, y, "📚 Missing Skills: " + ", ".join(missing[:10]))
    y -= 25
    if jd_match:
        c.drawString(50, y, f"🔍 JD Match: {jd_match}%")
        y -= 25
    c.drawString(50, y, "📋 Interview Questions:")
    y -= 15
    for q in questions[:3]:
        c.drawString(65, y, f"- {q[:80]}")
        y -= 15
    c.drawString(50, y, f"🤖 Feedback: {feedback[:200]}")
    c.save()
    buffer.seek(0)
    return buffer

def generate_feedback(score, skills):
    if score < 50:
        return f"⚠️ Low score ({score}/100). Add more skills and projects."
    elif score < 75:
        return f"✅ Good score ({score}/100). Consider certifications."
    else:
        return f"🎉 Excellent score ({score}/100)! Focus on interview practice."

def plot_pie_chart(score):
    fig, ax = plt.subplots(figsize=(6, 4))
    sizes = [score, 100 - score]
    labels = ['Score', 'Remaining']
    colors = ['#8b5cf6', '#e9d5ff']
    explode = (0.05, 0)
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
           shadow=True, startangle=90, textprops={'fontsize': 12, 'weight': 'bold', 'color': '#4c1d95'})
    ax.set_title('📊 Score Breakdown', fontsize=14, weight='bold', color='#6b21a5')
    ax.set_facecolor('none')
    fig.patch.set_alpha(0)
    return fig

def plot_bar_chart(score):
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.barh(['Score'], [score], color='#8b5cf6', height=0.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel('Score out of 100', color='#5b21b6')
    ax.set_title('🎯 Overall Resume Score', fontsize=14, weight='bold', color='#6b21a5')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#d8b4fe')
    ax.tick_params(axis='x', colors='#6b21a5')
    ax.tick_params(axis='y', left=False)
    for i, v in enumerate([score]):
        ax.text(v + 1, i, f'{v}%', va='center', fontweight='bold', color='#7c3aed')
    ax.set_facecolor('none')
    fig.patch.set_alpha(0)
    return fig

# -------------------------
# UPLOAD PAGE
# -------------------------
if st.session_state.page == "upload":
    st.markdown("""
    <div style="text-align: center;">
        <h1 style="font-size: 2rem; margin-bottom: 0;">📄🤖 AI Resume Analyzer Pro</h1>
        <p style="color: #7c3aed; margin-top: 0.2rem;">✨ Upload your PDF and get AI-powered insights, interview prep & ATS tips ✨</p>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        resume_file = st.file_uploader("📂 Choose PDF file", type=["pdf"], label_visibility="collapsed")
        if resume_file:
            with st.spinner("⚡ Extracting text..."):
                text = extract_text_from_pdf(resume_file)
                st.session_state.resume_text = text
            st.success("✅ Upload successful! Click below to analyze.")
            if st.button("🚀 Analyze Resume"):
                st.session_state.page = "analysis"
                st.rerun()

# -------------------------
# ANALYSIS PAGE
# -------------------------
elif st.session_state.page == "analysis":
    text = st.session_state.resume_text
    with st.spinner("⚡ Analyzing..."):
        skills = extract_skills_combined(text)
        has_degree = "b.e" in text or "bachelor" in text or "b.tech" in text
        try:
            score = fuzzy_score(len(skills), has_degree)
        except:
            score = linear_score(skills, text)
        st.session_state.skills = skills
        st.session_state.score = score
    st.subheader("🧠 Skills Detected")
    if skills:
        skills_html = "".join([f'<span class="skill-tag">{s}</span>' for s in skills])
        st.markdown(f'<div>{skills_html}</div>', unsafe_allow_html=True)
    else:
        st.warning("📄 No standard skills detected. Add keywords like Python, SQL, etc.")
    st.subheader(f"📊 Score: {score}/100")
    if st.button("✨ See Full Results"):
        st.session_state.page = "results"
        st.rerun()

# -------------------------
# RESULTS PAGE (all 5 features)
# -------------------------
elif st.session_state.page == "results":
    score = st.session_state.score
    skills = st.session_state.skills
    text = st.session_state.resume_text

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎯 Final Score", f"{score}/100")
    with col2:
        st.metric("💻 Skills Found", len(skills))
    with col3:
        st.metric("📄 Resume Length", f"{len(text.split())} words")

    st.pyplot(plot_pie_chart(score))
    st.pyplot(plot_bar_chart(score))

    st.write("### 🤖 AI Feedback")
    st.info(generate_feedback(score, skills))

    with st.expander("🔍 Compare with Job Description"):
        jd_text = st.text_area("📝 Paste job description here", height=150)
        if st.button("✨ Calculate Match"):
            match_score, missing_jd = compare_with_jd(text, jd_text)
            st.metric("🎯 JD Match Score", f"{match_score}%")
            if missing_jd:
                st.write("**📚 Missing keywords to add:**", ", ".join(missing_jd[:10]))
            st.session_state.jd_match = match_score

    st.markdown("---")
    st.write("### 📋 Interview Questions")
    questions = generate_questions_simple(skills, text)
    for q in questions:
        st.markdown(f"- {q}")

    st.markdown("---")
    st.write("### 📚 Missing Skills to Learn")
    missing = get_missing_skills(skills, extended_skills_db)
    if missing:
        cols = st.columns(4)
        for i, m in enumerate(missing):
            cols[i % 4].markdown(f"📖 `{m}`")
        st.caption("✨ Add these skills to increase your score and attract recruiters.")
    else:
        st.success("🎉 Great job! You have all the suggested technical skills.")

    with st.expander("⚙️ ATS Compatibility Check"):
        issues, suggestions = ats_check(text)
        if issues:
            st.warning("⚠️ Issues found:")
            for issue in issues:
                st.write(f"- {issue}")
        else:
            st.success("✅ No major ATS issues detected!")
        st.write("**💡 Suggestions:**")
        for sug in suggestions:
            st.write(f"- {sug}")

    with st.expander("✍️ Resume Bullet Point Rewriter"):
        lines = text.split('\n')
        bullets = [l.strip() for l in lines if l.strip() and (l.strip()[0] in '-•*' or re.match(r'^\d+\.', l.strip()))]
        if bullets:
            selected = st.selectbox("📌 Select a bullet point to rewrite", bullets)
            if st.button("✨ Rewrite with AI"):
                rewritten = rewrite_bullet(selected)
                st.session_state.rewritten_bullet = rewritten
                st.success("✍️ Rewritten!")
            if st.session_state.rewritten_bullet:
                st.text_area("✨ Improved version", st.session_state.rewritten_bullet, height=100)
                st.code("📋 Copy the text above manually (Ctrl+C)", language="text")
        else:
            st.info("📄 No bullet points detected in your resume text.")

    st.markdown("---")
    jd_match_val = getattr(st.session_state, 'jd_match', None)
    pdf_buffer = generate_pdf_report(score, skills, missing, questions, generate_feedback(score, skills), jd_match_val)
    st.download_button("📥 Download Report as PDF", pdf_buffer, "resume_report.pdf", mime="application/pdf")

    if st.button("← Back to Upload New Resume"):
        st.session_state.page = "upload"
        st.rerun()
