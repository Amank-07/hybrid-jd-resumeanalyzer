import streamlit as st
import os
import json
import re
from openai import OpenAI
from typing import Dict, Any, Optional, Dict
import PyPDF2
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page configuration
st.set_page_config(
    page_title="AI JD → Resume Modifier",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.1rem;
    }
    .input-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #e9ecef;
        margin-bottom: 1rem;
    }
    .result-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .skill-badge {
        display: inline-block;
        background-color: #e3f2fd;
        color: #1976d2;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.25rem;
        font-size: 0.9rem;
    }
    .keyword-tag {
        display: inline-block;
        background-color: #fff3e0;
        color: #e65100;
        padding: 0.4rem 0.8rem;
        border-radius: 15px;
        margin: 0.2rem;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .stProgress > div > div > div > div {
        background-color: #667eea;
    }
    .upload-area {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f8f9fa;
    }
    </style>
""", unsafe_allow_html=True)

# Header with gradient
st.markdown("""
    <div class="main-header">
        <h1>📝 AI Job Description → Resume Modifier</h1>
        <p>Get instant AI-powered insights to optimize your resume for any job posting</p>
    </div>
""", unsafe_allow_html=True)

# Check for API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("⚠️ Please set the OPENAI_API_KEY environment variable to use this app.")
    st.stop()

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

def preprocess_text_for_nlp(text: str) -> str:
    """
    Basic text preprocessing for NLP:
    - Lowercasing
    - Removing non-alphanumeric characters
    - Normalizing whitespace
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def compute_nlp_metrics(jd_text: str, resume_text: str) -> Dict[str, Any]:
    """
    Compute simple NLP-based similarity and keyword overlap metrics
    between the job description and the resume using TF-IDF and cosine
    similarity.

    This is a classical NLP technique that does not rely on the LLM.
    It helps justify the project as an NLP-based system for research.
    """
    try:
        clean_jd = preprocess_text_for_nlp(jd_text)
        clean_resume = preprocess_text_for_nlp(resume_text)

        if not clean_jd or not clean_resume:
            return {}

        # Build TF-IDF vectors for both texts
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform([clean_jd, clean_resume])

        # Cosine similarity between JD and resume
        similarity = cosine_similarity(
            tfidf_matrix[0:1],
            tfidf_matrix[1:2]
        )[0][0]

        feature_names = vectorizer.get_feature_names_out()
        jd_scores = tfidf_matrix.toarray()[0]
        resume_scores = tfidf_matrix.toarray()[1]

        # Top keywords from each text
        top_k = 15
        jd_top_indices = jd_scores.argsort()[::-1][:top_k]
        resume_top_indices = resume_scores.argsort()[::-1][:top_k]

        jd_keywords = [feature_names[i] for i in jd_top_indices]
        resume_keywords = [feature_names[i] for i in resume_top_indices]

        jd_set = set(jd_keywords)
        resume_set = set(resume_keywords)

        overlap_keywords = sorted(jd_set & resume_set)
        missing_keywords = [kw for kw in jd_keywords if kw not in resume_set]

        return {
            "text_similarity": round(float(similarity) * 100, 1),  # percentage
            "jd_top_keywords": jd_keywords,
            "resume_top_keywords": resume_keywords,
            "overlap_keywords": overlap_keywords,
            "missing_keywords": missing_keywords,
        }
    except Exception:
        # In case anything goes wrong, fail gracefully and skip NLP metrics
        return {}

def extract_text_from_pdf(uploaded_file) -> Optional[str]:
    """
    Extract text from an uploaded PDF file.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        Extracted text as string, or None if extraction fails
    """
    try:
        # Read the PDF file
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        
        # Extract text from all pages
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n"
        
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None

def build_llm_prompt(jd_text: str, resume_text: str) -> str:
    """
    Build a structured prompt for the LLM to analyze resume against job description.
    
    Args:
        jd_text: Job description text
        resume_text: Resume text
        
    Returns:
        Formatted prompt string
    """
    prompt = f"""You are an expert ATS (Applicant Tracking System) and resume analyzer. 
Your task is to compare a job description with a candidate's resume and identify gaps, 
suggest improvements, and provide a fit assessment.

Job Description:
{jd_text}

Resume:
{resume_text}

Analyze the resume against the job description and respond ONLY in valid JSON with this exact structure:
{{ 
  "current_ats_score": 65,
  "expected_ats_score": 85,
  "missing_skills": ["skill1", "skill2", "..."],
  "bullet_points_to_add": ["bullet point 1", "bullet point 2", "..."],
  "keywords_ats": ["keyword1", "keyword2", "..."],
  "fit_percentage": 75,
  "summary_feedback": "A brief explanation of the fit percentage and overall assessment.",
  "practice_questions": ["question 1", "question 2", "..."]
}}

Important:
- current_ats_score: A number between 0-100 representing the current ATS score of the resume based on keyword matching, formatting, and JD alignment
- expected_ats_score: A number between 0-100 representing the expected ATS score if all suggested changes (missing skills, bullet points, keywords) are incorporated into the resume
- missing_skills: List of skills mentioned in the JD but not clearly evident in the resume
- bullet_points_to_add: Specific, actionable bullet points the candidate should add to their resume
- keywords_ats: Important keywords/phrases from the JD that ATS systems typically look for but are missing or underrepresented in the resume
- fit_percentage: A number between 0-100 representing how well the resume matches the JD overall
- summary_feedback: A concise explanation (2-3 sentences) of the fit percentage and overall assessment
- practice_questions: A list of clear, interview-style questions derived from the job description that the candidate can practice to become stronger for this role

Respond ONLY with valid JSON. Do not include any text before or after the JSON."""
    
    return prompt

def analyze_resume(jd_text: str, resume_text: str) -> Dict[str, Any]:
    """
    Call the LLM API to analyze resume against job description.
    
    Args:
        jd_text: Job description text
        resume_text: Resume text
        
    Returns:
        Dictionary with analysis results
    """
    try:
        prompt = build_llm_prompt(jd_text, resume_text)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using gpt-4o-mini for cost efficiency, can be changed to gpt-4o
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert ATS and resume analyzer. You compare job descriptions with resumes and provide structured, actionable feedback. Always respond in valid JSON format only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,  # Lower temperature for more consistent, structured output
            response_format={"type": "json_object"}  # Request JSON format
        )
        
        # Extract and parse JSON response
        response_text = response.choices[0].message.content.strip()
        
        # Try to parse JSON
        try:
            result = json.loads(response_text)
            return result
        except json.JSONDecodeError as e:
            st.error(f"Failed to parse JSON response: {e}")
            st.code(response_text, language="text")
            return None
            
    except Exception as e:
        st.error(f"Error calling LLM API: {str(e)}")
        return None

def display_results(results: Dict[str, Any], nlp_metrics: Optional[Dict[str, Any]] = None):
    """
    Display analysis results in a formatted way.
    
    Args:
        results: Dictionary containing analysis results
    """
    if not results:
        return
    
    # Success banner
    st.success("✅ Analysis Complete! Review your results below.")
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ATS Score Section - Prominent Display with Cards
    st.markdown("### 🎯 ATS Score Analysis")
    st.markdown("---")
    
    current_ats = results.get("current_ats_score", 0)
    expected_ats = results.get("expected_ats_score", 0)
    improvement = expected_ats - current_ats
    
    # Create three columns for side-by-side comparison
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 1.5rem; border-radius: 10px; color: white; text-align: center;'>
                <h3 style='color: white; margin-bottom: 0.5rem;'>Current Score</h3>
                <h1 style='color: white; font-size: 3rem; margin: 0.5rem 0;'>{}</h1>
            </div>
        """.format(f"{current_ats}%"), unsafe_allow_html=True)
        st.progress(current_ats / 100)
        st.caption("Your resume's current ATS compatibility")
    
    with col2:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                        padding: 1.5rem; border-radius: 10px; color: white; text-align: center;'>
                <h3 style='color: white; margin-bottom: 0.5rem;'>Expected Score</h3>
                <h1 style='color: white; font-size: 3rem; margin: 0.5rem 0;'>{}</h1>
            </div>
        """.format(f"{expected_ats}%"), unsafe_allow_html=True)
        st.progress(expected_ats / 100)
        st.caption("Potential score after improvements")
    
    with col3:
        improvement_emoji = "📈" if improvement > 0 else "📉" if improvement < 0 else "➡️"
        improvement_color = "#10b981" if improvement > 0 else "#ef4444" if improvement < 0 else "#6b7280"
        st.markdown(f"""
            <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                        padding: 1.5rem; border-radius: 10px; color: white; text-align: center;'>
                <h3 style='color: white; margin-bottom: 0.5rem;'>Improvement</h3>
                <h1 style='color: white; font-size: 3rem; margin: 0.5rem 0;'>{improvement_emoji} {improvement:+d}%</h1>
            </div>
        """, unsafe_allow_html=True)
        if improvement > 0:
            st.success(f"🎉 Your resume could improve by **{improvement}%** with suggested changes!")
        elif improvement < 0:
            st.warning("⚠️ Review suggestions carefully.")
        else:
            st.info("No change expected.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Overall Fit and Summary in a card
    st.markdown("### 📊 Overall Assessment")
    
    fit_pct = results.get("fit_percentage", 0)
    summary = results.get("summary_feedback", "No feedback available.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"""
            <div style='background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; text-align: center; border-left: 4px solid #667eea;'>
                <h2 style='color: #667eea; margin: 0;'>{fit_pct}%</h2>
                <p style='color: #6b7280; margin-top: 0.5rem;'>Overall Match</p>
            </div>
        """, unsafe_allow_html=True)
        st.progress(fit_pct / 100)
    
    with col2:
        st.markdown(f"""
            <div style='background-color: #f0f9ff; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #0ea5e9;'>
                <h4 style='color: #0c4a6e; margin-top: 0;'>💡 Assessment</h4>
                <p style='color: #075985;'>{summary}</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Missing Skills - Enhanced Display
    st.markdown("### 🔍 Missing Skills")
    missing_skills = results.get("missing_skills", [])
    if missing_skills:
        st.markdown(f"<p style='color: #6b7280;'>Found <strong>{len(missing_skills)}</strong> skills mentioned in the JD that need more emphasis in your resume:</p>", unsafe_allow_html=True)
        for skill in missing_skills:
            st.markdown(f"""
                <div style='background-color: #fef3c7; padding: 0.75rem 1rem; border-radius: 8px; 
                            margin: 0.5rem 0; border-left: 3px solid #f59e0b;'>
                    <span style='color: #92400e;'>⚠️ {skill}</span>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.success("🎉 No missing skills identified! Your resume covers all required skills.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Bullet Points to Add - Enhanced Display
    st.markdown("### ✏️ Suggested Bullet Points to Add")
    bullet_points = results.get("bullet_points_to_add", [])
    if bullet_points:
        st.markdown(f"<p style='color: #6b7280;'>Consider adding these <strong>{len(bullet_points)}</strong> bullet points to strengthen your resume:</p>", unsafe_allow_html=True)
        for i, point in enumerate(bullet_points, 1):
            st.markdown(f"""
                <div style='background-color: #ecfdf5; padding: 1rem; border-radius: 8px; 
                            margin: 0.75rem 0; border-left: 4px solid #10b981;'>
                    <p style='color: #065f46; margin: 0;'>
                        <strong style='color: #10b981;'>{i}.</strong> {point}
                    </p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("✨ No specific bullet points suggested. Your resume structure looks good!")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    
    # ATS Keywords - Enhanced Display
    st.markdown("### 🔑 Important ATS Keywords")
    keywords = results.get("keywords_ats", [])
    if keywords:
        st.markdown(f"<p style='color: #6b7280;'>These <strong>{len(keywords)}</strong> keywords are important for ATS systems. Consider incorporating them naturally:</p>", unsafe_allow_html=True)
        st.markdown("<div style='padding: 1rem; background-color: #f8f9fa; border-radius: 10px;'>", unsafe_allow_html=True)
        for kw in keywords:
            st.markdown(f"""
                <span style='display: inline-block; background-color: #fff3e0; color: #e65100; 
                            padding: 0.5rem 1rem; border-radius: 20px; margin: 0.3rem; 
                            font-weight: 500; font-size: 0.9rem;'>
                    {kw}
                </span>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("✨ No additional keywords identified. Your resume already includes relevant terms!")

    # Practice questions section (LLM-generated) to help user prepare
    practice_questions = results.get("practice_questions", [])
    if practice_questions:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### 📝 Practice Questions Based on This JD")
        st.markdown(
            "<p style='color: #6b7280;'>Use these questions to practice for interviews targeting this role. "
            "They are derived from the job description and focus on required skills and responsibilities.</p>",
            unsafe_allow_html=True,
        )
        for i, q in enumerate(practice_questions, 1):
            st.markdown(f"""
                <div style='background-color: #f9fafb; padding: 0.9rem 1.1rem; border-radius: 8px; 
                            margin: 0.4rem 0; border-left: 3px solid #6366f1;'>
                    <p style='color: #111827; margin: 0; font-size: 0.95rem;'>
                        <strong style='color: #4f46e5;'>{i}.</strong> {q}
                    </p>
                </div>
            """, unsafe_allow_html=True)

    # Classical NLP analysis section (TF-IDF + cosine similarity)
    if nlp_metrics:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### 🔬 NLP Analysis (TF-IDF & Similarity)")
        st.markdown(
            "<p style='color: #6b7280;'>These metrics are computed using traditional NLP techniques, "
            "independent of the LLM, to quantify how similar the texts are and how well keywords overlap.</p>",
            unsafe_allow_html=True,
        )

        similarity = nlp_metrics.get("text_similarity", 0.0)
        overlap_keywords = nlp_metrics.get("overlap_keywords", [])
        missing_keywords = nlp_metrics.get("missing_keywords", [])

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
                <div style='background-color: #f0fdf4; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #10b981;'>
                    <h4 style='color: #065f46; margin-top: 0;'>🔗 Text Similarity (Cosine)</h4>
                    <h2 style='color: #10b981; margin: 0.5rem 0;'>{similarity}%</h2>
                    <p style='color: #047857; margin: 0; font-size: 0.9rem;'>
                        Higher values mean the JD and resume share more similar terms.
                    </p>
                </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
                <div style='background-color: #eff6ff; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #3b82f6;'>
                    <h4 style='color: #1d4ed8; margin-top: 0;'>📚 Keyword Overlap (TF-IDF)</h4>
                    <p style='color: #1e40af; margin: 0; font-size: 0.9rem;'>
                        Overlapping keywords: <strong>{len(overlap_keywords)}</strong> | 
                        Missing important JD keywords: <strong>{len(missing_keywords)}</strong>
                    </p>
                </div>
            """, unsafe_allow_html=True)

        if overlap_keywords:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Shared high-importance keywords between JD and resume:**")
            st.markdown("<div style='padding: 0.5rem;'>", unsafe_allow_html=True)
            for kw in overlap_keywords[:15]:
                st.markdown(f"""
                    <span style='display: inline-block; background-color: #e0f2fe; color: #0369a1; 
                                padding: 0.3rem 0.8rem; border-radius: 15px; margin: 0.2rem; 
                                font-size: 0.85rem;'>
                        {kw}
                    </span>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        if missing_keywords:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Important JD keywords not strongly present in the resume (TF-IDF-based):**")
            st.markdown("<div style='padding: 0.5rem;'>", unsafe_allow_html=True)
            for kw in missing_keywords[:15]:
                st.markdown(f"""
                    <span style='display: inline-block; background-color: #fef3c7; color: #92400e; 
                                padding: 0.3rem 0.8rem; border-radius: 15px; margin: 0.2rem; 
                                font-size: 0.85rem;'>
                        {kw}
                    </span>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

# Main app layout
st.markdown("### 📥 Input Your Documents")
st.markdown("Upload PDFs or paste text directly. Both methods work seamlessly!")
st.markdown("<br>", unsafe_allow_html=True)

# Input sections with better styling
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
        <div style='background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; 
                    border: 2px solid #e9ecef; margin-bottom: 1rem;'>
            <h3 style='color: #667eea; margin-top: 0;'>📋 Job Description</h3>
        </div>
    """, unsafe_allow_html=True)
    
    # JD Upload option
    jd_file = st.file_uploader(
        "📎 Upload Job Description PDF",
        type=["pdf"],
        key="jd_file_upload",
        help="Upload a PDF file containing the job description"
    )
    
    if jd_file is not None:
        st.info(f"📄 File uploaded: **{jd_file.name}**")
    
    st.markdown("""
        <div style='text-align: center; color: #6b7280; margin: 1rem 0;'>
            <strong>─ OR ─</strong>
        </div>
    """, unsafe_allow_html=True)
    
    # JD Text input
    jd_text_input = st.text_area(
        "✍️ Paste the job description here:",
        height=280,
        placeholder="Paste the complete job description text here...\n\nExample:\nJob Title: Software Engineer\nCompany: Tech Corp\nRequirements: Python, React, 3+ years experience...",
        key="jd_input",
        help="Copy and paste the job description from the job posting"
    )
    
    # Process JD: prioritize uploaded file, fallback to text input
    jd_text = ""
    if jd_file is not None:
        with st.spinner("⏳ Extracting text from JD PDF..."):
            extracted_text = extract_text_from_pdf(jd_file)
            if extracted_text:
                jd_text = extracted_text
                st.success(f"✅ Successfully extracted **{len(jd_text):,}** characters from JD PDF")
                # Show preview
                with st.expander("👁️ Preview extracted JD text (first 500 chars)"):
                    st.text(jd_text[:500] + "..." if len(jd_text) > 500 else jd_text)
    else:
        jd_text = jd_text_input
        if jd_text.strip():
            st.info(f"📝 Text input: **{len(jd_text):,}** characters")

with col2:
    st.markdown("""
        <div style='background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; 
                    border: 2px solid #e9ecef; margin-bottom: 1rem;'>
            <h3 style='color: #667eea; margin-top: 0;'>📄 Resume</h3>
        </div>
    """, unsafe_allow_html=True)
    
    # Resume Upload option
    resume_file = st.file_uploader(
        "📎 Upload Resume PDF",
        type=["pdf"],
        key="resume_file_upload",
        help="Upload a PDF file containing your resume"
    )
    
    if resume_file is not None:
        st.info(f"📄 File uploaded: **{resume_file.name}**")
    
    st.markdown("""
        <div style='text-align: center; color: #6b7280; margin: 1rem 0;'>
            <strong>─ OR ─</strong>
        </div>
    """, unsafe_allow_html=True)
    
    # Resume Text input
    resume_text_input = st.text_area(
        "✍️ Paste your resume here:",
        height=280,
        placeholder="Paste your complete resume text here...\n\nExample:\nJohn Doe\nSoftware Engineer\nSkills: Python, React, JavaScript...",
        key="resume_input",
        help="Copy and paste your resume text"
    )
    
    # Process Resume: prioritize uploaded file, fallback to text input
    resume_text = ""
    if resume_file is not None:
        with st.spinner("⏳ Extracting text from Resume PDF..."):
            extracted_text = extract_text_from_pdf(resume_file)
            if extracted_text:
                resume_text = extracted_text
                st.success(f"✅ Successfully extracted **{len(resume_text):,}** characters from Resume PDF")
                # Show preview
                with st.expander("👁️ Preview extracted Resume text (first 500 chars)"):
                    st.text(resume_text[:500] + "..." if len(resume_text) > 500 else resume_text)
    else:
        resume_text = resume_text_input
        if resume_text.strip():
            st.info(f"📝 Text input: **{len(resume_text):,}** characters")

st.markdown("<br>", unsafe_allow_html=True)

# Analyze button with better styling
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    analyze_clicked = st.button(
        "🚀 Analyze Resume Now",
        type="primary",
        use_container_width=True,
        help="Click to start the AI analysis"
    )

st.markdown("<br>", unsafe_allow_html=True)

# Analyze button logic
if analyze_clicked:
    # Validation with better error messages
    if not jd_text.strip():
        st.error("""
            ⚠️ **Job Description Required**
            
            Please either:
            - Upload a JD PDF file, OR
            - Paste the job description text in the text area
        """)
        st.stop()
    
    if not resume_text.strip():
        st.error("""
            ⚠️ **Resume Required**
            
            Please either:
            - Upload a Resume PDF file, OR
            - Paste your resume text in the text area
        """)
        st.stop()
    
    # Show loading with better UX
    with st.spinner("🤖 AI is analyzing your resume against the job description... This may take 10-30 seconds."):
        # First compute classical NLP metrics locally (no API involved)
        nlp_metrics = compute_nlp_metrics(jd_text, resume_text)
        # Then call the LLM for rich, semantic analysis
        results = analyze_resume(jd_text, resume_text)
    
    if results:
        st.markdown("<br>", unsafe_allow_html=True)
        st.balloons()  # Celebration animation
        # Pass both LLM results and NLP metrics to the renderer
        display_results(results, nlp_metrics)
    else:
        st.error("""
            ❌ **Analysis Failed**
            
            Please check:
            - Your internet connection
            - API key is valid
            - Try again in a moment
        """)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #6b7280; padding: 1rem;'>
        <p style='margin: 0.5rem 0;'>
            <strong>AI JD → Resume Modifier</strong> | Built with ❤️ using Streamlit & OpenAI
        </p>
        <p style='margin: 0.5rem 0; font-size: 0.85rem;'>
            Get instant insights to optimize your resume and land your dream job
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

