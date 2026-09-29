import streamlit as st
import pytesseract
import pymupdf as fitz
from PIL import Image
from google import genai
import io
import os
import time
import platform
from datetime import datetime
from dotenv import load_dotenv

# =============================
# LOAD ENV
# =============================
load_dotenv()

# =============================
# CONFIG
# =============================
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load API key: .env (local) OR Streamlit secrets (cloud)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    try:
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        GOOGLE_API_KEY = None

GEMINI_MODEL = "gemini-3.6-flash"  # primary
FALLBACK_MODELS = ["gemini-flash-latest", "gemini-pro-latest", "gemini-3.6-flash"]
GITHUB_LINK = "https://github.com/ShadabAttar007"

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(page_title="AI Study Buddy", page_icon="🚀", layout="wide")

# =============================
# API KEY CHECK
# =============================
if not GOOGLE_API_KEY:
    st.error("⚠️ GOOGLE_API_KEY not found. Add it to `.env` locally or to Streamlit Secrets on cloud.")
    st.stop()

# =============================
# GEMINI CLIENT
# =============================
@st.cache_resource
def get_gemini_client():
    return genai.Client(api_key=GOOGLE_API_KEY)

# =============================
# SESSION STATE
# =============================
defaults = {
    "subjects": {},
    "total_quizzes": 0,
    "total_flashcards": 0,
    "active_tool": "home",
    "chat_history": [],
    "profile_name": "Student",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =============================
# CSS
# =============================
st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at 50% 0%, #2a0a0a 0%, #0a0a0c 50%, #000000 100%);
        color: #E0E0E0;
    }
    [data-testid="stSidebar"] {
        background: rgba(10, 10, 12, 0.85) !important;
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 50, 50, 0.2) !important;
    }
    h1, h2, h3, h4, h5, h6, p, div, label, span {
        color: #FFFFFF !important;
        font-family: 'Segoe UI', sans-serif;
    }
    .stCaption, small { color: #9CA3AF !important; }
    .stTextInput input, .stTextArea textarea {
        background: rgba(22, 27, 34, 0.8) !important;
        color: #FFFFFF !important;
        border: 1px solid #30363D !important;
        border-radius: 8px !important;
    }
    .stButton > button {
        background: rgba(22, 27, 34, 0.6) !important;
        color: #E0E0E0 !important;
        border: 1px solid #30363D !important;
        border-radius: 12px !important;
        padding: 20px !important;
        height: 170px !important;
        font-size: 16px !important;
        font-weight: bold !important;
        white-space: pre-wrap !important;
        backdrop-filter: blur(5px);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        border-color: #FF4B4B !important;
        background: rgba(255, 75, 75, 0.15) !important;
        transform: translateY(-5px) scale(1.02) !important;
        box-shadow: 0 15px 30px rgba(255, 75, 75, 0.3) !important;
    }
    .hero-card {
        background: linear-gradient(135deg, #FF4B4B 0%, #E01A1A 100%);
        padding: 30px;
        border-radius: 16px;
        margin-bottom: 25px;
        box-shadow: 0 20px 50px rgba(255, 75, 75, 0.5);
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .hero-card h1 { color: #FFFFFF !important; margin: 0; font-size: 3rem; font-weight: 900; }
    .hero-card p { color: #FFD5D5 !important; margin-top: 5px; font-size: 1.2rem; }
    .github-btn {
        display: block; text-align: center;
        background: rgba(255, 75, 75, 0.1);
        color: #FF4B4B !important; padding: 10px; border-radius: 8px;
        border: 1px solid #FF4B4B; text-decoration: none; font-weight: bold;
        margin-top: 10px; transition: 0.3s;
    }
    .github-btn:hover { background: #FF4B4B; color: #FFFFFF !important; }
    .user-msg {
        text-align: right; background: #FF4B4B !important; color: #FFFFFF !important;
        padding: 10px; border-radius: 10px 0 10px 10px; margin-bottom: 10px;
    }
    .bot-msg {
        text-align: left; background: rgba(22, 27, 34, 0.9) !important;
        color: #FFFFFF !important; padding: 10px; border-radius: 0 10px 10px 10px;
        margin-bottom: 10px; border: 1px solid #30363D;
    }
</style>
""", unsafe_allow_html=True)

# =============================
# HELPERS
# =============================
import time

# List of models to try, from newest to older
FALLBACK_MODELS = [
    "gemini-3.6-flash",   # Your current model
    "gemini-3.5-flash",   # Next best option
    "gemini-2.5-flash",   # A stable fallback
]

def call_gemini(prompt, max_retries=3):
    """Try multiple models with retries to handle 503 high-demand errors."""
    last_error = None
    
    for model in FALLBACK_MODELS:
        for attempt in range(max_retries):
            try:
                client = get_gemini_client()
                response = client.models.generate_content(
                    model=model,
                    contents=prompt
                )
                return response.text if response.text else "No response."
                
            except Exception as e:
                last_error = str(e)
                
                # If the model doesn't exist, skip to the next one immediately
                if "404" in last_error or "not found" in last_error.lower():
                    break 
                
                # If the model is just busy (503), wait and retry this same model
                if "503" in last_error or "UNAVAILABLE" in last_error:
                    if attempt < max_retries - 1:
                        time.sleep(3 * (attempt + 1)) # Wait 3s, 6s...
                        continue
                
                # For any other error, stop trying
                break
                
    return f"⚠️ All models are busy or unavailable. Please try again in a moment."


def extract_text(uploaded_file):
    try:
        file_bytes = uploaded_file.getvalue()
        file_ext = uploaded_file.name.rsplit(".", 1)[-1].lower()

        if file_ext == "txt":
            return file_bytes.decode("utf-8")

        elif file_ext == "pdf":
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            text = "\n".join([page.get_text() for page in doc])
            doc.close()
            return text

        else:
            img = Image.open(io.BytesIO(file_bytes))
            return pytesseract.image_to_string(img)

    except Exception as e:
        st.error(f"Extraction error: {e}")
        return None


def get_greeting():
    h = datetime.now().hour
    if 5 <= h < 12:
        return "Good Morning"
    elif 12 <= h < 17:
        return "Good Afternoon"
    elif 17 <= h < 21:
        return "Good Evening"
    return "Good Night"

# =============================
# SIDEBAR
# =============================
with st.sidebar:
    st.markdown("<h2 style='text-align:center; color:#FF4B4B;'>🚀 AI</h2>", unsafe_allow_html=True)
    st.divider()

    st.session_state.profile_name = st.text_input(
        "Profile Name", value=st.session_state.profile_name
    )

    streak = st.session_state.total_quizzes + st.session_state.total_flashcards
    st.caption(f"🔥 Streak: {streak} actions")

    st.divider()
    st.markdown("### 📅 Calendar")
    st.caption(f"Today: **{datetime.now().strftime('%A, %B %d, %Y')}**")

    st.markdown("---")
    st.markdown(
        f'<a href="{GITHUB_LINK}" target="_blank" class="github-btn">🐙 Visit GitHub</a>',
        unsafe_allow_html=True
    )

    st.divider()
    if st.button("🏠 Home"):
        st.session_state.active_tool = "home"
        st.rerun()

# =============================
# HERO
# =============================
st.markdown("""
<div class="hero-card">
    <h1>Build the Future</h1>
    <p>with Artificial Intelligence</p>
</div>
""", unsafe_allow_html=True)

st.markdown(f"### {get_greeting()}, {st.session_state.profile_name}!")
st.caption("What would you like to learn today?")
st.divider()

# =============================
# ACTION CARDS
# =============================
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    if st.button("⬆️\n\nUpload\nMaterial\n\nPDF, TXT, IMG"):
        st.session_state.active_tool = "upload"
        st.rerun()

with c2:
    if st.button("🧠\n\nAsk AI\nTutor\n\nGet Answers"):
        st.session_state.active_tool = "tutor"
        st.rerun()

with c3:
    if st.button("📝\n\nGenerate\nNotes\n\nSmart Notes"):
        st.session_state.active_tool = "notes"
        st.rerun()

with c4:
    if st.button("❓\n\nCreate\nQuiz\n\nMCQs"):
        st.session_state.active_tool = "quiz"
        st.rerun()

with c5:
    if st.button("🃏\n\nCreate\nFlashcards\n\nRevision"):
        st.session_state.active_tool = "flashcards"
        st.rerun()

st.divider()

# =============================
# DYNAMIC MODULES
# =============================
tool = st.session_state.active_tool

# ---------- HOME ----------
if tool == "home":
    st.subheader("📁 Recent Materials")
    if st.session_state.subjects:
        for subj in st.session_state.subjects:
            with st.container(border=True):
                st.markdown(f"**📄 {subj}**")
    else:
        st.info("No materials yet. Click **Upload** above to get started.")

# ---------- UPLOAD ----------
elif tool == "upload":
    st.subheader("📤 Upload Material")
    subj = st.text_input("Subject Name (e.g., Biology)")
    file = st.file_uploader("Choose file", type=["pdf", "txt", "png", "jpg", "jpeg"])

    if st.button("🚀 Analyze"):
        if subj and file:
            with st.spinner("Extracting text..."):
                text = extract_text(file)

            if text and text.strip():
                with st.spinner("AI is generating your notes..."):
                    notes = call_gemini(
                        f"Create comprehensive study notes for the following material:\n\n{text[:15000]}"
                    )
                st.session_state.subjects[subj] = {"Notes": notes}
                st.success(f"✅ Notes for '{subj}' created!")
                st.session_state.active_tool = "notes"
                st.rerun()
            else:
                st.error("❌ Text extraction failed or file is empty.")
        else:
            st.warning("Please enter a subject name and upload a file.")

# ---------- TUTOR ----------
elif tool == "tutor":
    st.subheader("🧠 AI Tutor")

    for msg in st.session_state.chat_history:
        cls = "user-msg" if msg["role"] == "user" else "bot-msg"
        icon = "🧑‍🎓" if msg["role"] == "user" else "🤖"
        st.markdown(
            f'<div class="{cls}">{icon} {msg["content"]}</div>',
            unsafe_allow_html=True
        )

    user_q = st.text_area("Ask anything...", key="tutor_input")

    b1, b2 = st.columns([1, 5])
    with b1:
        if st.button("Send"):
            if user_q.strip():
                st.session_state.chat_history.append(
                    {"role": "user", "content": user_q}
                )
                with st.spinner("Thinking..."):
                    resp = call_gemini(user_q)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": resp}
                )
                st.rerun()
    with b2:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

# ---------- NOTES ----------
elif tool == "notes":
    st.subheader("📚 My Notes")
    if not st.session_state.subjects:
        st.info("No notes yet. Upload a material first.")
    else:
        for subj, data in st.session_state.subjects.items():
            with st.expander(f"📁 {subj}", expanded=False):
                st.markdown(data["Notes"])

# ---------- QUIZ ----------
elif tool == "quiz":
    st.subheader("📝 Quiz Generator")
    subjects = list(st.session_state.subjects.keys())

    if subjects:
        sel = st.selectbox("Pick Subject", subjects)
        if st.button("Generate Quiz"):
            with st.spinner("Creating quiz..."):
                resp = call_gemini(
                    f"Create 5 multiple-choice questions with correct answers based on:\n\n"
                    f"{st.session_state.subjects[sel]['Notes'][:5000]}"
                )
            st.markdown(resp)
            st.session_state.total_quizzes += 1
    else:
        st.warning("Upload material first.")

# ---------- FLASHCARDS ----------
elif tool == "flashcards":
    st.subheader("🃏 Flashcards")
    subjects = list(st.session_state.subjects.keys())

    if subjects:
        sel = st.selectbox("Pick Subject", subjects)
        if st.button("Generate Cards"):
            with st.spinner("Creating flashcards..."):
                resp = call_gemini(
                    f"Create 5 question-and-answer flashcards based on:\n\n"
                    f"{st.session_state.subjects[sel]['Notes'][:5000]}"
                )
            st.markdown(resp)
            st.session_state.total_flashcards += 1
    else:
        st.warning("Upload material first.")

# =============================
# FOOTER
# =============================
st.divider()
st.markdown(
    '<div style="text-align:center;padding:10px;color:#9CA3AF;">'
    '"The beautiful thing about learning is that no one can take it away from you. — B.B. King"'
    '</div>',
    unsafe_allow_html=True
)
st.markdown(
    f'<div style="text-align:center;padding:10px;color:#9CA3AF;">'
    f'Built with ❤️ by <a href="{GITHUB_LINK}" target="_blank" style="color:#FF4B4B;">Shadab Attar</a>'
    f'</div>',
    unsafe_allow_html=True
)