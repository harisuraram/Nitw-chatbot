import streamlit as st
from chatbot import (
    load_gemini_client,
    load_embedding_model,
    load_vector_database,
    query_chatbot
)

# Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Cache Resources (Load once) ---
@st.cache_resource
def initialize_chatbot():
    """Load all models and database once at startup."""
    try:
        gemini_client = load_gemini_client()
        embedding_model = load_embedding_model()
        index, metadata = load_vector_database()
        
        return gemini_client, embedding_model, index, metadata
    
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return None, None, None, None

# --- Page setup ---
st.set_page_config(page_title="NITW Assistant", page_icon="🧑‍💻", layout="centered")

# --- Neon Theme CSS ---
st.markdown("""
  <style>
    body {
        background-color: #0E1117;
        color: #FAFAFA;
        font-family: 'Segoe UI', sans-serif;
    }
    .stTextInput > div > div > input {
        background-color: #1E1E1E;
        color: #00FFFF;
        border: 1px solid #00FFFF;
        border-radius: 8px;
        padding: 0.75rem;
        font-size: 16px;
    }
    .stButton>button {
        background-color: #00FFFF;
        color: #000;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        cursor: pointer;
        transition: 0.3s;
        box-shadow: 0px 0px 10px #00FFFF;
    }
    .stButton>button:hover {
        background-color: #00FFAA;
        box-shadow: 0px 0px 20px #00FFAA;
    }
    .chat-container {
        max-width: 800px;
        margin: auto;
        padding: 1rem;
        border-radius: 10px;
    }
    .message {
        padding: 0.75rem;
        border-radius: 10px;
        margin-bottom: 10px;
        font-size: 16px;
        line-height: 1.5;
    }
    .user-msg {
        background-color: #1F1F1F;
        border: 1px solid #00FFFF;
        color: #00FFFF;
        text-align: right;
    }
    .bot-msg {
        background-color: #111;
        border-left: 4px solid #00FFFF;
        color: #FAFAFA;
        text-shadow: 0px 0px 5px #00FFFF;
        }
    </style>
""", unsafe_allow_html=True)

# --- Load models once ---
gemini_client, embedding_model, index, metadata = initialize_chatbot()

if gemini_client is None:
    st.stop()

# --- App title ---
st.markdown('<div class="header-title">🧑‍💻 NITW Q&A Desk</div>', unsafe_allow_html=True)

# --- Sidebar (Optional) ---
with st.sidebar:
    st.markdown('<span class="sidebar-title">Control Panel</span>', unsafe_allow_html=True)
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.input_value = ""
        st.rerun()
    st.markdown("---")
    st.markdown('<div class="sidebar-content">Model: <b>{}</b></div>'.format(EMBEDDING_MODEL), unsafe_allow_html=True)
    st.markdown('<div class="sidebar-content">Total exchanges: <b>{}</b></div>'.format(len(st.session_state.get('messages', []))), unsafe_allow_html=True)
    st.markdown('<div class="sidebar-content">RAG-powered chatbot for NIT Warangal.<br>Uses custom embeddings & Gemini AI.</div>', unsafe_allow_html=True)


# --- Initialize session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "input_value" not in st.session_state:
    st.session_state.input_value = ""

if "submitted" not in st.session_state:
    st.session_state.submitted = False

# --- Chat container ---
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f"<div class='message user-msg'><b>User:</b> {msg['content']}</div>",
            unsafe_allow_html=True
        )
    elif msg["role"] == "assistant":
        st.markdown(
            f"<div class='message bot-msg'><b>Chatbot:</b> {msg['content']}</div>",
            unsafe_allow_html=True
        )

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="footer-caption">© 2025 NITW Assistant | Powered by Gemini LLM</div>', unsafe_allow_html=True)


# --- Input Bar ---
st.markdown("---")

# Create a form for better Enter key handling
with st.form(key="chat_form", clear_on_submit=True):
    col1, col2 = st.columns([6, 1])
    with col1:
        user_input = st.text_input(
            "Ask something...", 
            key="current_input",
            placeholder="Type your question here...",
            label_visibility="collapsed"
        )
    with col2:
        ask_button = st.form_submit_button("Ask")

# --- Handle Input ---
if ask_button and user_input.strip():
    # Store the query
    query = user_input.strip()
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Get bot response
    with st.spinner("🔍 Thinking..."):
        bot_response = query_chatbot(
            query,
            st.session_state.messages[:-1],  # Exclude the just-added user message
            index,
            metadata,
            embedding_model,
            gemini_client
        )
    
    # Add bot message
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    
    # Rerun to display new messages and clear input
    st.rerun()