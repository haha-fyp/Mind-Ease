import streamlit as st
from groq import Groq
from supabase import create_client
from sentence_transformers import SentenceTransformer

# ================= CONFIG =================
GROQ_API_KEY = "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"
SUPABASE_URL = "https://eiwdlivdhoiwzmfxzvzx.supabase.co"
SUPABASE_SERVICE_KEY = "sb_secret_siw1sPLl7Ldrkt09nolJ7g_3JKqO-hL"
# =========================================

# Initialize clients (cached)5
@st.cache_resource
def load_clients():
    groq_client = Groq(api_key=GROQ_API_KEY)
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return groq_client, supabase, embedder

client, supabase, embedder = load_clients()

# RAG retrieval
def retrieve_context(query):
    query_embedding = embedder.encode(query).tolist()

    response = supabase.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_threshold": 0.7,
            "match_count": 5
        }
    ).execute()

    if response.data:
        return "\n\n".join([item["content"] for item in response.data])
    return ""

# UI
st.set_page_config(page_title="Mind Ease", page_icon="🧠")
st.title("🧠 Mind Ease")
st.caption("A Mental Health Support Chatbot (RAG-based)")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("How are you feeling today?")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    context = retrieve_context(user_input)

    messages = st.session_state.messages.copy()

    if context:
        messages.append({
            "role": "system",
            "content": f"""
Answer ONLY using the context below.
If the answer is not present, say "I don't know".

Context:
{context}
"""
        })

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.7,
        max_tokens=1024
    )

    reply = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": reply})

    with st.chat_message("assistant"):
        st.markdown(reply)
