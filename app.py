
# ============================================================
# TRILINGUAL RAG BOT
# English + Tamil + Sinhala
# FAISS + SQLite + Ollama + Streamlit
# FULLY OFFLINE — LLM handles all translation
# ============================================================

import streamlit as st
import sqlite3
import numpy as np
import os
import pickle
import warnings
import faiss
import time
import json
import requests
from datetime import datetime
from collections import Counter

warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_XET"] = "1"

from sentence_transformers import SentenceTransformer
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0

# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {
    "db_path"          : "knowledge_base.db",
    "faiss_index_path" : "faiss_index.bin",
    "doc_store_path"   : "doc_store.pkl",
    "embedding_model"  : "intfloat/multilingual-e5-small",
    "ollama_model"     : "gemma3:12b",
    "ollama_base_url"  : "http://localhost:11434",   # REST endpoint
    "top_k"            : 3,
    "score_threshold"  : 0.3,
    "temperature"      : 0.1,
    "max_tokens"       : 512,
}

LANGUAGE_MAP = {
    "en": "English",
    "ta": "Tamil   (தமிழ்)",
    "si": "Sinhala (සිංහල)"
}

LANGUAGE_FULL = {
    "en": "English",
    "ta": "Tamil",
    "si": "Sinhala"
}

LANGUAGE_FLAGS = {
    "en": "🇬🇧",
    "ta": "🇮🇳",
    "si": "🇱🇰"
}

# Unicode ranges
SINHALA_START = 0x0D80
SINHALA_END   = 0x0DFF
TAMIL_START   = 0x0B80
TAMIL_END     = 0x0BFF

# ============================================================
# STREAMLIT PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Trilingual RAG Bot",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    .main { background-color: #0e1117; }

    .header-banner {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 30px; border-radius: 15px; text-align: center;
        margin-bottom: 25px; border: 1px solid #e94560;
        box-shadow: 0 4px 20px rgba(233,69,96,0.3);
    }
    .header-title {
        font-size: 2.5rem; font-weight: 800; color: #ffffff;
        margin: 0; letter-spacing: 2px;
    }
    .header-subtitle {
        font-size: 1rem; color: #a0aec0; margin-top: 8px; letter-spacing: 1px;
    }
    .lang-badges {
        margin-top: 15px; display: flex; justify-content: center;
        gap: 15px; flex-wrap: wrap;
    }
    .lang-badge {
        background: rgba(233,69,96,0.2); border: 1px solid #e94560;
        color: #e94560; padding: 5px 15px; border-radius: 20px;
        font-size: 0.85rem; font-weight: 600;
    }

    .chat-user {
        background: linear-gradient(135deg, #1e3a5f, #0f3460);
        border: 1px solid #2d6a9f; border-radius: 15px 15px 5px 15px;
        padding: 15px 20px; margin: 10px 0; color: #e2e8f0;
        max-width: 80%; margin-left: auto;
        box-shadow: 0 2px 10px rgba(45,106,159,0.3);
    }
    .chat-bot {
        background: linear-gradient(135deg, #1a2e1a, #1e3a1e);
        border: 1px solid #2d7a2d; border-radius: 15px 15px 15px 5px;
        padding: 15px 20px; margin: 10px 0; color: #e2e8f0;
        max-width: 80%; box-shadow: 0 2px 10px rgba(45,122,45,0.3);
    }
    .chat-label-user {
        font-size: 0.75rem; color: #63b3ed; font-weight: 700;
        margin-bottom: 6px; text-transform: uppercase; letter-spacing: 1px;
    }
    .chat-label-bot {
        font-size: 0.75rem; color: #68d391; font-weight: 700;
        margin-bottom: 6px; text-transform: uppercase; letter-spacing: 1px;
    }
    .chat-meta {
        font-size: 0.70rem; color: #718096; margin-top: 8px; text-align: right;
    }

    .retrieved-doc {
        background: #1a1a2e; border: 1px solid #2d3748;
        border-left: 4px solid #e94560; border-radius: 8px;
        padding: 12px 15px; margin: 8px 0; font-size: 0.85rem;
    }
    .doc-title { color: #e94560; font-weight: 700; font-size: 0.9rem; }
    .doc-category {
        background: rgba(233,69,96,0.15); color: #fc8181;
        padding: 2px 8px; border-radius: 10px;
        font-size: 0.75rem; font-weight: 600;
    }
    .doc-score { color: #68d391; font-weight: 700; }
    .doc-content { color: #a0aec0; margin-top: 6px; line-height: 1.5; }

    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #2d3748; border-radius: 12px;
        padding: 20px; text-align: center; margin: 5px 0;
    }
    .metric-value { font-size: 2rem; font-weight: 800; color: #e94560; }
    .metric-label {
        font-size: 0.8rem; color: #718096;
        text-transform: uppercase; letter-spacing: 1px; margin-top: 5px;
    }

    .status-ok    { color: #68d391; font-weight: 600; }
    .status-error { color: #fc8181; font-weight: 600; }
    .status-warn  { color: #f6ad55; font-weight: 600; }

    .lang-detect-box {
        background: linear-gradient(135deg, #1a2e1a, #2d3748);
        border: 1px solid #4a5568; border-radius: 10px;
        padding: 12px 15px; margin: 8px 0;
        font-size: 0.85rem; color: #a0aec0;
    }
    .pipeline-step {
        background: #1a1a2e; border: 1px solid #2d3748;
        border-radius: 8px; padding: 10px 15px; margin: 4px 0;
        font-size: 0.82rem; color: #a0aec0;
        display: flex; align-items: center; gap: 10px;
    }
    .unicode-box {
        background: #0d1117; border: 1px solid #30363d;
        border-radius: 8px; padding: 10px;
        font-family: monospace; font-size: 0.8rem; color: #8b949e;
    }
    .section-header {
        font-size: 1rem; font-weight: 700; color: #e94560;
        text-transform: uppercase; letter-spacing: 2px;
        margin: 20px 0 10px 0; padding-bottom: 5px;
        border-bottom: 1px solid #e94560;
    }
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #e94560, transparent);
        margin: 20px 0;
    }

    .stTextInput > div > div > input {
        background-color: #1a1a2e !important; border: 1px solid #4a5568 !important;
        color: #e2e8f0 !important; border-radius: 10px !important;
        padding: 12px !important; font-size: 1rem !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #e94560, #c0392b) !important;
        color: white !important; border: none !important;
        border-radius: 10px !important; padding: 10px 25px !important;
        font-weight: 700 !important; font-size: 0.9rem !important;
        transition: all 0.3s ease !important; width: 100% !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 15px rgba(233,69,96,0.4) !important;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ============================================================
# OLLAMA — REST-BASED HELPERS  (fixes "always offline" bug)
# ============================================================
# The `ollama` Python package's .list() call fails in many
# environments because it defaults to a different port or
# raises on unexpected response shapes.  We talk directly to
# the Ollama REST API instead, which is always reliable.

def ollama_get(path, timeout=3):
    """GET from Ollama REST API."""
    url = f"{CONFIG['ollama_base_url']}{path}"
    r   = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


def ollama_post(path, payload, timeout=120):
    """POST to Ollama REST API."""
    url = f"{CONFIG['ollama_base_url']}{path}"
    r   = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def check_ollama_status():
    """
    Check Ollama availability via REST.
    Returns (is_running: bool, model_names: list[str]).
    """
    try:
        data   = ollama_get("/api/tags", timeout=3)
        models = data.get("models", [])
        names  = [m.get("name", "") for m in models]
        return True, names
    except Exception:
        return False, []


def ollama_chat(model_name, messages, temperature=0.1, max_tokens=512):
    """
    Call Ollama /api/chat and return (content, error).
    Uses streaming=False for simplicity.
    """
    payload = {
        "model"   : model_name,
        "messages": messages,
        "stream"  : False,
        "options" : {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    try:
        data    = ollama_post("/api/chat", payload, timeout=180)
        content = data["message"]["content"]
        return content, None
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to Ollama. Make sure `ollama serve` is running."
    except requests.exceptions.Timeout:
        return None, "Ollama request timed out."
    except Exception as e:
        return None, str(e)


# ============================================================
# UNICODE LANGUAGE DETECTION
# ============================================================

def count_script_characters(text, start, end):
    return sum(1 for ch in text if start <= ord(ch) <= end)


def detect_language_unicode(text):
    clean_text = text.strip()
    if not clean_text:
        return "en", 0, 0, 0, 0.0, 0.0

    sinhala_count = count_script_characters(clean_text, SINHALA_START, SINHALA_END)
    tamil_count   = count_script_characters(clean_text, TAMIL_START,   TAMIL_END)
    total_chars   = len(clean_text.replace(" ", ""))

    sinhala_ratio = sinhala_count / total_chars if total_chars > 0 else 0.0
    tamil_ratio   = tamil_count   / total_chars if total_chars > 0 else 0.0

    THRESHOLD = 0.10
    if sinhala_ratio >= THRESHOLD:
        lang = "si"
    elif tamil_ratio >= THRESHOLD:
        lang = "ta"
    else:
        lang = "en"

    return lang, sinhala_count, tamil_count, total_chars, sinhala_ratio, tamil_ratio


def detect_language_langdetect(text):
    try:
        return detect(text)
    except Exception:
        return "en"


def detect_language(text):
    """
    Hybrid detection:
    1. Unicode scan  → catches Sinhala / Tamil perfectly
    2. langdetect    → English confirmation fallback
    Returns (lang_code, details_dict)
    """
    (unicode_lang,
     sinhala_count, tamil_count,
     total_chars,
     sinhala_ratio, tamil_ratio) = detect_language_unicode(text)

    details = {
        "method"         : "Unicode",
        "sinhala_chars"  : sinhala_count,
        "tamil_chars"    : tamil_count,
        "total_chars"    : total_chars,
        "sinhala_ratio"  : sinhala_ratio,
        "tamil_ratio"    : tamil_ratio,
        "unicode_result" : unicode_lang,
        "fallback_result": None,
        "final_lang"     : unicode_lang,
    }

    if unicode_lang in ["si", "ta"]:
        details["method"] = "Unicode (Primary)"
        return unicode_lang, details

    fallback = detect_language_langdetect(text)
    details["fallback_result"] = fallback
    details["method"]          = "langdetect (Fallback)"

    if fallback in ["en", "ta", "si"]:
        details["final_lang"] = fallback
        return fallback, details

    details["final_lang"] = "en"
    return "en", details


# ============================================================
# LLM-BASED TRANSLATION  (fully offline, no Google Translate)
# ============================================================

def translate_to_english_llm(text, source_lang, model_name):
    """
    Ask Ollama to translate Tamil/Sinhala text into English.
    Returns (translated_text, info_string).
    """
    if source_lang == "en":
        return text, "No translation needed"

    lang_name = LANGUAGE_FULL.get(source_lang, source_lang)

    system = (
        "You are a professional translator. "
        "Translate the user's text from "
        f"{lang_name} to English. "
        "Output ONLY the English translation — "
        "no explanations, no notes, no punctuation wrappers."
    )

    content, error = ollama_chat(
        model_name,
        [
            {"role": "system", "content": system},
            {"role": "user",   "content": text},
        ],
        temperature=0.1,
        max_tokens=256,
    )

    if error:
        # Graceful degradation: pass the original text and warn
        return text, f"Translation failed ({error}) — using original text"

    translated = content.strip()
    return translated, translated


def translate_response_llm(text, target_lang, model_name):
    """
    Ask Ollama to translate an English response into the target language.
    Returns translated string (falls back to English on error).
    """
    if target_lang == "en":
        return text

    lang_name = LANGUAGE_FULL.get(target_lang, target_lang)

    system = (
        "You are a professional translator. "
        f"Translate the following English text into {lang_name}. "
        "Output ONLY the translation — no extra commentary."
        "Avoid mixing languages."
    )

    content, error = ollama_chat(
        model_name,
        [
            {"role": "system", "content": system},
            {"role": "user",   "content": text},
        ],
        temperature=0.1,
        max_tokens=700,
    )

    if error:
        return text   # Fall back to English if translation fails

    return content.strip()


# ============================================================
# DATABASE SETUP
# ============================================================

def create_database():
    conn   = sqlite3.connect(CONFIG["db_path"])
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS knowledge (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL,
            title    TEXT NOT NULL,
            content  TEXT NOT NULL
        )
    """)

    cursor.execute("SELECT COUNT(*) FROM knowledge")
    count = cursor.fetchone()[0]

    if count == 0:
        sample_data = [
            ("health",
             "Dengue Fever Symptoms",
             "Dengue fever symptoms include high fever, severe headache, pain behind the eyes, "
             "joint and muscle pain, rash, and mild bleeding. Symptoms usually appear 4 to 10 "
             "days after infection and last for 2 to 7 days."),

            ("health",
             "Dengue Prevention",
             "To prevent dengue fever, eliminate standing water around your home, use mosquito "
             "repellent, wear long-sleeved clothing, use mosquito nets, and keep windows and "
             "doors closed or screened."),

            ("health",
             "Malaria Treatment",
             "Malaria treatment depends on the type of malaria parasite and severity. Common "
             "treatments include antimalarial medications such as chloroquine, artemisinin-based "
             "combination therapies, and supportive care."),

            ("education",
             "School Admission Process",
             "School admission requires birth certificate, vaccination records, previous school "
             "records if applicable, and parent identification documents. Applications must be "
             "submitted before the deadline."),

            ("education",
             "Scholarship Eligibility",
             "Scholarships are available for students with academic excellence, financial need, "
             "or special talents. Students must maintain a minimum GPA of 3.5 and submit "
             "applications by March 31 each year."),

            ("government",
             "National ID Card Application",
             "To apply for a national ID card, you need to visit the nearest divisional "
             "secretariat with your birth certificate, Grama Niladhari certificate, and two "
             "passport-size photographs."),

            ("government",
             "Passport Application Process",
             "Passport application requires national ID card, birth certificate, police clearance "
             "report, and completed application form. Processing takes 3 to 10 working days for "
             "normal service."),

            ("agriculture",
             "Paddy Cultivation Season",
             "Sri Lanka has two main paddy cultivation seasons. Maha season runs from October to "
             "March and Yala season runs from April to September. Farmers receive government "
             "subsidies for seeds and fertilizer."),

            ("agriculture",
             "Fertilizer Subsidy Program",
             "The government provides fertilizer subsidies to registered farmers. Farmers must "
             "register at the nearest agricultural office with land deed or lease agreement to "
             "receive subsidized fertilizer."),

            ("finance",
             "Bank Loan Application",
             "To apply for a bank loan, you need income proof, bank statements for last 6 months,"
             " national ID, and collateral documents if applicable. Loan approval typically takes "
             "5 to 7 working days."),

            ("finance",
             "Samurdhi Benefits",
             "Samurdhi is a government welfare program providing financial assistance to "
             "low-income families. Eligible families receive monthly allowances and can access "
             "low-interest loans through Samurdhi banks."),

            ("health",
             "COVID-19 Vaccination",
             "COVID-19 vaccines are available at government hospitals and health centers. Bring "
             "your national ID card for registration. Booster doses are recommended every 6 "
             "months for high-risk groups."),

            ("health",
             "Diabetes Management",
             "Diabetes management includes regular blood sugar monitoring, healthy diet, physical "
             "exercise, and medication as prescribed. Patients should avoid high-sugar foods and "
             "maintain a healthy weight."),

            ("government",
             "Driving License Application",
             "To apply for a driving license, visit the motor traffic department with national ID,"
             " medical certificate, and completed application form. You must pass written and "
             "practical driving tests."),
        ]

        cursor.executemany(
            "INSERT INTO knowledge (category, title, content) VALUES (?, ?, ?)",
            sample_data
        )
        conn.commit()

    conn.close()


def load_documents_from_db():
    conn   = sqlite3.connect(CONFIG["db_path"])
    cursor = conn.cursor()
    cursor.execute("SELECT id, category, title, content FROM knowledge")
    rows   = cursor.fetchall()
    conn.close()
    return [
        {
            "db_id"    : row[0],
            "category" : row[1],
            "title"    : row[2],
            "content"  : row[3],
            "full_text": f"passage: {row[2]}. {row[3]}"
        }
        for row in rows
    ]


def get_db_stats():
    conn   = sqlite3.connect(CONFIG["db_path"])
    cursor = conn.cursor()
    cursor.execute("SELECT category, COUNT(*) FROM knowledge GROUP BY category")
    cats   = cursor.fetchall()
    cursor.execute("SELECT COUNT(*) FROM knowledge")
    total  = cursor.fetchone()[0]
    conn.close()
    return total, dict(cats)


def get_all_documents_display():
    conn   = sqlite3.connect(CONFIG["db_path"])
    cursor = conn.cursor()
    cursor.execute("SELECT id, category, title, content FROM knowledge ORDER BY category, id")
    rows   = cursor.fetchall()
    conn.close()
    return rows


def add_document_to_db(category, title, content):
    conn   = sqlite3.connect(CONFIG["db_path"])
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO knowledge (category, title, content) VALUES (?, ?, ?)",
        (category, title, content)
    )
    new_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return new_id


def delete_document_from_db(doc_id):
    conn   = sqlite3.connect(CONFIG["db_path"])
    cursor = conn.cursor()
    cursor.execute("DELETE FROM knowledge WHERE id = ?", (doc_id,))
    conn.commit()
    conn.close()


# ============================================================
# EMBEDDING MODEL (cached)
# ============================================================

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer(CONFIG["embedding_model"])


# ============================================================
# FAISS INDEX
# ============================================================

def build_faiss_index(documents, model):
    texts      = [doc["full_text"] for doc in documents]
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=32,
        convert_to_numpy=True
    ).astype(np.float32)

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index, documents


def save_faiss_index(index, doc_store):
    faiss.write_index(index, CONFIG["faiss_index_path"])
    with open(CONFIG["doc_store_path"], "wb") as f:
        pickle.dump(doc_store, f)


def load_faiss_index_from_disk():
    if (os.path.exists(CONFIG["faiss_index_path"]) and
            os.path.exists(CONFIG["doc_store_path"])):
        index = faiss.read_index(CONFIG["faiss_index_path"])
        with open(CONFIG["doc_store_path"], "rb") as f:
            doc_store = pickle.load(f)
        return index, doc_store
    return None, None


@st.cache_resource(show_spinner=False)
def initialize_faiss(_model):
    create_database()
    index, doc_store = load_faiss_index_from_disk()
    if index is None:
        documents        = load_documents_from_db()
        index, doc_store = build_faiss_index(documents, _model)
        save_faiss_index(index, doc_store)
    return index, doc_store


def rebuild_index(_model):
    documents        = load_documents_from_db()
    index, doc_store = build_faiss_index(documents, _model)
    save_faiss_index(index, doc_store)
    return index, doc_store


# ============================================================
# FAISS RETRIEVAL
# ============================================================

def retrieve_with_faiss(query_english, model, index, doc_store,
                        top_k=3, score_threshold=0.3):
    query_emb = model.encode(
        [f"query: {query_english}"],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype(np.float32)

    distances, indices = index.search(query_emb, top_k)

    results = []
    for rank, (score, idx) in enumerate(zip(distances[0], indices[0])):
        if idx == -1:
            continue
        if score < score_threshold:
            continue
        results.append({
            "document": doc_store[idx],
            "score"   : float(score),
            "faiss_id": int(idx),
            "rank"    : rank + 1
        })
    return results


# ============================================================
# LLM — ANSWER GENERATION
# ============================================================

def format_context(retrieved_docs):
    parts = []
    for r in retrieved_docs:
        doc = r["document"]
        parts.append(
            f"[Source {r['rank']}]\n"
            f"Category : {doc['category'].upper()}\n"
            f"Title    : {doc['title']}\n"
            f"Content  : {doc['content']}\n"
            f"Score    : {r['score']:.4f}"
        )
    return "\n\n---\n\n".join(parts)


def get_ollama_response(english_query, context, model_name):
    """Generate a grounded English answer from retrieved context."""
    system_prompt = f"""You are a helpful assistant.
Answer the question based ONLY on the provided context.

Rules:
- Use ONLY the context below to answer.
- If the answer is not in the context, say exactly:
  "I don't have information about that."
- Be clear, concise and accurate.
- Always respond in ENGLISH.
- Do not make up information.

Context:
{context}
"""
    return ollama_chat(
        model_name,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"Question: {english_query}"},
        ],
        temperature=CONFIG["temperature"],
        max_tokens=CONFIG["max_tokens"],
    )


# ============================================================
# FULL RAG PIPELINE
# ============================================================

def run_rag_pipeline(user_query, model, faiss_index, doc_store,
                     top_k, score_threshold, ollama_model):
    result = {
        "query"           : user_query,
        "timestamp"       : datetime.now().strftime("%H:%M:%S"),
        "lang_code"       : "en",
        "lang_name"       : "English",
        "lang_details"    : {},
        "english_query"   : user_query,
        "translation_info": "",
        "retrieved_docs"  : [],
        "context"         : "",
        "english_response": "",
        "final_response"  : "",
        "error"           : None,
        "timings"         : {}
    }

    # ── Step 1: Language Detection ────────────────────────────
    t0                      = time.time()
    lang_code, lang_details = detect_language(user_query)
    result["lang_code"]     = lang_code
    result["lang_name"]     = LANGUAGE_MAP.get(lang_code, "English")
    result["lang_details"]  = lang_details
    result["timings"]["lang_detection"] = round(time.time() - t0, 3)

    # ── Step 2: Translate query → English (via LLM) ───────────
    t0                         = time.time()
    english_query, trans_info  = translate_to_english_llm(
        user_query, lang_code, ollama_model
    )
    result["english_query"]    = english_query
    result["translation_info"] = trans_info
    result["timings"]["translation"] = round(time.time() - t0, 3)

    # ── Step 3: FAISS Retrieval ───────────────────────────────
    t0                       = time.time()
    retrieved                = retrieve_with_faiss(
        english_query, model, faiss_index, doc_store,
        top_k=top_k, score_threshold=score_threshold
    )
    result["retrieved_docs"] = retrieved
    result["timings"]["retrieval"] = round(time.time() - t0, 3)

    if not retrieved:
        msg                      = "I don't have relevant information to answer your question."
        result["english_response"] = msg
        if lang_code != "en":
            t0 = time.time()
            result["final_response"] = translate_response_llm(msg, lang_code, ollama_model)
            result["timings"]["response_translation"] = round(time.time() - t0, 3)
        else:
            result["final_response"] = msg
        return result

    # ── Step 4: Format Context ────────────────────────────────
    result["context"] = format_context(retrieved)

    # ── Step 5: LLM Answer (English) ─────────────────────────
    t0                         = time.time()
    eng_response, error        = get_ollama_response(
        english_query, result["context"], ollama_model
    )
    result["timings"]["llm"]   = round(time.time() - t0, 3)

    if error:
        result["error"]          = error
        result["final_response"] = f"❌ Ollama Error: {error}"
        return result

    result["english_response"] = eng_response

    # ── Step 6: Translate Response → user language (via LLM) ─
    t0 = time.time()
    result["final_response"] = translate_response_llm(
        eng_response, lang_code, ollama_model
    )
    result["timings"]["response_translation"] = round(time.time() - t0, 3)

    return result


# ============================================================
# STREAMLIT SESSION STATE INIT
# ============================================================

def init_session_state():
    defaults = {
        "chat_history"  : [],
        "faiss_index"   : None,
        "doc_store"     : None,
        "last_result"   : None,
        "total_queries" : 0,
        "lang_counter"  : {"en": 0, "ta": 0, "si": 0},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar(model):
    with st.sidebar:

        st.markdown("""
        <div style='text-align:center; padding: 10px 0 20px 0;'>
            <div style='font-size:2.5rem;'>🌐</div>
            <div style='font-size:1.1rem; font-weight:800;
                        color:#e94560; letter-spacing:2px;'>RAG BOT</div>
            <div style='font-size:0.75rem; color:#718096;'>Trilingual Intelligence</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # ── System Status ─────────────────────────────────────
        st.markdown('<div class="section-header">⚙️ System Status</div>',
                    unsafe_allow_html=True)

        st.markdown(
            '<span class="status-ok">✅ Embedding Model Ready</span><br>'
            f'<small style="color:#718096;">{CONFIG["embedding_model"]}</small>',
            unsafe_allow_html=True
        )

        if st.session_state.faiss_index is not None:
            n = st.session_state.faiss_index.ntotal
            st.markdown(
                f'<span class="status-ok">✅ FAISS Index ({n} vectors)</span>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<span class="status-warn">⏳ Building FAISS Index...</span>',
                unsafe_allow_html=True
            )

        ollama_ok, ollama_models = check_ollama_status()
        if ollama_ok:
            st.markdown(
                '<span class="status-ok">✅ Ollama Running</span>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<span class="status-error">❌ Ollama Offline — run: ollama serve</span>',
                unsafe_allow_html=True
            )

        st.markdown(
            '<span style="color:#a0aec0; font-size:0.78rem;">'
            '🔒 Fully Offline — no Google Translate</span>',
            unsafe_allow_html=True
        )

        st.markdown("---")

        # ── Settings ──────────────────────────────────────────
        st.markdown('<div class="section-header">🎛️ Settings</div>',
                    unsafe_allow_html=True)

        if ollama_ok and ollama_models:
            selected_model = st.selectbox("🤖 Ollama Model", options=ollama_models, index=0)
        else:
            selected_model = st.text_input("🤖 Ollama Model", value=CONFIG["ollama_model"])

        top_k = st.slider(
            "📚 Retrieved Documents (Top-K)",
            min_value=1, max_value=6, value=CONFIG["top_k"],
            help="Number of documents to retrieve from FAISS"
        )

        score_threshold = st.slider(
            "🎯 Similarity Threshold",
            min_value=0.0, max_value=1.0,
            value=CONFIG["score_threshold"], step=0.05,
            help="Minimum cosine similarity score"
        )

        temperature = st.slider(
            "🌡️ LLM Temperature",
            min_value=0.0, max_value=1.0,
            value=CONFIG["temperature"], step=0.05,
            help="Lower = more factual"
        )
        CONFIG["temperature"] = temperature

        st.markdown("---")

        # ── Language Stats ────────────────────────────────────
        st.markdown('<div class="section-header">🌐 Language Stats</div>',
                    unsafe_allow_html=True)

        lc   = st.session_state.lang_counter
        cols = st.columns(3)
        with cols[0]: st.metric("🇬🇧 EN", lc.get("en", 0))
        with cols[1]: st.metric("🇮🇳 TA", lc.get("ta", 0))
        with cols[2]: st.metric("🇱🇰 SI", lc.get("si", 0))
        st.caption(f"Total queries: {st.session_state.total_queries}")

        st.markdown("---")

        # ── Database Stats ────────────────────────────────────
        st.markdown('<div class="section-header">🗄️ Database</div>',
                    unsafe_allow_html=True)

        total_docs, cat_stats = get_db_stats()
        st.caption(f"Total documents: {total_docs}")

        for cat, cnt in sorted(cat_stats.items()):
            emoji = {
                "health": "🏥", "education": "🎓",
                "government": "🏛️", "agriculture": "🌾", "finance": "💰"
            }.get(cat, "📄")
            st.markdown(
                f'<div style="display:flex; justify-content:space-between; '
                f'font-size:0.82rem; color:#a0aec0; padding:2px 0;">'
                f'<span>{emoji} {cat.capitalize()}</span>'
                f'<span style="color:#e94560; font-weight:700;">{cnt}</span></div>',
                unsafe_allow_html=True
            )

        st.markdown("---")

        # ── Maintenance ───────────────────────────────────────
        st.markdown('<div class="section-header">🔧 Maintenance</div>',
                    unsafe_allow_html=True)

        if st.button("🔄 Rebuild FAISS Index"):
            with st.spinner("Rebuilding..."):
                idx, ds = rebuild_index(model)
                st.session_state.faiss_index = idx
                st.session_state.doc_store   = ds
                initialize_faiss.clear()
            st.success(f"✅ Index rebuilt! ({idx.ntotal} vectors)")
            st.rerun()

        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history  = []
            st.session_state.last_result   = None
            st.session_state.total_queries = 0
            st.session_state.lang_counter  = {"en": 0, "ta": 0, "si": 0}
            st.rerun()

        st.markdown("---")
        st.caption("🔒 Fully Local | No Cloud API")
        st.caption("FAISS + SQLite + Ollama")

    return selected_model, top_k, score_threshold


# ============================================================
# CHAT TAB
# ============================================================

def render_chat_tab(model, faiss_index, doc_store,
                    top_k, score_threshold, ollama_model):

    # ── Example Queries ───────────────────────────────────────
    st.markdown('<div class="section-header">💡 Example Queries</div>',
                unsafe_allow_html=True)

    examples = {
        "🇬🇧 English": [
            "What are the symptoms of dengue fever?",
            "How to apply for a passport?",
            "What is Samurdhi benefit program?",
        ],
        "🇮🇳 Tamil": [
            "டெங்கு காய்ச்சலின் அறிகுறிகள் என்ன?",
            "வங்கி கடன் விண்ணப்பிக்க என்ன தேவை?",
            "கல்வி உதவித்தொகை எப்படி பெறுவது?",
        ],
        "🇱🇰 Sinhala": [
            "ඩෙංගු රෝග ලක්ෂණ මොනවාද?",
            "ජාතික හැඳුනුම්පත ඉල්ලුම් කරන්නේ කෙසේද?",
            "පොහොර සහනාධාරය ලබා ගන්නේ කෙසේද?",
        ],
    }

    ex_cols = st.columns(3)
    example_clicked = None

    for col, (lang_label, queries) in zip(ex_cols, examples.items()):
        with col:
            st.caption(lang_label)
            for q in queries:
                label = q[:35] + "..." if len(q) > 35 else q
                if st.button(label, key=f"ex_{q[:20]}", use_container_width=True):
                    example_clicked = q

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ── Chat History ──────────────────────────────────────────
    st.markdown('<div class="section-header">💬 Conversation</div>',
                unsafe_allow_html=True)

    if not st.session_state.chat_history:
        st.markdown("""
        <div style='text-align:center; padding:40px; color:#4a5568;'>
            <div style='font-size:3rem;'>🌐</div>
            <div style='font-size:1.1rem; margin-top:10px;'>
                Ask anything in English, Tamil, or Sinhala
            </div>
            <div style='font-size:0.85rem; margin-top:5px; color:#2d3748;'>
                ඕනෑම භාෂාවෙන් අසන්න | எந்த மொழியிலும் கேளுங்கள்
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for chat in st.session_state.chat_history:
            flag = LANGUAGE_FLAGS.get(chat.get("lang_code", "en"), "🌐")
            total_time = sum(chat.get("timings", {}).values())

            st.markdown(f"""
            <div class="chat-user">
                <div class="chat-label-user">{flag} You · {chat.get('timestamp','')}</div>
                <div>{chat['query']}</div>
                <div class="chat-meta">{LANGUAGE_MAP.get(chat.get('lang_code','en'), 'English')}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="chat-bot">
                <div class="chat-label-bot">🤖 RAG Bot · {chat.get('timestamp','')}</div>
                <div>{chat['response']}</div>
                <div class="chat-meta">
                    ⏱️ {total_time:.2f}s |
                    📚 {len(chat.get('retrieved_docs', []))} docs retrieved
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ── Input Area ────────────────────────────────────────────
    input_col, btn_col = st.columns([5, 1])

    with input_col:
        default_val = example_clicked if example_clicked else ""
        user_input  = st.text_input(
            "Your question",
            value=default_val,
            placeholder="Type in English, Tamil (தமிழ்) or Sinhala (සිංහල)...",
            label_visibility="collapsed",
            key="main_input"
        )

    with btn_col:
        send_clicked = st.button("Send 🚀", use_container_width=True)

    # ── Process ───────────────────────────────────────────────
    query_to_run = None
    if send_clicked and user_input.strip():
        query_to_run = user_input.strip()
    elif example_clicked:
        query_to_run = example_clicked

    if query_to_run:
        with st.spinner("🔍 Processing your query..."):
            result = run_rag_pipeline(
                query_to_run, model, faiss_index, doc_store,
                top_k, score_threshold, ollama_model
            )

        st.session_state.chat_history.append({
            "query"         : query_to_run,
            "response"      : result["final_response"],
            "lang_code"     : result["lang_code"],
            "timestamp"     : result["timestamp"],
            "retrieved_docs": result["retrieved_docs"],
            "timings"       : result["timings"],
        })

        st.session_state.total_queries += 1
        lc = st.session_state.lang_counter
        lc[result["lang_code"]] = lc.get(result["lang_code"], 0) + 1
        st.session_state.last_result = result
        st.rerun()


# ============================================================
# PIPELINE INSPECTOR TAB
# ============================================================

def render_inspector_tab():
    st.markdown('<div class="section-header">🔬 Pipeline Inspector</div>',
                unsafe_allow_html=True)

    result = st.session_state.last_result
    if result is None:
        st.info("💡 Run a query first to inspect the pipeline.")
        return

    flag      = LANGUAGE_FLAGS.get(result["lang_code"], "🌐")
    lang_name = LANGUAGE_MAP.get(result["lang_code"], "English")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card">'
                    f'<div class="metric-value">{flag}</div>'
                    f'<div class="metric-label">{lang_name.split()[0]}</div></div>',
                    unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card">'
                    f'<div class="metric-value">{len(result["retrieved_docs"])}</div>'
                    f'<div class="metric-label">Docs Retrieved</div></div>',
                    unsafe_allow_html=True)
    with c3:
        total_t = sum(result["timings"].values())
        st.markdown(f'<div class="metric-card">'
                    f'<div class="metric-value">{total_t:.2f}s</div>'
                    f'<div class="metric-label">Total Time</div></div>',
                    unsafe_allow_html=True)
    with c4:
        top_score = (max(r["score"] for r in result["retrieved_docs"])
                     if result["retrieved_docs"] else 0)
        st.markdown(f'<div class="metric-card">'
                    f'<div class="metric-value">{top_score:.3f}</div>'
                    f'<div class="metric-label">Top Score</div></div>',
                    unsafe_allow_html=True)

    st.markdown("---")
    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown("#### 🔍 Language Detection")
        ld = result["lang_details"]
        st.markdown(f"""
        <div class="lang-detect-box">
            <b>Method:</b> {ld.get('method', 'N/A')}<br>
            <b>Final Language:</b> {flag} {lang_name}<br>
            <b>Unicode Result:</b> {ld.get('unicode_result', 'N/A')}<br>
            <b>Langdetect Result:</b> {ld.get('fallback_result', 'N/A')}
        </div>""", unsafe_allow_html=True)

        st.markdown("**📊 Unicode Analysis**")
        st.markdown(f"""
        <div class="unicode-box">
            Total chars   : {ld.get('total_chars', 0)}<br>
            Sinhala chars : {ld.get('sinhala_chars', 0)}
                ({ld.get('sinhala_ratio', 0):.1%}) [U+0D80–U+0DFF]<br>
            Tamil chars   : {ld.get('tamil_chars', 0)}
                ({ld.get('tamil_ratio', 0):.1%}) [U+0B80–U+0BFF]<br>
            Threshold     : 10%
        </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 🔄 Translation (via Ollama LLM)")
        if result["lang_code"] == "en":
            st.success("No translation needed (English query)")
        else:
            st.markdown(f"""
            <div class="lang-detect-box">
                <b>Original:</b> {result['query']}<br><br>
                <b>→ English:</b> {result['english_query']}
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### ⏱️ Timing Breakdown")
        timing_labels = {
            "lang_detection"       : "Language Detection",
            "translation"          : "Query Translation (LLM)",
            "retrieval"            : "FAISS Retrieval",
            "llm"                  : "LLM Generation",
            "response_translation" : "Response Translation (LLM)"
        }
        for key, label in timing_labels.items():
            t = result["timings"].get(key, 0)
            if t > 0:
                st.markdown(
                    f'<div class="pipeline-step">'
                    f'<span>⏱️</span><span style="flex:1">{label}</span>'
                    f'<span style="color:#68d391; font-weight:700;">{t:.3f}s</span>'
                    f'</div>', unsafe_allow_html=True)

    with right_col:
        st.markdown("#### 📚 Retrieved Documents (FAISS)")
        if not result["retrieved_docs"]:
            st.warning("No documents retrieved above threshold.")
        else:
            for r in result["retrieved_docs"]:
                doc   = r["document"]
                score = r["score"]
                color = "#68d391" if score > 0.6 else \
                        "#f6ad55" if score > 0.4 else "#fc8181"
                cat_emoji = {
                    "health": "🏥", "education": "🎓",
                    "government": "🏛️", "agriculture": "🌾", "finance": "💰"
                }.get(doc["category"], "📄")
                st.markdown(f"""
                <div class="retrieved-doc">
                    <div style="display:flex; justify-content:space-between;
                                align-items:center; margin-bottom:6px;">
                        <span class="doc-title">{cat_emoji} {doc['title']}</span>
                        <span class="doc-score" style="color:{color};">{score:.4f}</span>
                    </div>
                    <span class="doc-category">{doc['category'].upper()}</span>
                    <span style="color:#718096; font-size:0.75rem; margin-left:8px;">
                        FAISS ID: {r['faiss_id']}
                    </span>
                    <div class="doc-content">
                        {doc['content'][:200]}{'...' if len(doc['content']) > 200 else ''}
                    </div>
                </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 🤖 LLM Response (English)")
        if result.get("english_response"):
            st.markdown(f'<div class="lang-detect-box">{result["english_response"]}</div>',
                        unsafe_allow_html=True)

        if result["lang_code"] != "en":
            st.markdown(f"#### {flag} Final Response ({lang_name.split()[0]})")
            st.markdown(
                f'<div class="lang-detect-box" style="border-color:#68d391;">'
                f'{result["final_response"]}</div>',
                unsafe_allow_html=True
            )


# ============================================================
# KNOWLEDGE BASE TAB
# ============================================================

def render_knowledge_tab(model):
    st.markdown('<div class="section-header">🗄️ Knowledge Base Management</div>',
                unsafe_allow_html=True)

    tab_view, tab_add = st.tabs(["📋 View Documents", "➕ Add Document"])

    with tab_view:
        rows = get_all_documents_display()
        total_docs, _ = get_db_stats()

        all_cats    = sorted(set(r[1] for r in rows))
        filter_cats = st.multiselect("Filter by Category", options=all_cats,
                                     default=all_cats, key="cat_filter")
        filtered = [r for r in rows if r[1] in filter_cats]
        st.caption(f"Showing {len(filtered)} of {total_docs} documents")

        for row in filtered:
            doc_id, category, title, content = row
            cat_emoji = {
                "health": "🏥", "education": "🎓",
                "government": "🏛️", "agriculture": "🌾", "finance": "💰"
            }.get(category, "📄")

            with st.expander(f"{cat_emoji} [{doc_id}] {title}"):
                st.markdown(f"**Category:** `{category.upper()}`")
                st.markdown(f"**Content:** {content}")

                if st.button(f"🗑️ Delete Document #{doc_id}",
                             key=f"del_{doc_id}", type="secondary"):
                    delete_document_from_db(doc_id)
                    with st.spinner("Rebuilding FAISS index after deletion..."):
                        new_idx, new_ds = rebuild_index(model)
                        st.session_state.faiss_index = new_idx
                        st.session_state.doc_store   = new_ds
                        initialize_faiss.clear()
                    st.success(f"✅ Document #{doc_id} deleted & index rebuilt!")
                    st.rerun()

    with tab_add:
        st.markdown("#### ➕ Add New English Document")
        st.info("📌 All documents must be in **English**. "
                "The bot translates queries to English before searching.")

        with st.form("add_doc_form", clear_on_submit=True):
            new_category = st.selectbox(
                "Category",
                options=["health", "education", "government",
                         "agriculture", "finance", "other"]
            )
            new_title   = st.text_input("Title", placeholder="e.g., Typhoid Fever Symptoms")
            new_content = st.text_area("Content (English)",
                                       placeholder="Enter document content in English...",
                                       height=150)
            submitted = st.form_submit_button("➕ Add Document", type="primary")

            if submitted:
                if not new_title.strip() or not new_content.strip():
                    st.error("❌ Title and Content are required.")
                else:
                    new_id  = add_document_to_db(
                        new_category.strip(), new_title.strip(), new_content.strip()
                    )
                    new_doc = {
                        "db_id"    : new_id,
                        "category" : new_category,
                        "title"    : new_title.strip(),
                        "content"  : new_content.strip(),
                        "full_text": f"passage: {new_title.strip()}. {new_content.strip()}"
                    }
                    new_emb = model.encode(
                        [new_doc["full_text"]],
                        normalize_embeddings=True,
                        convert_to_numpy=True
                    ).astype(np.float32)

                    st.session_state.faiss_index.add(new_emb)
                    st.session_state.doc_store.append(new_doc)
                    save_faiss_index(st.session_state.faiss_index, st.session_state.doc_store)

                    st.success(
                        f"✅ Document added! "
                        f"FAISS now has {st.session_state.faiss_index.ntotal} vectors."
                    )
                    st.rerun()


# ============================================================
# ABOUT TAB
# ============================================================

def render_about_tab():
    st.markdown('<div class="section-header">ℹ️ About This System</div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        #### 🏗️ Architecture
        ```
        User Query (EN / TA / SI)
               ↓
        Unicode Language Detection
          [Sinhala: U+0D80–U+0DFF]
          [Tamil  : U+0B80–U+0BFF]
               ↓
        Ollama LLM → Translate to English
          (fully offline, no Google Translate)
               ↓
        multilingual-e5-small
          (384-dim embedding)
               ↓
        FAISS IndexFlatIP
          (Cosine Similarity)
               ↓
        Top-K Documents Retrieved
               ↓
        Ollama LLM — Grounded Answer (EN)
               ↓
        Ollama LLM → Translate to User Lang
               ↓
        Final Response
        ```
        """)

    with c2:
        st.markdown("""
        #### 🛠️ Tech Stack

        | Component | Technology |
        |-----------|-----------|
        | Vector DB | FAISS IndexFlatIP |
        | Metadata DB | SQLite |
        | Embeddings | multilingual-e5-small |
        | Lang Detection | Unicode + langdetect |
        | Translation | Ollama LLM (offline) |
        | LLM | Ollama REST API |
        | UI | Streamlit |

        #### 🌐 Language Support

        | Language | Script | Unicode Range |
        |----------|--------|--------------|
        | English | Latin | U+0041–U+007A |
        | Tamil | Tamil | U+0B80–U+0BFF |
        | Sinhala | Sinhala | U+0D80–U+0DFF |

        #### 🔑 Key Features
        - ✅ Fully offline (zero cloud APIs)
        - ✅ LLM-powered translation (no Google Translate)
        - ✅ Unicode-based Sinhala/Tamil detection
        - ✅ FAISS exact vector search
        - ✅ Incremental document addition
        - ✅ Real-time pipeline inspection
        """)

    st.markdown("---")
    st.markdown("""
    #### ⚠️ Important Notes

    1. **All translation is done by Ollama LLM** → no internet required
    2. **DB content is English only** → queries are translated before searching
    3. **Unicode detection is priority** → langdetect only used as English fallback
    4. **FAISS IndexFlatIP** → exact search, supports incremental add
    5. **Deletion requires rebuild** → FAISS does not support vector deletion
    6. **Ollama status uses REST API** → `GET /api/tags` on localhost:11434
    7. **Run `ollama serve` before starting** if Ollama shows offline
    """)


# ============================================================
# MAIN
# ============================================================

def main():
    init_session_state()

    st.markdown("""
    <div class="header-banner">
        <div class="header-title">🌐 TRILINGUAL RAG BOT</div>
        <div class="header-subtitle">
            English · Tamil · Sinhala &nbsp;|&nbsp;
            FAISS + SQLite + Ollama &nbsp;|&nbsp; 🔒 Fully Offline
        </div>
        <div class="lang-badges">
            <span class="lang-badge">🇬🇧 English</span>
            <span class="lang-badge">🇮🇳 தமிழ் Tamil</span>
            <span class="lang-badge">🇱🇰 සිංහල Sinhala</span>
            <span class="lang-badge">🔒 No Cloud APIs</span>
            <span class="lang-badge">⚡ FAISS Vector Search</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("⏳ Loading multilingual embedding model..."):
        model = load_embedding_model()

    if st.session_state.faiss_index is None:
        with st.spinner("⏳ Initializing FAISS index..."):
            idx, ds = initialize_faiss(model)
            st.session_state.faiss_index = idx
            st.session_state.doc_store   = ds

    selected_model, top_k, score_threshold = render_sidebar(model)

    tab_chat, tab_inspect, tab_kb, tab_about = st.tabs([
        "💬 Chat",
        "🔬 Pipeline Inspector",
        "🗄️ Knowledge Base",
        "ℹ️ About"
    ])

    with tab_chat:
        render_chat_tab(
            model, st.session_state.faiss_index,
            st.session_state.doc_store,
            top_k, score_threshold, selected_model
        )
    with tab_inspect:
        render_inspector_tab()
    with tab_kb:
        render_knowledge_tab(model)
    with tab_about:
        render_about_tab()


if __name__ == "__main__":
    main()
