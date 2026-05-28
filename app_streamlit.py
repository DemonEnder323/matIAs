import streamlit as st
import base64
import os
import threading
import numpy as np
import requests
import fitz  # PyMuPDF
from openai import OpenAI

# ─────────────────────────────────────────────
# Configuración de página
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Online System",
    page_icon="🧠",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@700;800&display=swap');

html, body, [class*="css"] {
    background-color: #0f0f0f;
    color: #e8e8e8;
    font-family: 'JetBrains Mono', monospace;
}
h1 { font-family: 'Syne', sans-serif; font-size: 2.2rem; color: #b8ff57; letter-spacing: -1px; }
.stTextInput > div > div > input {
    background: #1a1a1a;
    color: #e8e8e8;
    border: 1px solid #333;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
}
.stButton > button {
    background: #b8ff57;
    color: #0f0f0f;
    font-weight: 700;
    border: none;
    border-radius: 6px;
    padding: 0.5rem 1.5rem;
    font-family: 'Syne', sans-serif;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }
.stFileUploader { background: #1a1a1a; border-radius: 8px; border: 1px dashed #333; }
.status-box {
    background: #1a1a1a;
    border-left: 3px solid #b8ff57;
    padding: 0.5rem 1rem;
    border-radius: 0 6px 6px 0;
    margin: 0.5rem 0;
    font-size: 0.85rem;
    color: #aaa;
}
.respuesta-box {
    background: #141414;
    border: 1px solid #222;
    border-radius: 10px;
    padding: 1.5rem;
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Estado de sesión
# ─────────────────────────────────────────────
if "vectores_documento" not in st.session_state:
    st.session_state.vectores_documento = []
if "nombre_pdf" not in st.session_state:
    st.session_state.nombre_pdf = None
if "historial" not in st.session_state:
    st.session_state.historial = []

# ─────────────────────────────────────────────
# Sidebar: Claves API
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    openai_key = st.text_input("OpenAI / Abacus API Key", type="password",
                               value=os.getenv("OPENAI_API_KEY", ""))
    jina_key   = st.text_input("Jina AI API Key (RAG)",  type="password",
                               value=os.getenv("JINA_API_KEY", ""))
    base_url   = st.text_input("Base URL", value="https://routellm.abacus.ai/v1")
    modelo_default = st.selectbox("Modelo texto", ["deepseek-v3.2", "gpt-4o", "gpt-4o-mini"])
    st.divider()
    st.caption("📄 Sube un PDF para activar RAG\n🖼 Sube imagen para visión")

# ─────────────────────────────────────────────
# Clientes
# ─────────────────────────────────────────────
@st.cache_resource
def get_client(api_key, url):
    return OpenAI(base_url=url, api_key=api_key)

# ─────────────────────────────────────────────
# Embeddings Jina
# ─────────────────────────────────────────────
def obtener_embedding(texto, key):
    try:
        r = requests.post(
            "https://api.jina.ai/v1/embeddings",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": "jina-embeddings-v3", "input": [texto.replace("\n", " ")]},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["data"][0]["embedding"]
    except Exception as e:
        st.error(f"Error embeddings Jina: {e}")
        return None

def dividir_texto(texto, chunk_size=1000):
    palabras = texto.split()
    return [" ".join(palabras[i:i+chunk_size]) for i in range(0, len(palabras), chunk_size - 200)]

def indexar_pdf(archivo_bytes, jina_key):
    doc = fitz.open(stream=archivo_bytes, filetype="pdf")
    texto = "\n".join(p.get_text() for p in doc)
    doc.close()
    if not texto.strip():
        return []
    chunks = dividir_texto(texto)
    vectores = []
    prog = st.progress(0, text="Generando embeddings…")
    for i, chunk in enumerate(chunks):
        v = obtener_embedding(chunk, jina_key)
        if v:
            vectores.append({"texto": chunk, "vector": v})
        prog.progress((i+1)/len(chunks), text=f"Chunk {i+1}/{len(chunks)}")
    prog.empty()
    return vectores

def buscar_contexto(pregunta, vectores, jina_key, top_k=4):
    if not vectores: return ""
    v_preg = obtener_embedding(pregunta, jina_key)
    if not v_preg: return ""
    vp = np.array(v_preg)
    resultados = []
    for item in vectores:
        vd = np.array(item["vector"])
        sim = np.dot(vp, vd) / (np.linalg.norm(vp) * np.linalg.norm(vd))
        resultados.append((item["texto"], sim))
    resultados.sort(key=lambda x: x[1], reverse=True)
    return "\n\n---\n\n".join(r[0] for r in resultados[:top_k])

# ─────────────────────────────────────────────
# Layout principal
# ─────────────────────────────────────────────
st.markdown("# 🧠 RAG ONLINE SYSTEM")
st.markdown("<div class='status-box'>Asistente IA con soporte PDF (RAG), imágenes y LaTeX</div>",
            unsafe_allow_html=True)

col_izq, col_der = st.columns([1, 2])

with col_izq:
    st.markdown("### 📂 Archivos")
    archivo = st.file_uploader("PDF o Imagen", type=["pdf", "png", "jpg", "jpeg"])

    imagen_b64 = None
    if archivo:
        ext = os.path.splitext(archivo.name)[1].lower()
        if ext == ".pdf":
            if jina_key:
                if st.button("🔍 Indexar PDF"):
                    with st.spinner("Indexando…"):
                        st.session_state.vectores_documento = indexar_pdf(archivo.read(), jina_key)
                        st.session_state.nombre_pdf = archivo.name
                    st.success(f"✅ RAG listo: {archivo.name}")
            else:
                st.warning("Ingresa tu Jina API Key en el sidebar para usar RAG.")
        elif ext in {".png", ".jpg", ".jpeg"}:
            imagen_b64 = base64.b64encode(archivo.read()).decode("utf-8")
            st.image(archivo, caption=archivo.name, use_container_width=True)

    if st.session_state.nombre_pdf:
        st.markdown(f"<div class='status-box'>📄 PDF indexado: <b>{st.session_state.nombre_pdf}</b><br>{len(st.session_state.vectores_documento)} fragmentos</div>",
                    unsafe_allow_html=True)

with col_der:
    st.markdown("### 💬 Pregunta")
    pregunta = st.text_input("", placeholder="Escribe tu pregunta o problema matemático…")

    if st.button("⚡ Resolver"):
        if not pregunta.strip() and not imagen_b64:
            st.warning("Escribe una pregunta o sube una imagen.")
        elif not openai_key:
            st.error("Ingresa tu API Key en el sidebar.")
        else:
            client = get_client(openai_key, base_url)
            modelo = modelo_default
            user_content = None

            if imagen_b64:
                modelo = "gpt-4o-mini"
                user_content = [
                    {"type": "text", "text": f"Resuelve: {pregunta}\nUsa LaTeX con $ $."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{imagen_b64}"}}
                ]
            elif st.session_state.vectores_documento:
                with st.spinner("🔍 Buscando en documento…"):
                    contexto = buscar_contexto(pregunta, st.session_state.vectores_documento, jina_key)
                user_content = f"CONTEXTO:\n{contexto}\n\nPREGUNTA:\n{pregunta}\n\nResponde usando LaTeX con $ $."
            else:
                user_content = f"{pregunta}\nUsa LaTeX con $ $."

            st.markdown("---")
            contenedor = st.empty()
            resultado = ""

            try:
                stream = client.chat.completions.create(
                    model=modelo,
                    messages=[
                        {"role": "system", "content": "Eres un asistente experto. Responde en Markdown y usa LaTeX ($ $) para matemáticas."},
                        {"role": "user", "content": user_content}
                    ],
                    stream=True,
                )
                for event in stream:
                    delta = event.choices[0].delta.content
                    if delta:
                        resultado += delta
                        contenedor.markdown(
                            f"<div class='respuesta-box'>{resultado}</div>",
                            unsafe_allow_html=True
                        )
                st.success("✅ Completado")
                st.session_state.historial.append({"pregunta": pregunta, "respuesta": resultado})
            except Exception as e:
                st.error(f"❌ Error: {e}")

# ─────────────────────────────────────────────
# Historial
# ─────────────────────────────────────────────
if st.session_state.historial:
    with st.expander(f"🕓 Historial ({len(st.session_state.historial)} entradas)"):
        for i, item in enumerate(reversed(st.session_state.historial)):
            st.markdown(f"**Pregunta {len(st.session_state.historial)-i}:** {item['pregunta']}")
            st.markdown(item["respuesta"])
            st.divider()
