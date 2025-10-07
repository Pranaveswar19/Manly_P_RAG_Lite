import streamlit as st
from pathlib import Path
from config import CONFIG

from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_INDEX_DIR = DATA_DIR / "index"

st.set_page_config(page_title="Manly P. Hall RAG", layout="wide")
st.title("📜 Ask Manly P. Hall")
st.caption("RAG over Manly P. Hall’s works with FAISS + reranking.")

st.sidebar.header("Settings")
top_k = st.sidebar.slider("Top-K retrieved chunks", 5, 40, CONFIG.default_k)
top_n = st.sidebar.slider("Top-N after rerank", 3, 15, CONFIG.default_top_n)
show_sources = st.sidebar.checkbox("Show sources", value=True)

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
Settings.llm = OpenAI(model=CONFIG.default_llm_model, temperature=0)

@st.cache_resource(show_spinner=False)
def _load_index(persist_dir: str):
    vector_store = FaissVectorStore.from_persist_dir(persist_dir=persist_dir)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, persist_dir=persist_dir
    )
    index = load_index_from_storage(storage_context)
    return index, vector_store

persist_dir = CONFIG.default_persist_dir or str(DEFAULT_INDEX_DIR)

colA, colB = st.sidebar.columns(2)
with colA:
    if st.button("Reload index", use_container_width=True):
        st.cache_resource.clear()

index, vector_store = _load_index(persist_dir)

probe_vec = Settings.embed_model.get_text_embedding("probe")
index_dim = getattr(vector_store._faiss_index, "d", None)

st.sidebar.markdown(
    f"**Index dim:** {index_dim}  \n"
    f"**Embed dim:** {len(probe_vec)}  \n"
    f"**Embedder:** `{EMBED_MODEL_NAME}`"
)

if index_dim is None:
    st.error("Could not read FAISS index dimension. Check your persisted index.")
    st.stop()

if len(probe_vec) != index_dim:
    st.error(
        "❌ Embedding dimension mismatch.\n\n"
        f"- FAISS index expects **{index_dim}**\n"
        f"- Current model `{EMBED_MODEL_NAME}` outputs **{len(probe_vec)}**\n\n"
        "Fix: Rebuild the index with this model or change the model to match the index."
    )
    st.stop()

question = st.text_area(
    "Ask a question:",
    placeholder="e.g., What is the symbolic meaning of the number 33 in Freemasonry?",
)
ask = st.button("Ask")

if ask and question.strip():
    with st.spinner("Retrieving and generating answer..."):
        retriever = index.as_retriever(similarity_top_k=top_k)

        reranker = SentenceTransformerRerank(
            top_n=top_n, model="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

        qe = RetrieverQueryEngine.from_args(
            retriever=retriever,
            node_postprocessors=[reranker],
            response_mode="compact",
        )

        qdim = len(Settings.embed_model.get_text_embedding(question))
        if qdim != index_dim:
            st.error(
                f"Live query embedding dim {qdim} != index dim {index_dim}. "
                "This indicates a different embedder is being used at query time."
            )
            st.stop()

        response = qe.query(question)

    st.markdown("### 🧠 Answer")
    st.write(str(response).strip())

    if show_sources and hasattr(response, "source_nodes"):
        st.markdown("### 📚 Sources")
        for i, sn in enumerate(response.source_nodes, start=1):
            meta = sn.node.metadata or {}
            src = meta.get("file_path") or meta.get("file_name") or "unknown"
            with st.expander(f"[{i}] {Path(src).name}"):
                st.write(sn.node.text[:800])
else:
    st.info("Enter a question and click **Ask** to query the archive.")
