import os
import logging
import streamlit as st
from pathlib import Path
from typing import Tuple, List, Optional
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
for key in ["RAG_PERSIST_DIR", "RAG_LLM_MODEL", "RAG_TOP_K", "RAG_TOP_N"]:
    if key in st.secrets:
        os.environ[key] = str(st.secrets[key])

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_INDEX_DIR = DATA_DIR / "index"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

st.set_page_config(page_title="Manly P. Hall RAG", layout="wide")
st.title("📜 Ask Manly P. Hall")
st.caption("RAG over Manly P. Hall's works with FAISS + reranking.")


class RAGError(Exception):
    """Base exception for RAG application."""
    pass


class IndexLoadError(RAGError):
    """Raised when index cannot be loaded."""
    pass


class EmbeddingDimensionError(RAGError):
    """Raised when embedding dimensions don't match."""
    pass


def validate_environment() -> Tuple[bool, List[str]]:
    """
    Validate required environment/secrets.
    
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    if not os.getenv("OPENAI_API_KEY"):
        errors.append("❌ OPENAI_API_KEY not configured in Streamlit secrets")
        errors.append("   Add it in: Settings → Secrets")
    
    persist_dir = os.getenv("RAG_PERSIST_DIR", str(DEFAULT_INDEX_DIR))
    if not Path(persist_dir).exists():
        errors.append(f"❌ Index directory not found: {persist_dir}")
        errors.append("   Upload your index files to the repository")
    
    return len(errors) == 0, errors


def handle_error(error: Exception, context: str = "") -> None:
    """
    Centralized error handler with user-friendly messages.
    
    Args:
        error: The exception that occurred
        context: Additional context
    """
    error_msg = f"{context}: {str(error)}" if context else str(error)
    logger.error(error_msg, exc_info=True)
    
    if isinstance(error, IndexLoadError):
        st.error("🔴 Failed to load the search index")
        st.info("**Solution:** Ensure index files are uploaded to `data/index/`")
    elif isinstance(error, EmbeddingDimensionError):
        st.error("🔴 Embedding model mismatch detected")
        st.info("**Solution:** Index was built with a different embedding model. "
                "Contact administrator to rebuild the index.")
    else:
        st.error(f"🔴 An error occurred: {str(error)}")
        with st.expander("Show details"):
            st.code(error_msg)


@st.cache_resource(show_spinner=False)
def initialize_models():
    """
    Initialize and cache embedding and LLM models.
    
    Returns:
        Tuple of (embed_model, llm)
    """
    try:
        logger.info(f"Initializing embedding model: {EMBED_MODEL_NAME}")
        embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
        
        llm_model = os.getenv("RAG_LLM_MODEL", "gpt-4o-mini")
        logger.info(f"Initializing LLM: {llm_model}")
        llm = OpenAI(model=llm_model, temperature=0)
        
        return embed_model, llm
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}", exc_info=True)
        raise RAGError(f"Model initialization failed: {e}")


@st.cache_resource(show_spinner=False)
def load_index(persist_dir: str):
    """
    Load FAISS index with comprehensive error handling.
    
    Args:
        persist_dir: Directory where index is persisted
        
    Returns:
        Tuple of (index, vector_store)
    """
    try:
        persist_path = Path(persist_dir)
        
        if not persist_path.exists():
            raise IndexLoadError(f"Index directory does not exist: {persist_dir}")
        
        required_files = ["docstore.json", "index_store.json"]
        missing_files = [f for f in required_files if not (persist_path / f).exists()]
        if missing_files:
            raise IndexLoadError(f"Missing index files: {', '.join(missing_files)}")
        
        logger.info(f"Loading index from: {persist_dir}")
        vector_store = FaissVectorStore.from_persist_dir(persist_dir=persist_dir)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=persist_dir
        )
        index = load_index_from_storage(storage_context)
        
        logger.info("✓ Index loaded successfully")
        return index, vector_store
        
    except IndexLoadError:
        raise
    except Exception as e:
        logger.error(f"Failed to load index: {e}", exc_info=True)
        raise IndexLoadError(f"Could not load index from {persist_dir}: {e}")


def validate_embedding_dimensions(embed_model, vector_store) -> Tuple[bool, int, int]:
    """
    Validate that embedding dimensions match.
    
    Args:
        embed_model: The embedding model
        vector_store: The FAISS vector store
        
    Returns:
        Tuple of (is_valid, model_dim, index_dim)
    """
    try:
        probe_vec = embed_model.get_text_embedding("dimension probe")
        model_dim = len(probe_vec)
        
        fa = getattr(vector_store, "faiss_index", None)
        if fa is None:
            fa = getattr(vector_store, "_faiss_index", None)
        
        if fa is None:
            logger.warning("Could not access FAISS index")
            return False, model_dim, -1
        
        index_dim = getattr(fa, "d", None)
        if index_dim is None:
            logger.warning("Could not read FAISS index dimension")
            return False, model_dim, -1
        
        is_valid = (model_dim == index_dim)
        
        if not is_valid:
            logger.error(f"Dimension mismatch: model={model_dim}, index={index_dim}")
        else:
            logger.info(f"✓ Dimensions validated: {model_dim}")
        
        return is_valid, model_dim, index_dim
        
    except Exception as e:
        logger.error(f"Failed to validate dimensions: {e}", exc_info=True)
        return False, -1, -1


@st.cache_resource
def get_reranker(top_n: int, model: str = RERANK_MODEL_NAME):
    """
    Get cached reranker model.
    
    Args:
        top_n: Number of documents after reranking
        model: Reranker model name
        
    Returns:
        SentenceTransformerRerank instance
    """
    try:
        logger.info(f"Initializing reranker: {model}")
        return SentenceTransformerRerank(top_n=top_n, model=model)
    except Exception as e:
        logger.error(f"Failed to initialize reranker: {e}", exc_info=True)
        raise RAGError(f"Reranker initialization failed: {e}")

def main():
    """Main application logic."""
    
    is_valid, errors = validate_environment()
    if not is_valid:
        st.error("⚠️ Configuration Issues")
        for error in errors:
            st.markdown(error)
        st.stop()
    
    st.sidebar.header("Settings")
    RAG_TOP_K = int(os.getenv("RAG_TOP_K", "12"))
    RAG_TOP_N = int(os.getenv("RAG_TOP_N", "6"))
    
    top_k = st.sidebar.slider("Top-K retrieved chunks", 5, 40, RAG_TOP_K)
    top_n = st.sidebar.slider("Top-N after rerank", 3, 15, RAG_TOP_N)
    show_sources = st.sidebar.checkbox("Show sources", value=True)
    
    try:
        embed_model, llm = initialize_models()
        Settings.embed_model = embed_model
        Settings.llm = llm
    except RAGError as e:
        handle_error(e, "Model initialization")
        st.stop()
    
    persist_dir = os.getenv("RAG_PERSIST_DIR", str(DEFAULT_INDEX_DIR))
    
    colA, colB = st.sidebar.columns(2)
    with colA:
        if st.button("Reload index", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()
    
    try:
        with st.spinner("Loading search index..."):
            index, vector_store = load_index(persist_dir)
    except IndexLoadError as e:
        handle_error(e, "Index loading")
        st.stop()
    
    is_valid, model_dim, index_dim = validate_embedding_dimensions(embed_model, vector_store)
    
    st.sidebar.markdown(
        f"**Index dim:** {index_dim}  \n"
        f"**Model dim:** {model_dim}  \n"
        f"**Embedder:** `{EMBED_MODEL_NAME}`"
    )
    
    if not is_valid:
        if index_dim == -1:
            st.error("❌ Could not read FAISS index dimension. Index may be corrupted.")
            st.info("**Solution:** Contact administrator to rebuild the index")
        else:
            st.error(
                f"❌ **Embedding Dimension Mismatch**\n\n"
                f"- FAISS index expects: **{index_dim}** dimensions\n"
                f"- Current model outputs: **{model_dim}** dimensions\n\n"
                f"**The index was built with a different embedding model.**"
            )
            st.warning(
                "**To fix:**\n\n"
                "Contact the administrator to rebuild the index with:\n"
                f"`{EMBED_MODEL_NAME}`\n\n"
                "Or update `EMBED_MODEL_NAME` in the code to match the index."
            )
        st.stop()
    
    question = st.text_area(
        "Ask a question:",
        placeholder="e.g., What is the symbolic meaning of the number 33 in Freemasonry?",
        height=100
    )
    ask = st.button("Ask", type="primary")
    
    if ask and question.strip():
        try:
            with st.spinner("🔍 Retrieving and generating answer..."):
                retriever = index.as_retriever(similarity_top_k=top_k)
                
                reranker = get_reranker(top_n=top_n)
                
                qe = RetrieverQueryEngine.from_args(
                    retriever=retriever,
                    node_postprocessors=[reranker],
                    response_mode="compact",
                )
                
                qdim = len(Settings.embed_model.get_text_embedding(question))
                if qdim != index_dim:
                    st.error(
                        f"❌ Query embedding dimension ({qdim}) doesn't match "
                        f"index dimension ({index_dim}). This shouldn't happen!"
                    )
                    st.stop()
                
                response = qe.query(question)
                
                if not response or not str(response).strip():
                    st.warning("⚠️ No answer generated. Try rephrasing your question.")
                    st.stop()
            
            st.markdown("### 🧠 Answer")
            st.write(str(response).strip())
            
            if show_sources and hasattr(response, "source_nodes") and response.source_nodes:
                st.markdown("### 📚 Sources")
                for i, sn in enumerate(response.source_nodes, start=1):
                    meta = sn.node.metadata or {}
                    src = meta.get("file_path") or meta.get("file_name") or "unknown"
                    score = getattr(sn, "score", None)
                    score_str = f" (score: {score:.3f})" if score is not None else ""
                    
                    with st.expander(f"[{i}] {Path(src).name}{score_str}"):
                        st.write(sn.node.text[:800])
                        if len(sn.node.text) > 800:
                            st.caption("... (truncated)")
            elif show_sources:
                st.info("No source documents available")
        
        except Exception as e:
            handle_error(e, "Query execution")
    
    else:
        st.info("💡 Enter a question and click **Ask** to query the archive.")


if __name__ == "__main__":
    main()
