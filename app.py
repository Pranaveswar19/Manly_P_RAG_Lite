import os
import logging
import streamlit as st
from pathlib import Path
from typing import Tuple, List
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load secrets (Streamlit Cloud style)
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
for key in ["RAG_PERSIST_DIR", "RAG_LLM_MODEL", "RAG_TOP_K", "RAG_TOP_N"]:
    if key in st.secrets:
        os.environ[key] = str(st.secrets[key])

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_INDEX_DIR = DATA_DIR / "index"

# Embedding model - MUST match index build
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Page config
st.set_page_config(
    page_title="Ask Manly P. Hall",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================================
# Error Handling
# ============================================================================

class RAGError(Exception):
    """Base exception for RAG application."""
    pass


class IndexLoadError(RAGError):
    """Raised when index cannot be loaded."""
    pass


def validate_environment() -> Tuple[bool, List[str]]:
    """Validate required environment/secrets."""
    errors = []
    
    if not os.getenv("OPENAI_API_KEY"):
        errors.append("❌ OPENAI_API_KEY not configured")
        errors.append("   Add it in: Settings → Secrets")
    
    persist_dir = os.getenv("RAG_PERSIST_DIR", str(DEFAULT_INDEX_DIR))
    if not Path(persist_dir).exists():
        errors.append(f"❌ Index directory not found: {persist_dir}")
    
    return len(errors) == 0, errors


def handle_error(error: Exception, context: str = "") -> None:
    """Centralized error handler."""
    error_msg = f"{context}: {str(error)}" if context else str(error)
    logger.error(error_msg, exc_info=True)
    
    if isinstance(error, IndexLoadError):
        st.error("🔴 Failed to load knowledge base")
        st.info("**Solution:** Please contact the administrator")
    else:
        st.error("🔴 An error occurred. Please try again or contact support.")
        with st.expander("Technical details"):
            st.code(error_msg)


# ============================================================================
# Core Functions
# ============================================================================

@st.cache_resource(show_spinner=False)
def initialize_models():
    """Initialize and cache models."""
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
    """Load FAISS index."""
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
        raise IndexLoadError(f"Could not load index: {e}")


def validate_embedding_dimensions(embed_model, vector_store) -> bool:
    """Validate dimensions match (returns simple bool)."""
    try:
        probe_vec = embed_model.get_text_embedding("probe")
        model_dim = len(probe_vec)
        
        fa = getattr(vector_store, "faiss_index", None) or getattr(vector_store, "_faiss_index", None)
        if fa is None:
            return False
        
        index_dim = getattr(fa, "d", None)
        if index_dim is None:
            return False
        
        is_valid = (model_dim == index_dim)
        if not is_valid:
            logger.error(f"Dimension mismatch: model={model_dim}, index={index_dim}")
        else:
            logger.info(f"✓ Dimensions validated: {model_dim}")
        
        return is_valid
        
    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=True)
        return False


@st.cache_resource
def get_reranker(top_n: int, model: str = RERANK_MODEL_NAME):
    """Get cached reranker."""
    try:
        logger.info(f"Initializing reranker: {model}")
        return SentenceTransformerRerank(top_n=top_n, model=model)
    except Exception as e:
        logger.error(f"Failed to initialize reranker: {e}", exc_info=True)
        raise RAGError(f"Reranker initialization failed: {e}")


# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main application logic."""
    
    # Header
    st.title("📜 Ask Manly P. Hall")
    st.markdown("*Explore the wisdom of Manly P. Hall through AI-powered search*")
    st.divider()
    
    # Validate environment
    is_valid, errors = validate_environment()
    if not is_valid:
        st.error("⚠️ Configuration Error")
        for error in errors:
            st.markdown(error)
        st.stop()
    
    # Initialize models
    try:
        embed_model, llm = initialize_models()
        Settings.embed_model = embed_model
        Settings.llm = llm
    except RAGError as e:
        handle_error(e, "Model initialization")
        st.stop()
    
    # Load index
    persist_dir = os.getenv("RAG_PERSIST_DIR", str(DEFAULT_INDEX_DIR))
    
    try:
        with st.spinner("Loading knowledge base..."):
            index, vector_store = load_index(persist_dir)
    except IndexLoadError as e:
        handle_error(e, "Index loading")
        st.stop()
    
    # Validate dimensions (silent check)
    if not validate_embedding_dimensions(embed_model, vector_store):
        st.error("❌ System configuration error. Please contact the administrator.")
        st.stop()
    
    # Sidebar settings (minimal, clean)
    with st.sidebar:
        st.header("⚙️ Settings")
        
        RAG_TOP_K = int(os.getenv("RAG_TOP_K", "12"))
        RAG_TOP_N = int(os.getenv("RAG_TOP_N", "6"))
        
        top_k = st.slider(
            "Passages to retrieve",
            5, 40, RAG_TOP_K,
            help="Higher = broader search"
        )
        
        top_n = st.slider(
            "Passages for answer",
            3, 15, RAG_TOP_N,
            help="Lower = faster, focused"
        )
        
        st.divider()
        
        if st.button("🔄 Reload", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()
    
    # Query interface
    question = st.text_area(
        "**What would you like to know?**",
        placeholder="Ask about symbolism, philosophy, mysteries, esoteric teachings...",
        height=100,
        label_visibility="visible"
    )
    
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        ask = st.button("🔍 Search", type="primary", use_container_width=True)
    with col2:
        if st.button("Clear", use_container_width=True):
            st.rerun()
    
    if ask and question.strip():
        try:
            with st.spinner("Searching the archives..."):
                # Create retriever
                retriever = index.as_retriever(similarity_top_k=top_k)
                
                # Get reranker
                reranker = get_reranker(top_n=top_n)
                
                # Create query engine
                qe = RetrieverQueryEngine.from_args(
                    retriever=retriever,
                    node_postprocessors=[reranker],
                    response_mode="compact",
                )
                
                # Execute query
                response = qe.query(question)
                
                if not response or not str(response).strip():
                    st.warning("⚠️ No answer found. Try rephrasing your question.")
                    st.stop()
            
            # Display answer (clean, no sources)
            st.success("**Answer**")
            st.markdown(str(response).strip())
            
            st.caption("*Answer synthesized from Manly P. Hall's works*")
        
        except Exception as e:
            handle_error(e, "Search")
    
    elif not question.strip() and ask:
        st.info("💡 Please enter a question.")
    
    # Footer
    st.divider()
    with st.expander("ℹ️ About"):
        st.markdown("""
        ### About This Tool
        
        Search through Manly P. Hall's extensive body of work using AI-powered semantic search.
        
        **How it works:**
        - Your question is analyzed semantically
        - Relevant passages are retrieved from indexed texts
        - An AI synthesizes a coherent answer
        
        **Tips:**
        - Be specific
        - Ask about concepts, symbols, or teachings
        - Rephrase if needed
        
        **Note:** Answers are AI-generated based on Manly P. Hall's works.
        """)


if __name__ == "__main__":
    main()
