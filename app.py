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


class EmbeddingDimensionError(RAGError):
    """Raised when embedding dimensions don't match."""
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
        errors.append("   Upload your index files to the repository")
    
    return len(errors) == 0, errors


def handle_error(error: Exception, context: str = "") -> None:
    """Centralized error handler with user-friendly messages."""
    error_msg = f"{context}: {str(error)}" if context else str(error)
    logger.error(error_msg, exc_info=True)
    
    if isinstance(error, IndexLoadError):
        st.error("🔴 Failed to load the search index")
        st.info("**Solution:** Ensure index files are uploaded to `data/index/`")
    elif isinstance(error, EmbeddingDimensionError):
        st.error("🔴 Embedding model mismatch detected")
        st.info("**Solution:** Contact administrator to rebuild the index.")
    else:
        st.error(f"🔴 An error occurred: {str(error)}")
        with st.expander("Show technical details"):
            st.code(error_msg)


# ============================================================================
# Core Functions
# ============================================================================

@st.cache_resource(show_spinner=False)
def initialize_models():
    """Initialize and cache embedding and LLM models."""
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
    """Load FAISS index with comprehensive error handling."""
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
    """Validate that embedding dimensions match."""
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
    """Get cached reranker model."""
    try:
        logger.info(f"Initializing reranker: {model}")
        return SentenceTransformerRerank(top_n=top_n, model=model)
    except Exception as e:
        logger.error(f"Failed to initialize reranker: {e}", exc_info=True)
        raise RAGError(f"Reranker initialization failed: {e}")


def extract_book_name(file_path: str) -> str:
    """
    Extract clean book name from file path.
    
    Examples:
        '/path/to/The_Secret_Teachings.pdf' -> 'The Secret Teachings'
        'book_title.txt' -> 'Book Title'
    """
    if not file_path:
        return "Unknown Source"
    
    # Get filename without path
    filename = Path(file_path).stem
    
    # Replace underscores and hyphens with spaces
    clean_name = filename.replace('_', ' ').replace('-', ' ')
    
    # Remove common suffixes
    for suffix in [' ocr', ' text', ' txt', ' pdf']:
        if clean_name.lower().endswith(suffix):
            clean_name = clean_name[:-len(suffix)]
    
    # Title case
    clean_name = clean_name.title()
    
    return clean_name.strip() or "Unknown Source"


# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main application logic."""
    
    # Header
    st.title("📜 Ask Manly P. Hall")
    st.markdown("*Explore the teachings of Manly P. Hall through AI-powered search*")
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
    
    # Validate embedding dimensions (silently, only log errors)
    is_valid, model_dim, index_dim = validate_embedding_dimensions(embed_model, vector_store)
    
    if not is_valid:
        if index_dim == -1:
            st.error("❌ Search index is corrupted or incompatible")
            st.info("**Solution:** Contact administrator")
        else:
            st.error("❌ Search index configuration error")
            st.info(
                "The search index was built with a different model configuration. "
                "Please contact the administrator to rebuild the index."
            )
        logger.error(f"Dimension mismatch: model={model_dim}, index={index_dim}")
        st.stop()
    
    # Sidebar settings (minimal)
    with st.sidebar:
        st.header("⚙️ Search Settings")
        
        RAG_TOP_K = int(os.getenv("RAG_TOP_K", "12"))
        RAG_TOP_N = int(os.getenv("RAG_TOP_N", "6"))
        
        top_k = st.slider(
            "Number of passages to retrieve",
            min_value=5,
            max_value=40,
            value=RAG_TOP_K,
            help="More passages = broader search but slower"
        )
        
        top_n = st.slider(
            "Number of passages to use for answer",
            min_value=3,
            max_value=15,
            value=RAG_TOP_N,
            help="Fewer passages = faster, more focused answers"
        )
        
        show_sources = st.checkbox(
            "Show source books",
            value=True,
            help="Display which books were used to generate the answer"
        )
        
        st.divider()
        
        if st.button("🔄 Reload Index", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()
        
        st.caption("*Powered by LlamaIndex & OpenAI*")
    
    # Query interface
    question = st.text_area(
        "**What would you like to know?**",
        placeholder="Ask about symbolism, philosophy, ancient mysteries, or esoteric teachings...",
        height=100,
        help="Be specific for better results"
    )
    
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        ask = st.button("🔍 Ask", type="primary", use_container_width=True)
    with col2:
        if st.button("Clear", use_container_width=True):
            st.rerun()
    
    if ask and question.strip():
        try:
            with st.spinner("🔍 Searching Manly P. Hall's works..."):
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
                
                # Validate query embedding dimension
                qdim = len(Settings.embed_model.get_text_embedding(question))
                if qdim != index_dim:
                    st.error("❌ Query processing error. Please try again.")
                    logger.error(f"Query dim {qdim} != index dim {index_dim}")
                    st.stop()
                
                # Execute query
                response = qe.query(question)
                
                if not response or not str(response).strip():
                    st.warning("⚠️ No answer found. Try rephrasing your question.")
                    st.stop()
            
            # Display answer
            st.success("**Answer:**")
            st.markdown(str(response).strip())
            
            # Display sources (clean version)
            if show_sources and hasattr(response, "source_nodes") and response.source_nodes:
                st.divider()
                st.markdown("##### 📚 Source Books")
                
                # Extract unique book names
                book_names = set()
                for sn in response.source_nodes:
                    meta = sn.node.metadata or {}
                    src = meta.get("file_path") or meta.get("file_name") or ""
                    book_name = extract_book_name(src)
                    book_names.add(book_name)
                
                # Display as a clean list
                if book_names:
                    # Remove "Unknown Source" if other sources exist
                    if len(book_names) > 1 and "Unknown Source" in book_names:
                        book_names.remove("Unknown Source")
                    
                    for book in sorted(book_names):
                        st.markdown(f"- *{book}*")
                else:
                    st.caption("Source information unavailable")
        
        except Exception as e:
            handle_error(e, "Query execution")
    
    elif not question.strip() and ask:
        st.info("💡 Please enter a question to search.")
    
    # Footer
    st.divider()
    with st.expander("ℹ️ About this tool"):
        st.markdown("""
        This tool uses AI to search through Manly P. Hall's extensive body of work, 
        including his books and lectures on symbolism, philosophy, and esoteric traditions.
        
        **How it works:**
        1. Your question is converted into a semantic representation
        2. Relevant passages are retrieved from the indexed texts
        3. An AI model synthesizes these passages into a coherent answer
        
        **Tips for best results:**
        - Be specific in your questions
        - Ask about concepts, symbols, or teachings
        - Try rephrasing if you don't get a good answer
        
        **Note:** This tool provides information based on Manly P. Hall's works. 
        Answers are AI-generated and should be verified for critical research.
        """)


if __name__ == "__main__":
    main()
