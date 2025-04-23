# ==================================================
# 🤖 Conversational Agent Setup (Groq + FAISS)
# - Loads retriever from FAISS vector index
# - Connects to Groq LLaMA 3.1 model
# - Returns user-aware responses from indexed chunks
#
# ✅ TODOs for future enhancement:
#   - Add reranking layer or semantic filters
#   - Enable LangGraph or AutoGen workflows
#   - Integrate memory for multi-turn chat
#   - Enable real-time streaming or follow-up generation
# ==================================================

from dotenv import load_dotenv
import os

# ✅ Load environment variables
load_dotenv()

# ==================================================
# 📦 Imports
# ==================================================
from llama_index.llms.groq import Groq
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import load_index_from_storage
from llama_index.core.storage import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from rag_pipeline import load_and_split_csv, build_faiss_index

# ==================================================
# ✅ Ensure FAISS index exists (first run safety)
# ==================================================
def ensure_index_exists(index_dir="vector_store/", data_path="data/labeled_with_severity_nuanced.csv"):
    if not os.path.exists(index_dir) or not os.listdir(index_dir):
        print("⚠️ Index not found. Building a new one from dataset...")
        nodes = load_and_split_csv(csv_path=data_path)
        build_faiss_index(nodes, index_save_dir=index_dir)
    else:
        print("✅ FAISS index already exists. Skipping index creation.")

# ==================================================
# 🔍 Load Retriever from FAISS + Embedding
# ==================================================
def load_retriever(index_dir="vector_store/", top_k=5):
    """
    Load a retriever from a saved FAISS index and embedding model.

    Args:
        index_dir (str): Directory where the index is saved.
        top_k (int): Number of top relevant chunks to retrieve.

    Returns:
        VectorIndexRetriever: Configured retriever for querying.
    """
    ensure_index_exists(index_dir=index_dir)

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    vector_store = FaissVectorStore.from_persist_dir(index_dir)
    storage_context = StorageContext.from_defaults(
        persist_dir=index_dir,
        vector_store=vector_store
    )

    index = load_index_from_storage(
        storage_context=storage_context,
        embed_model=embed_model
    )
    retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)

    return retriever

# ==================================================
# 🧠 Set up Groq LLaMA 3.1 for Assistant
# ==================================================
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
llm = Groq(
    model="llama-3.1-8b-instant",
    system_prompt=(
        "You are a compassionate and helpful mental health assistant.\n"
        "Use supportive language and stay brief.\n"
        "Base answers on the retrieved information, and avoid giving medical advice."
    )
)

# ==================================================
# 🔁 Create Query Engine with Extension Hooks
# ==================================================
retriever = load_retriever()
query_engine = RetrieverQueryEngine.from_args(
    retriever=retriever,
    llm=llm,
    response_mode="compact"
)

# ==================================================
# 💬 Handle Incoming User Query
# ==================================================
def handle_user_query(user_input):
    """
    Uses LlamaIndex + Groq LLaMA 3.1 to answer user input with contextual awareness.

    Args:
        user_input (str): The question or message from the user.

    Returns:
        str: The LLM's response.
    """
    response = query_engine.query(user_input)
    return str(response)