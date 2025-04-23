# ==================================================
# 🔄 RAG Pipeline with Groq + FAISS + HuggingFace
# - Loads mental health data from CSV
# - Embeds and indexes with BGE + FAISS
# - Uses Groq's LLaMA3 for retrieval-augmented responses
#
# ✅ TODOs for future enhancement:
#   - Add agent routing or workflow composition
#   - Integrate memory module for contextual conversations
#   - Experiment with different chunkers (e.g. semantic, sliding window)
#   - Improve embedding or rerank retrieved nodes
# ==================================================

import pandas as pd
from llama_index.core.schema import TextNode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
import faiss
import os
from dotenv import load_dotenv

load_dotenv()

# ==================================================
# 📄 Load and Split CSV
# ==================================================
def load_and_split_csv(csv_path="data/labeled_with_severity_nuanced.csv"):
    """
    Load mental health records from a CSV and split into chunks.
    Combines 'text', 'label', and 'severity_index' to form enriched text input.

    Returns:
        List[TextNode]: Sentence-level chunks enriched with metadata for retrieval.
    """
    df = pd.read_csv(csv_path)

    rows = []
    for _, row in df.iterrows():
        combined_text = (
            f"Source: {row.get('source', '')}\n"
            f"Timestamp: {row.get('timestamp', '')}\n"
            f"Label: {row.get('label', '')} (Severity {row.get('severity_index', '')})\n\n"
            f"{row.get('text', '')}"
        )
        text_node = TextNode(
            text=combined_text,
            metadata={
                "label": row.get("label", ""),
                "severity": row.get("severity_index", ""),
                "source": row.get("source", "")
            }
        )
        rows.append(text_node)

    # Chunk strategy: 512 tokens w/ 128 token overlap
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=128)
    nodes = splitter.get_nodes_from_documents(rows)

    print(f"✅ Loaded {len(df)} rows and split into {len(nodes)} chunks.")
    return nodes


# ==================================================
# 📦 Build FAISS Index and Save to Disk
# ==================================================
def build_faiss_index(nodes, index_save_dir="vector_store/"):
    """
    Creates and saves a FAISS vector index for efficient similarity search.

    Args:
        nodes (List[TextNode]): Chunked input nodes to index.
        index_save_dir (str): Directory path to store the index.

    Returns:
        VectorStoreIndex: Indexed vector store ready for querying.
    """
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    # Use embedding length to define index dimensions
    dimension = len(embed_model.get_query_embedding("test query"))
    faiss_index = faiss.IndexFlatL2(dimension)
    vector_store = FaissVectorStore(faiss_index=faiss_index)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model)

    if not os.path.exists(index_save_dir):
        os.makedirs(index_save_dir)
    index.storage_context.persist(persist_dir=index_save_dir)

    print(f"✅ FAISS index created and saved to '{index_save_dir}'")
    return index


# ==================================================
# ❓ Query with Groq LLaMA3 + FAISS
# ==================================================
def query_index(query, index_dir="vector_store/"):
    """
    Loads FAISS index and uses Groq-backed LLaMA 3.1 to answer a query.

    Args:
        query (str): Input question.
        index_dir (str): Directory of stored index.

    Returns:
        str: Answer from the LLM.
    """
    from llama_index.llms.groq import Groq

    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    vector_store = FaissVectorStore.from_persist_dir(index_dir)

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        persist_dir=index_dir
    )

    index = load_index_from_storage(
        storage_context=storage_context,
        embed_model=embed_model
    )

    llm = Groq(model="llama3-8b-8192")
    query_engine = index.as_query_engine(llm=llm)
    response = query_engine.query(query)

    print("\n🤖 Response:\n")
    print(response)
    return str(response)
