# Importing required libraries for hybrid retrieval and XAI attribution engine
import logging
import numpy as np
import ollama
import chromadb
from chromadb.config import Settings
from chromadb.api import ClientAPI
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from typing import List, Dict, Any, Optional, Tuple

# Configuring logging for RAG pipeline operations
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Defining ChromaDB and model configuration constants
CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL = "nomic-embed-text"
PARENT_COLLECTION = "parent_chunks"
CHILD_COLLECTION = "child_chunks"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K_DENSE = 10
TOP_K_BM25 = 10
TOP_K_FINAL = 5
RRF_K = 60


# Generating local embedding vector using nomic-embed-text via Ollama
def embed_text(text: str) -> List[float]:
    response = ollama.embeddings(
        model=EMBEDDING_MODEL,
        prompt=text
    )
    return list(response["embedding"])


# Encapsulating ChromaDB client and collection lifecycle in RAGStore
class RAGStore:
    def __init__(self) -> None:
        self.client: ClientAPI = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        self.parent_col, self.child_col = self._init_collections()
        self.reranker = CrossEncoder(CROSS_ENCODER_MODEL)
        logger.info(f"Initializing RAGStore with ChromaDB at: {CHROMA_PATH}")
        logger.info("Loading CrossEncoder reranker model as singleton...")

    # Initializing parent and child collections in ChromaDB
    def _init_collections(self) -> Tuple[chromadb.Collection, chromadb.Collection]:
        parent_col = self.client.get_or_create_collection(
            name=PARENT_COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )
        child_col = self.client.get_or_create_collection(
            name=CHILD_COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("Initializing parent and child ChromaDB collections...")
        return parent_col, child_col

    # Storing parent chunk with embedding into ChromaDB parent collection
    def store_parent(self, parent: Dict[str, Any]) -> None:
        embedding = embed_text(parent["content"])
        self.parent_col.upsert(
            ids=[parent["chunk_id"]],
            embeddings=[embedding],
            documents=[parent["content"]],
            metadatas=[{
                "file_path": parent["file_path"],
                "file_name": parent["metadata"]["file_name"],
                "type": "parent",
                "total_lines": str(parent["total_lines"]),
                "children": ",".join(parent["children"])
            }]
        )
        logger.info(f"Storing parent chunk: {parent['metadata']['file_name']}")

    # Storing child chunk with parent binding into ChromaDB child collection
    def store_child(self, child: Dict[str, Any]) -> None:
        embedding = embed_text(child["content"])
        self.child_col.upsert(
            ids=[child["chunk_id"]],
            embeddings=[embedding],
            documents=[child["content"]],
            metadatas=[{
                "file_path": child["file_path"],
                "file_name": child["metadata"]["file_name"],
                "type": "child",
                "node_type": child["node_type"],
                "function_name": child["metadata"]["function_name"],
                "start_line": str(child["start_line"]),
                "end_line": str(child["end_line"]),
                "parent_id": child["parent_id"]
            }]
        )
        logger.info(
            f"Storing child chunk: {child['node_type']} "
            f"'{child['name']}' → parent: {child['parent_id'][:12]}..."
        )

    # Ingesting full TreeRAG index into ChromaDB hierarchical store
    def ingest_index(self, index: Dict[str, Any]) -> None:
        parents: List[Dict[str, Any]] = index.get("parents", [])
        children: List[Dict[str, Any]] = index.get("children", [])

        logger.info(f"Ingesting {len(parents)} parent chunks into ChromaDB...")
        for parent in parents:
            self.store_parent(parent)

        logger.info(f"Ingesting {len(children)} child chunks into ChromaDB...")
        for child in children:
            self.store_child(child)

        logger.info(
            f"Completing ChromaDB ingestion — "
            f"parents: {len(parents)}, children: {len(children)}"
        )

    # Retrieving top-k child chunks using dense vector similarity search
    def dense_retrieve(
        self,
        query: str,
        top_k: int = TOP_K_DENSE
    ) -> List[Dict[str, Any]]:
        query_embedding = embed_text(query)
        results = self.child_col.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        chunks: List[Dict[str, Any]] = []
        documents = results.get("documents") or [[]]
        metadatas = results.get("metadatas") or [[]]
        distances = results.get("distances") or [[]]

        for doc, meta, dist in zip(documents[0], metadatas[0], distances[0]):
            chunks.append({
                "content": doc,
                "metadata": meta,
                "distance": dist,
                "similarity": round(1 - dist, 4)
            })

        logger.info(f"Retrieving {len(chunks)} chunks via dense search...")
        return chunks

    # Retrieving top-k child chunks using BM25 sparse keyword matching
    def bm25_retrieve(
        self,
        query: str,
        all_chunks: List[Dict[str, Any]],
        top_k: int = TOP_K_BM25
    ) -> List[Dict[str, Any]]:
        tokenized_corpus = [chunk["content"].lower().split() for chunk in all_chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)

        ranked_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in ranked_indices:
            chunk = all_chunks[idx].copy()
            chunk["bm25_score"] = float(scores[idx])
            results.append(chunk)

        logger.info(f"Retrieving {len(results)} chunks via BM25 search...")
        return results

    # Combining dense and BM25 results using Reciprocal Rank Fusion algorithm
    def reciprocal_rank_fusion(
        self,
        dense_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        k: int = RRF_K
    ) -> List[Dict[str, Any]]:
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Dict[str, Any]] = {}

        for rank, chunk in enumerate(dense_results):
            cid = chunk["metadata"].get("function_name", "") + chunk["content"][:30]
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (k + rank + 1)
            chunk_map[cid] = chunk

        for rank, chunk in enumerate(bm25_results):
            cid = chunk["metadata"].get("function_name", "") + chunk["content"][:30]
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (k + rank + 1)
            chunk_map[cid] = chunk

        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
        fused = []
        for cid in sorted_ids:
            chunk = chunk_map[cid].copy()
            chunk["rrf_score"] = round(rrf_scores[cid], 6)
            fused.append(chunk)

        logger.info(f"Fusing {len(fused)} unique chunks via RRF...")
        return fused

    # Reranking fused chunks using local CrossEncoder similarity model
    def cross_encoder_rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = TOP_K_FINAL
    ) -> List[Dict[str, Any]]:
        pairs = [[query, chunk["content"]] for chunk in chunks]
        scores = self.reranker.predict(pairs)

        for chunk, score in zip(chunks, scores):
            chunk["cross_encoder_score"] = float(score)

        reranked = sorted(
            chunks, key=lambda x: x["cross_encoder_score"], reverse=True
        )[:top_k]
        logger.info(f"Reranking {len(reranked)} chunks via CrossEncoder...")
        return reranked

    # Calculating Softmax-based TreeRAG attribution scores for XAI output
    def calculate_attribution_scores(
        self,
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        scores = np.array([chunk["cross_encoder_score"] for chunk in chunks])
        exp_scores = np.exp(scores - np.max(scores))
        softmax_scores = exp_scores / exp_scores.sum()

        for chunk, attribution in zip(chunks, softmax_scores):
            chunk["attribution_score"] = round(float(attribution), 4)
            chunk["attribution_pct"] = round(float(attribution) * 100, 2)

        logger.info("Calculating Softmax TreeRAG attribution scores...")
        return chunks

    # Fetching parent chunk context by ID for full file context retrieval
    def get_parent_by_id(self, parent_id: str) -> Optional[Dict[str, Any]]:
        result = self.parent_col.get(
            ids=[parent_id],
            include=["documents", "metadatas"]
        )

        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []

        if documents and metadatas:
            logger.info(f"Fetching parent chunk by ID: {parent_id[:12]}...")
            return {
                "content": documents[0],
                "metadata": metadatas[0]
            }
        return None

    # Executing full hybrid TreeRAG retrieval pipeline with XAI attribution
    def hybrid_retrieve(
        self,
        query: str,
        top_k: int = TOP_K_FINAL
    ) -> List[Dict[str, Any]]:
        logger.info(f"Starting hybrid TreeRAG retrieval for query: '{query[:60]}...'")

        # Fetching all chunks for BM25 corpus construction...
        all_results = self.child_col.get(include=["documents", "metadatas"])
        all_docs = all_results.get("documents") or []
        all_metas = all_results.get("metadatas") or []
        all_chunks = [
            {"content": doc, "metadata": meta}
            for doc, meta in zip(all_docs, all_metas)
        ]

        if not all_chunks:
            logger.warning("No chunks found in ChromaDB — running indexing first...")
            return []

        dense_results = self.dense_retrieve(query, top_k=top_k * 2)
        bm25_results = self.bm25_retrieve(query, all_chunks, top_k=top_k * 2)
        fused_results = self.reciprocal_rank_fusion(dense_results, bm25_results)
        reranked = self.cross_encoder_rerank(query, fused_results, top_k=top_k)
        attributed = self.calculate_attribution_scores(reranked)

        logger.info(
            f"Completing hybrid retrieval — "
            f"returning {len(attributed)} chunks with attribution scores"
        )
        return attributed


if __name__ == "__main__":
    # Running standalone hybrid retrieval pipeline test
    from app.indexer import index_codebase

    logger.info("Starting hybrid TreeRAG pipeline test...")
    store = RAGStore()
    index = index_codebase("codebase/")
    store.ingest_index(index)

    query = "function that adds two numbers"
    results = store.hybrid_retrieve(query)

    print("\n--- TreeRAG Hybrid Retrieval + XAI Attribution ---")
    print(f"Query: {query}\n")
    for i, chunk in enumerate(results):
        print(f"[{i+1}] {chunk['metadata'].get('function_name', 'N/A')}")
        print(f"     Attribution  : {chunk['attribution_pct']}%")
        print(f"     CrossEncoder : {round(chunk['cross_encoder_score'], 4)}")
        print(f"     RRF Score    : {chunk.get('rrf_score', 'N/A')}")
        print(f"     File         : {chunk['metadata'].get('file_name', 'N/A')}")
        print(f"     Content      : {chunk['content'][:80]}...")
        print()