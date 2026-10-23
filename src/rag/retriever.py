"""Hybrid Precedence-Aware Retrieval Engine for ZentroMart Knowledge Base.

Combines lexical search (BM25) and dense semantic representations with
Reciprocal Rank Fusion (RRF) and deterministic policy conflict resolution (KB-00.1).

Model & Backend Diagnostics:
- Dense Semantic Model: BAAI/bge-small-en-v1.5 (English-optimized dense embeddings via FastEmbed ONNX).
- Sparse Fallback Model: Scikit-Learn sublinear TF-IDF (character + word n-grams for code-mixed / Hinglish).
- Language Capabilities: English, Hinglish (Latin subwords), and Hindi (Unicode tokenization in BM25).
"""
import re
import numpy as np
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from src.config import DEFAULT_EMBEDDING_MODEL
from src.rag.parser import KnowledgeChunk, load_all_chunks

class RetrievalResult(BaseModel):
    """Structured result returned by the retriever."""
    chunk_id: str
    doc_id: str
    section_id: str
    section_title: str
    content: str
    doc_type: str
    status: str
    authority_tier: int
    score: float = Field(description="Combined hybrid relevance score")
    rank: int = Field(description="1-based retrieval rank")
    bm25_score: float = 0.0
    dense_score: float = 0.0
    backend: str = Field(default="hybrid", description="Embedding / search backend used")

def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase words/alphanumerics supporting Latin and Devanagari."""
    if not text:
        return []
    # Matches words, numbers, codes (e.g. ELE-0012, RET-2), and Devanagari script
    tokens = re.findall(r"[\w\u0900-\u097F]+(?:-[\w\u0900-\u097F]+)*", text.lower())
    return tokens

class BM25Searcher:
    """Lexical retrieval using Okapi BM25."""
    
    def __init__(self, chunks: List[KnowledgeChunk]):
        self.chunks = chunks
        self.corpus = [tokenize(c.content) for c in chunks]
        try:
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi(self.corpus)
        except ImportError:
            self.bm25 = None

    def search(self, query: str, top_k: int = 20) -> List[tuple[KnowledgeChunk, float]]:
        if not self.bm25:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_indices if scores[i] > 0]

class SemanticSearcher:
    """Dense semantic retrieval with accurate backend diagnostics."""
    
    def __init__(self, chunks: List[KnowledgeChunk], model_name: Optional[str] = None):
        self.chunks = chunks
        self.backend = None
        self.model_name = model_name or DEFAULT_EMBEDDING_MODEL
        self.embeddings = None
        
        # Try FastEmbed ONNX local dense embeddings
        try:
            from fastembed import TextEmbedding
            self.embed_model = TextEmbedding(self.model_name)
            contents = [c.content for c in chunks]
            self.embeddings = np.array(list(self.embed_model.embed(contents)))
            self.backend = f"fastembed_dense ({self.model_name})"
        except Exception as e:
            # Fallback to Scikit-Learn TF-IDF sparse lexical subword representation
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(
                ngram_range=(1, 3),
                sublinear_tf=True,
                token_pattern=r"[\w\u0900-\u097F]+(?:-[\w\u0900-\u097F]+)*"
            )
            contents = [c.content for c in chunks]
            self.embeddings = self.vectorizer.fit_transform(contents).toarray()
            self.backend = "sklearn_sparse_tfidf_fallback"

    def search(self, query: str, top_k: int = 20) -> List[tuple[KnowledgeChunk, float]]:
        if self.embeddings is None:
            return []
        if "fastembed" in self.backend:
            q_emb = np.array(list(self.embed_model.embed([query])))[0]
        else:
            q_emb = self.vectorizer.transform([query]).toarray()[0]
            
        norm_q = np.linalg.norm(q_emb)
        if norm_q == 0:
            return []
        q_emb_norm = q_emb / norm_q
        
        chunk_norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        chunk_norms[chunk_norms == 0] = 1.0
        normalized_chunks = self.embeddings / chunk_norms
        
        scores = np.dot(normalized_chunks, q_emb_norm)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_indices if scores[i] > 0]

class HybridRetriever:
    """Precedence-Aware Hybrid Retriever combining BM25, Semantic Search, and Policy Hierarchy."""
    
    def __init__(self, chunks: Optional[List[KnowledgeChunk]] = None, embedding_model: Optional[str] = None):
        self.chunks = chunks if chunks is not None else load_all_chunks()
        self.bm25_searcher = BM25Searcher(self.chunks)
        self.semantic_searcher = SemanticSearcher(self.chunks, model_name=embedding_model)
        self.chunk_map = {c.chunk_id: c for c in self.chunks}
        self.active_backend = f"hybrid (bm25 + {self.semantic_searcher.backend})"

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        bm25_weight: float = 0.5,
        semantic_weight: float = 0.5,
        rrf_k: int = 60
    ) -> List[RetrievalResult]:
        """Execute hybrid search with Reciprocal Rank Fusion and deterministic policy conflict resolution."""
        bm25_results = self.bm25_searcher.search(query, top_k=30)
        semantic_results = self.semantic_searcher.search(query, top_k=30)
        
        bm25_ranks = {item[0].chunk_id: (rank + 1, item[1]) for rank, item in enumerate(bm25_results)}
        semantic_ranks = {item[0].chunk_id: (rank + 1, item[1]) for rank, item in enumerate(semantic_results)}
        
        all_chunk_ids = sorted(set(bm25_ranks.keys()).union(set(semantic_ranks.keys())), reverse=True)
        if not all_chunk_ids:
            return []
            
        # Grandfathering query check: Strictly checks for delivery before 1 August 2026
        # per pol_returns_v4.md [RET-3] ("Electronics delivered before 1 August 2026 keep 10-day window")
        is_grandfathered_delivery = bool(
            re.search(r"delivered\s+before\s+(?:august|aug|1\s+aug)|delivery\s+before\s+(?:august|aug)|v3|version\s+3", query, re.IGNORECASE)
        )
        
        scored_chunks = []
        has_authoritative_policy = False
        
        for cid in all_chunk_ids:
            chunk = self.chunk_map[cid]
            b_rank, b_score = bm25_ranks.get(cid, (1000, 0.0))
            s_rank, s_score = semantic_ranks.get(cid, (1000, 0.0))
            
            # Reciprocal Rank Fusion
            rrf_score = (bm25_weight / (rrf_k + b_rank)) + (semantic_weight / (rrf_k + s_rank))
            
            # Hard Conflict Filter 1: Superseded policies (KB-00.1 & RET-3)
            # Unless explicitly inquiring about electronics delivered before 1 August 2026,
            # superseded policy chunks cannot override current policy
            if chunk.status == "superseded" and not is_grandfathered_delivery:
                continue  # Exclude superseded policy from active retrieval
                
            # Track presence of authoritative policy / guide
            if chunk.authority_tier in (1, 2) and chunk.doc_type != "catalog":
                has_authoritative_policy = True
                
            authority_multiplier = {
                1: 1.25,  # Current Policy & Meta
                2: 1.15,  # Agent Guide & Catalog
                3: 1.05,  # Help Article
                4: 1.00,  # FAQ
                5: 0.60,  # Marketing (subordinated to policy)
                6: 1.20 if is_grandfathered_delivery else 0.20,  # Grandfathered v3
            }.get(chunk.authority_tier, 1.0)
            
            final_score = rrf_score * authority_multiplier
            scored_chunks.append((chunk, final_score, b_score, s_score))
            
        # Hard Conflict Filter 2: Marketing Demotion (KB-00.1 Rule 5)
        # If authoritative current policies exist in candidates, ensure marketing claims
        # (Tier 5) never outrank authoritative policies on the same query
        if has_authoritative_policy:
            policy_scores = [sc[1] for sc in scored_chunks if sc[0].authority_tier in (1, 2) and sc[0].doc_type != "catalog"]
            min_policy_score = min(policy_scores) if policy_scores else 0.0
            adjusted_scored_chunks = []
            for chunk, f_score, b_sc, s_sc in scored_chunks:
                if chunk.authority_tier == 5 and f_score >= min_policy_score:
                    # Enforce that marketing is ranked below authoritative policy
                    f_score = min_policy_score * 0.75
                adjusted_scored_chunks.append((chunk, f_score, b_sc, s_sc))
            scored_chunks = adjusted_scored_chunks

        scored_chunks.sort(key=lambda x: (x[1], x[0].chunk_id), reverse=True)
        
        results: List[RetrievalResult] = []
        catalog_count = 0
        max_catalog_items = 1  # Cap redundant catalog rows to preserve policy slots
        
        for chunk, final_score, b_score, s_score in scored_chunks:
            if chunk.doc_type == "catalog":
                if catalog_count >= max_catalog_items:
                    continue
                catalog_count += 1
                
            results.append(RetrievalResult(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                section_id=chunk.section_id,
                section_title=chunk.section_title,
                content=chunk.content,
                doc_type=chunk.doc_type,
                status=chunk.status,
                authority_tier=chunk.authority_tier,
                score=round(float(final_score), 5),
                rank=len(results) + 1,
                bm25_score=round(float(b_score), 4),
                dense_score=round(float(s_score), 4),
                backend=self.semantic_searcher.backend
            ))
            if len(results) >= top_k:
                break
            
        return results
