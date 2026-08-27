from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import torch
from rank_bm25 import BM25Okapi
import re

from src.models.base import BaseRetriever

def tokenize_vietnamese(text: str) -> List[str]:
    """Tokenizes Vietnamese text using PyVi or regex fallback."""
    try:
        from pyvi import ViTokenizer
        segmented = ViTokenizer.tokenize(text.lower())
        return segmented.split()
    except Exception:
        try:
            from underthesea import word_tokenize
            tokens = word_tokenize(text.lower(), format="text").split()
            return tokens
        except Exception:
            return re.findall(r"\w+", text.lower())

class BM25Retriever(BaseRetriever):
    """
    Standard BM25 baseline over OCR/native document text with Vietnamese segmentation.
    """
    def __init__(self):
        self.corpus_page_ids: List[str] = []
        self.corpus_texts: List[str] = []
        self.bm25_index: Optional[BM25Okapi] = None

    def encode_queries(self, queries: List[str]) -> List[List[str]]:
        return [tokenize_vietnamese(q) for q in queries]

    def encode_documents(self, documents: List[str]) -> List[List[str]]:
        return [tokenize_vietnamese(d) for d in documents]

    def index_corpus(self, corpus_page_ids: List[str], corpus_texts: List[str]):
        self.corpus_page_ids = corpus_page_ids
        self.corpus_texts = corpus_texts
        tokenized_corpus = self.encode_documents(corpus_texts)
        self.bm25_index = BM25Okapi(tokenized_corpus)

    def retrieve(
        self,
        queries: List[str],
        query_ids: Optional[List[str]] = None,
        top_k: int = 10,
    ) -> Dict[str, List[Tuple[str, float]]]:
        if self.bm25_index is None:
            raise ValueError("Corpus has not been indexed yet! Call index_corpus first.")

        if query_ids is None:
            query_ids = [f"q_{i}" for i in range(len(queries))]

        results: Dict[str, List[Tuple[str, float]]] = {}
        for q_id, q_text in zip(query_ids, queries):
            tokenized_query = tokenize_vietnamese(q_text)
            scores = self.bm25_index.get_scores(tokenized_query)
            top_indices = np.argsort(scores)[::-1][:top_k]
            results[q_id] = [
                (self.corpus_page_ids[idx], float(scores[idx]))
                for idx in top_indices
            ]
        return results

class DenseBiEncoderRetriever(BaseRetriever):
    """
    Dense Single-Vector Bi-Encoder baseline (e.g. Vietnamese-Bi-Encoder or BGE-M3 text).
    """
    def __init__(
        self,
        model_name_or_path: str = "bkai-foundation-models/vietnamese-bi-encoder",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        from sentence_transformers import SentenceTransformer
        self.device = device
        self.model = SentenceTransformer(model_name_or_path, device=device)
        self.corpus_page_ids: List[str] = []
        self.corpus_embeddings: Optional[torch.Tensor] = None

    def encode_queries(self, queries: List[str]) -> torch.Tensor:
        embeddings = self.model.encode(
            queries,
            convert_to_tensor=True,
            show_progress_bar=False,
            device=self.device,
            normalize_embeddings=True,
        )
        return embeddings

    def encode_documents(self, documents: List[str]) -> torch.Tensor:
        embeddings = self.model.encode(
            documents,
            convert_to_tensor=True,
            show_progress_bar=False,
            device=self.device,
            normalize_embeddings=True,
        )
        return embeddings

    def index_corpus(self, corpus_page_ids: List[str], corpus_texts: List[str]):
        self.corpus_page_ids = corpus_page_ids
        self.corpus_embeddings = self.encode_documents(corpus_texts)

    def retrieve(
        self,
        queries: List[str],
        query_ids: Optional[List[str]] = None,
        top_k: int = 10,
    ) -> Dict[str, List[Tuple[str, float]]]:
        if self.corpus_embeddings is None:
            raise ValueError("Corpus has not been indexed yet! Call index_corpus first.")

        if query_ids is None:
            query_ids = [f"q_{i}" for i in range(len(queries))]

        query_embeddings = self.encode_queries(queries)
        # Cosine similarity: (N_queries, N_docs)
        cos_scores = torch.matmul(query_embeddings, self.corpus_embeddings.T)

        k = min(top_k, len(self.corpus_page_ids))
        top_scores, top_indices = torch.topk(cos_scores, k=k, dim=-1)

        results: Dict[str, List[Tuple[str, float]]] = {}
        for i, q_id in enumerate(query_ids):
            results[q_id] = [
                (self.corpus_page_ids[idx.item()], float(top_scores[i][j].item()))
                for j, idx in enumerate(top_indices[i])
            ]
        return results

