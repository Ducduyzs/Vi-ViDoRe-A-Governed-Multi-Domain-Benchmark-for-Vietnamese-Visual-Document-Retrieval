from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
from PIL import Image
import numpy as np
import torch

class BaseRetriever(ABC):
    """
    Abstract interface for all retrievers (Text BM25, Dense Bi-Encoder, Multimodal Late Interaction).
    """
    @abstractmethod
    def encode_queries(self, queries: List[str]) -> Any:
        """Encodes a list of query strings into vector representations."""
        pass

    @abstractmethod
    def encode_documents(self, documents: List[Any]) -> Any:
        """
        Encodes a list of document items (can be text strings or PIL Images) into representations.
        """
        pass

    @abstractmethod
    def retrieve(
        self,
        queries: List[str],
        corpus_page_ids: List[str],
        top_k: int = 10,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Performs retrieval for a list of queries over the pre-indexed corpus.
        Returns:
            Dict mapping query_id -> List of (page_id, score) sorted descending by score.
        """
        pass

