from typing import List, Dict, Set, Optional, Tuple
import random
from src.data.schema import QueryItem, PageMetadata
from src.models.text_baselines import BM25Retriever

class HardNegativeMiner:
    """
    Mines hard negative pages for retrieval adaptation:
    - Same-document / Same-PDF non-relevant pages (prevents template shortcuts).
    - Cross-document lexical hard negatives (BM25 false positives).
    """
    def __init__(
        self,
        corpus_pages: List[PageMetadata],
        bm25_retriever: Optional[BM25Retriever] = None,
    ):
        self.corpus_pages = corpus_pages
        self.page_dict = {p.page_id: p for p in corpus_pages}
        # Group page_ids by doc_id
        self.doc_to_pages: Dict[str, List[str]] = {}
        for p in corpus_pages:
            self.doc_to_pages.setdefault(p.doc_id, []).append(p.page_id)

        self.bm25_retriever = bm25_retriever

    def mine_same_doc_negatives(
        self,
        query: QueryItem,
        num_negatives: int = 2,
    ) -> List[str]:
        """Mines pages from the same PDF as the positive page that are NOT in target_page_ids."""
        pos_pages = set(query.target_page_ids)
        candidate_negatives: Set[str] = set()

        for pos_id in pos_pages:
            pos_meta = self.page_dict.get(pos_id)
            if pos_meta:
                doc_pages = self.doc_to_pages.get(pos_meta.doc_id, [])
                for pid in doc_pages:
                    if pid not in pos_pages:
                        candidate_negatives.add(pid)

        candidates = list(candidate_negatives)
        if len(candidates) <= num_negatives:
            return candidates
        return random.sample(candidates, num_negatives)

    def mine_lexical_negatives(
        self,
        query: QueryItem,
        top_k: int = 20,
        num_negatives: int = 3,
    ) -> List[str]:
        """Mines top BM25 retrieved pages that are false positives."""
        if self.bm25_retriever is None:
            return []

        pos_pages = set(query.target_page_ids)
        res = self.bm25_retriever.retrieve([query.query_text], query_ids=[query.query_id], top_k=top_k)
        retrieved_pairs = res.get(query.query_id, [])

        lexical_negs = [
            pid for pid, _ in retrieved_pairs if pid not in pos_pages and pid in self.page_dict
        ]
        return lexical_negs[:num_negatives]

    def mine_negatives_for_query(
        self,
        query: QueryItem,
        total_negatives: int = 5,
        same_doc_ratio: float = 0.4,
    ) -> List[str]:
        """
        Combines same-document negatives and lexical negatives into a balanced hard-negative set.
        """
        num_same_doc = max(1, int(total_negatives * same_doc_ratio))
        num_lexical = total_negatives - num_same_doc

        same_doc_negs = self.mine_same_doc_negatives(query, num_negatives=num_same_doc)
        lexical_negs = self.mine_lexical_negatives(query, num_negatives=num_lexical)

        combined = list(dict.fromkeys(same_doc_negs + lexical_negs))

        # If not enough, fill with random negatives from the entire corpus
        if len(combined) < total_negatives:
            all_pids = list(self.page_dict.keys())
            pos_pages = set(query.target_page_ids)
            remaining_needed = total_negatives - len(combined)
            random_pool = [pid for pid in all_pids if pid not in pos_pages and pid not in combined]
            if random_pool:
                sampled = random.sample(random_pool, min(remaining_needed, len(random_pool)))
                combined.extend(sampled)

        return combined[:total_negatives]

