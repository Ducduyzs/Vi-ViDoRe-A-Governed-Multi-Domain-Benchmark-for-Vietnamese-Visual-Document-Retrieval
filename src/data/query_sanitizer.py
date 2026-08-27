import re
from typing import List, Tuple, Set

# Vietnamese deictic / relative words that make a query non-retrievable in isolation
DEICTIC_PATTERNS = [
    r"\b(trang này|trang trên|trang dưới|trang bên)\b",
    r"\b(hình này|hình trên|hình dưới|hình bên|ảnh này|ảnh trên)\b",
    r"\b(bảng này|bảng trên|bảng dưới|bảng bên cạnh|bảng sau)\b",
    r"\b(sơ đồ này|sơ đồ trên|biểu đồ này|biểu đồ trên)\b",
    r"\b(văn bản này|tài liệu này|bài viết này|đoạn văn này)\b",
    r"\b(theo như bảng|như trong hình|trong bảng số liệu)\b",
    r"\b(tại đây|ở đây|sau đây|dưới đây)\b",
]

class QuerySanitizer:
    """
    Sanitizes and validates generated/written retrieval queries.
    Prevents lexical leakage, deictic references, and ambiguity.
    """
    def __init__(
        self,
        max_ngram_leakage: int = 5,
        min_words: int = 4,
        max_words: int = 40,
    ):
        self.max_ngram_leakage = max_ngram_leakage
        self.min_words = min_words
        self.max_words = max_words
        self.compiled_deictic = [re.compile(p, re.IGNORECASE) for p in DEICTIC_PATTERNS]

    def check_deictic_words(self, query: str) -> List[str]:
        """Returns list of matched deictic phrases that invalidate the query."""
        matches = []
        for pattern in self.compiled_deictic:
            found = pattern.findall(query)
            if found:
                matches.extend(found)
        return matches

    def clean_deictic_words(self, query: str) -> str:
        """Removes common deictic phrases from query."""
        cleaned = query
        for pattern in self.compiled_deictic:
            cleaned = pattern.sub("", cleaned)
        # Normalize whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def compute_lexical_leakage(self, query: str, document_text: str) -> float:
        """
        Computes the ratio of long query n-grams directly copied verbatim from document text.
        Leakage > 0.6 indicates the query is too easy / an exact text quote.
        """
        q_words = [w.lower() for w in re.findall(r"\w+", query)]
        d_words = [w.lower() for w in re.findall(r"\w+", document_text)]
        if len(q_words) < self.max_ngram_leakage or len(d_words) == 0:
            return 0.0

        d_ngrams: Set[str] = set()
        for i in range(len(d_words) - self.max_ngram_leakage + 1):
            d_ngrams.add(" ".join(d_words[i : i + self.max_ngram_leakage]))

        q_ngrams = []
        for i in range(len(q_words) - self.max_ngram_leakage + 1):
            q_ngrams.append(" ".join(q_words[i : i + self.max_ngram_leakage]))

        if not q_ngrams:
            return 0.0

        matched = sum(1 for ng in q_ngrams if ng in d_ngrams)
        return matched / len(q_ngrams)

    def validate_query(self, query: str, document_text: str = "") -> Tuple[bool, str]:
        """
        Runs complete validation rules.
        Returns: (is_valid, reason_or_status)
        """
        words = re.findall(r"\w+", query)
        word_count = len(words)

        if word_count < self.min_words:
            return False, f"TOO_SHORT: Only {word_count} words (min {self.min_words})"
        if word_count > self.max_words:
            return False, f"TOO_LONG: {word_count} words (max {self.max_words})"

        deictic = self.check_deictic_words(query)
        if deictic:
            return False, f"CONTAINS_DEICTIC_TERMS: {deictic}"

        if document_text:
            leakage = self.compute_lexical_leakage(query, document_text)
            if leakage >= 0.7:
                return False, f"HIGH_LEXICAL_LEAKAGE: {leakage:.2f} n-gram match ratio"

        return True, "VALID"

