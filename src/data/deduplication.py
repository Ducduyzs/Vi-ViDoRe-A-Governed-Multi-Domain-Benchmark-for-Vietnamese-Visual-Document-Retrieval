import hashlib
from pathlib import Path
from typing import Dict, List, Set, Tuple
from PIL import Image
import numpy as np

def compute_file_sha256(file_path: Path) -> str:
    """Computes SHA-256 hash of a file for exact duplicate detection."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def compute_image_phash(image: Image.Image, hash_size: int = 8) -> str:
    """
    Computes difference perceptual hash (dHash) for an image.
    Robust against resolution changes, light compression, and color shifts.
    """
    # Resize to (hash_size + 1, hash_size) and convert to grayscale
    img_gray = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    pixels = np.array(img_gray, dtype=np.float32)
    # Compare adjacent pixels
    difference = pixels[:, 1:] > pixels[:, :-1]
    # Convert bool array to hex string
    binary_str = "".join(["1" if b else "0" for b in difference.flatten()])
    return f"{int(binary_str, 2):0{hash_size * hash_size // 4}x}"

def hamming_distance(hash1: str, hash2: str) -> int:
    """Computes bitwise Hamming distance between two hex hashes."""
    int1 = int(hash1, 16)
    int2 = int(hash2, 16)
    return bin(int1 ^ int2).count("1")

class DatasetAuditor:
    """
    Audits dataset splits to prevent data leakage and template contamination.
    """
    def __init__(self, phash_distance_threshold: int = 8):
        self.phash_threshold = phash_distance_threshold

    def find_duplicates(
        self, pages: List[Dict[str, str]]
    ) -> Tuple[Dict[str, List[str]], List[Tuple[str, str, int]]]:
        """
        Detects exact duplicates (SHA-256) and near-duplicates (pHash).
        Returns:
            exact_duplicates: dict mapping sha256 to list of page_ids
            near_duplicates: list of (page_id_1, page_id_2, hamming_dist)
        """
        exact_map: Dict[str, List[str]] = {}
        for p in pages:
            sha = p["sha256"]
            exact_map.setdefault(sha, []).append(p["page_id"])

        exact_dups = {k: v for k, v in exact_map.items() if len(v) > 1}

        # Near duplicates pairwise check
        near_dups: List[Tuple[str, str, int]] = []
        n = len(pages)
        for i in range(n):
            for j in range(i + 1, n):
                p1, p2 = pages[i], pages[j]
                if p1["doc_id"] == p2["doc_id"]:
                    continue  # Skip checking pages inside the same document
                dist = hamming_distance(p1["phash"], p2["phash"])
                if dist <= self.phash_threshold:
                    near_dups.append((p1["page_id"], p2["page_id"], dist))

        return exact_dups, near_dups

    def audit_train_test_leakage(
        self, train_pages: List[Dict[str, str]], test_pages: List[Dict[str, str]]
    ) -> List[Tuple[str, str, str]]:
        """
        Checks if any test page is an exact or near-duplicate of a training page.
        Returns list of (train_page_id, test_page_id, reason).
        """
        leakages = []
        train_sha_set = {p["sha256"]: p["page_id"] for p in train_pages}
        train_docs = {p["doc_id"] for p in train_pages}

        for test_p in test_pages:
            # 1. Exact SHA check
            if test_p["sha256"] in train_sha_set:
                leakages.append(
                    (train_sha_set[test_p["sha256"]], test_p["page_id"], "EXACT_SHA256_MATCH")
                )
            # 2. Document overlap check
            if test_p["doc_id"] in train_docs:
                leakages.append((test_p["doc_id"], test_p["page_id"], "SAME_DOCUMENT_ID"))
            # 3. Near-duplicate check
            for train_p in train_pages:
                dist = hamming_distance(train_p["phash"], test_p["phash"])
                if dist <= self.phash_threshold:
                    leakages.append(
                        (train_p["page_id"], test_p["page_id"], f"NEAR_DUPLICATE_PHASH_DIST_{dist}")
                    )

        return leakages

