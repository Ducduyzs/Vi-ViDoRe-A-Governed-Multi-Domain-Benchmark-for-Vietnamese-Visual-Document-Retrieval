from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset

from src.data.schema import QueryItem, PageMetadata
from src.training.hard_negative_miner import HardNegativeMiner

class ViViDoReDataset(Dataset):
    """
    PyTorch Dataset for training Multimodal Late Interaction models (ColPali/ColQwen2).
    Returns (query_text, positive_image, list_of_negative_images).
    """
    def __init__(
        self,
        queries: List[QueryItem],
        corpus_pages: List[PageMetadata],
        negative_miner: Optional[HardNegativeMiner] = None,
        num_hard_negatives: int = 3,
        transform=None,
    ):
        self.queries = queries
        self.corpus_pages = corpus_pages
        self.page_dict = {p.page_id: p for p in corpus_pages}
        self.negative_miner = negative_miner
        self.num_hard_negatives = num_hard_negatives
        self.transform = transform

    def __len__(self) -> int:
        return len(self.queries)

    def _load_image(self, page_id: str) -> Image.Image:
        page_meta = self.page_dict.get(page_id)
        if page_meta and Path(page_meta.image_path).exists():
            return Image.open(page_meta.image_path).convert("RGB")
        # Blank placeholder fallback
        return Image.new("RGB", (448, 448), color=(255, 255, 255))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        query_item = self.queries[idx]
        pos_page_id = query_item.target_page_ids[0]

        pos_image = self._load_image(pos_page_id)
        if self.transform:
            pos_image = self.transform(pos_image)

        neg_images = []
        neg_page_ids = []
        if self.negative_miner:
            neg_page_ids = self.negative_miner.mine_negatives_for_query(
                query_item, total_negatives=self.num_hard_negatives
            )
            for neg_id in neg_page_ids:
                img = self._load_image(neg_id)
                if self.transform:
                    img = self.transform(img)
                neg_images.append(img)

        return {
            "query_id": query_item.query_id,
            "query_text": query_item.query_text,
            "pos_page_id": pos_page_id,
            "pos_image": pos_image,
            "neg_page_ids": neg_page_ids,
            "neg_images": neg_images,
            "domain": query_item.domain.value,
        }

