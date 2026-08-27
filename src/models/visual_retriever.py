from typing import List, Dict, Tuple, Optional
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F

from src.models.base import BaseRetriever
from src.models.maxsim import maxsim_pytorch, rank_documents_maxsim


def _concat_padded_embeddings(batches: List[torch.Tensor]) -> torch.Tensor:
    """Pads token sequences to a shared length before concatenating batches."""
    if not batches:
        raise ValueError("Cannot concatenate an empty embedding batch list.")

    max_length = max(batch.shape[1] for batch in batches)
    padded_batches = [
        F.pad(batch, (0, 0, 0, max_length - batch.shape[1]))
        for batch in batches
    ]
    return torch.cat(padded_batches, dim=0)


class ColPaliVisualRetriever(BaseRetriever):
    """
    Multimodal Late Interaction Retriever (ColPali / ColQwen2) for Visual Document Retrieval.
    Uses colpali_engine as the primary backend. Falls back to a direct
    HuggingFace implementation only if colpali_engine is unavailable, with
    proper handling for PaliGemmaProcessor's query-encoding quirks.
    """

    def __init__(
        self,
        model_name_or_path: str = "vidore/colpali-v1.2",
        revision: Optional[str] = None,
        adapter_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype: torch.dtype = torch.bfloat16
        if torch.cuda.is_available()
        else torch.float32,
    ):
        self.device = device
        self.torch_dtype = torch_dtype
        self.model_name = model_name_or_path
        self.revision = revision

        # --- Primary: Sentence Transformers MultiVectorEncoder ---
        try:
            from sentence_transformers import MultiVectorEncoder

            print(
                f"[*] Loading ColPali via Sentence Transformers from "
                f"'{model_name_or_path}'..."
            )
            model_kwargs = {"dtype": self.torch_dtype}
            if revision:
                model_kwargs["revision"] = revision
            self.model = MultiVectorEncoder(
                model_name_or_path,
                device=self.device,
                model_kwargs=model_kwargs,
            )
            self.processor = None
            self.backend = "sentence_transformers"
            print("[*] Sentence Transformers backend loaded successfully.")

        except ImportError:
            try:
                from colpali_engine.models import ColPali, ColPaliProcessor

                print(
                    "[!] MultiVectorEncoder unavailable. Falling back to the "
                    "deprecated colpali_engine backend."
                )
                self.processor = ColPaliProcessor.from_pretrained(model_name_or_path, revision=revision)
                self.model = ColPali.from_pretrained(
                    model_name_or_path,
                    torch_dtype=self.torch_dtype,
                    device_map=self.device,
                    revision=revision,
                )
                self.backend = "colpali_engine"
            except ImportError:
                print("[!] Falling back to raw transformers backend.")
                from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

                self.processor = AutoProcessor.from_pretrained(
                    model_name_or_path, trust_remote_code=True, revision=revision
                )
                self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                    model_name_or_path,
                    torch_dtype=self.torch_dtype,
                    trust_remote_code=True,
                    revision=revision,
                ).to(self.device)
                self.backend = "transformers_fallback"
                print(
                    "[!] Warning: transformers fallback has limited "
                    "query-encoding support."
                )

        # Optional LoRA adapter
        if adapter_path:
            if self.backend == "sentence_transformers":
                raise ValueError(
                    "External adapter_path is not supported by the "
                    "MultiVectorEncoder backend."
                )
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)

        self.model.eval()
        self.corpus_page_ids: List[str] = []
        self.corpus_embeddings: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode_queries(self, queries: List[str], batch_size: int = 4) -> torch.Tensor:
        """Encodes queries into token multi-vector representations (B, N_q, D)."""
        if self.backend == "sentence_transformers":
            embeddings = self.model.encode_query(
                queries,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=False,
            )
            if isinstance(embeddings, torch.Tensor):
                return embeddings.cpu()
            return _concat_padded_embeddings(
                [embedding.unsqueeze(0).cpu() for embedding in embeddings]
            )
        elif self.backend == "colpali_engine":
            return self._encode_queries_engine(queries, batch_size)
        else:
            return self._encode_queries_fallback(queries, batch_size)

    def _encode_queries_engine(
        self, queries: List[str], batch_size: int
    ) -> torch.Tensor:
        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(queries), batch_size):
                batch = queries[i : i + batch_size]
                inputs = self.processor.process_queries(batch).to(self.device)
                embeds = self.model(**inputs)
                # ColPali returns embeddings directly (not ModelOutput)
                if hasattr(embeds, "last_hidden_state"):
                    embeds = embeds.last_hidden_state
                all_embeddings.append(embeds.cpu())
        return _concat_padded_embeddings(all_embeddings)

    def _encode_queries_fallback(
        self, queries: List[str], batch_size: int
    ) -> torch.Tensor:
        """
        PaliGemmaProcessor requires images even for text-only inputs.
        We supply a minimal 1×1 white dummy image to satisfy that constraint,
        then extract hidden states from the language model backbone.
        """
        dummy_image = Image.new("RGB", (16, 16), color=(255, 255, 255))
        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(queries), batch_size):
                batch = queries[i : i + batch_size]
                dummy_images = [dummy_image] * len(batch)
                inputs = self.processor(
                    text=batch,
                    images=dummy_images,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)
                outputs = self.model(**inputs, output_hidden_states=True)
                # Use last hidden state as multi-vector representation
                embeds = outputs.hidden_states[-1]
                all_embeddings.append(embeds.cpu())
        return _concat_padded_embeddings(all_embeddings)

    def encode_documents(
        self, images: List[Image.Image], batch_size: int = 2
    ) -> torch.Tensor:
        """Encodes PIL images into patch multi-vector representations (B, N_d, D)."""
        if self.backend == "sentence_transformers":
            embeddings = self.model.encode_document(
                images,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=False,
            )
            if isinstance(embeddings, torch.Tensor):
                return embeddings.cpu()
            lengths = {embedding.shape[0] for embedding in embeddings}
            if len(lengths) != 1:
                raise ValueError(
                    "Document token lengths differ. MaxSim requires document masks "
                    "before variable-length document embeddings can be batched."
                )
            return torch.stack([embedding.cpu() for embedding in embeddings])
        elif self.backend == "colpali_engine":
            return self._encode_documents_engine(images, batch_size)
        else:
            return self._encode_documents_fallback(images, batch_size)

    def _encode_documents_engine(
        self, images: List[Image.Image], batch_size: int
    ) -> torch.Tensor:
        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i : i + batch_size]
                inputs = self.processor.process_images(batch).to(self.device)
                embeds = self.model(**inputs)
                if hasattr(embeds, "last_hidden_state"):
                    embeds = embeds.last_hidden_state
                all_embeddings.append(embeds.cpu())
        return torch.cat(all_embeddings, dim=0)

    def _encode_documents_fallback(
        self, images: List[Image.Image], batch_size: int
    ) -> torch.Tensor:
        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i : i + batch_size]
                inputs = self.processor(
                    images=batch, return_tensors="pt"
                ).to(self.device)
                outputs = self.model(**inputs, output_hidden_states=True)
                embeds = outputs.hidden_states[-1]
                all_embeddings.append(embeds.cpu())
        return torch.cat(all_embeddings, dim=0)

    # ------------------------------------------------------------------
    # Corpus indexing & retrieval
    # ------------------------------------------------------------------

    def index_corpus_from_images(
        self,
        corpus_page_ids: List[str],
        image_paths: List[str],
        batch_size: int = 2,
    ):
        """Loads and encodes all images in the corpus into multi-vector embeddings."""
        self.corpus_page_ids = corpus_page_ids
        print(f"[*] Loading {len(image_paths)} images from disk...")
        images = [Image.open(p).convert("RGB") for p in image_paths]
        print(f"[*] Encoding {len(images)} images (batch_size={batch_size})...")
        self.corpus_embeddings = self.encode_documents(images, batch_size=batch_size)
        print(
            f"[*] Corpus indexed: embeddings shape = {tuple(self.corpus_embeddings.shape)}"
        )

    def retrieve(
        self,
        queries: List[str],
        query_ids: Optional[List[str]] = None,
        top_k: int = 10,
        batch_size: int = 4,
    ) -> Dict[str, List[Tuple[str, float]]]:
        if self.corpus_embeddings is None:
            raise ValueError(
                "Corpus embeddings not computed. Call index_corpus_from_images() first."
            )

        if query_ids is None:
            query_ids = [f"q_{i}" for i in range(len(queries))]

        print(f"[*] Encoding {len(queries)} queries (batch_size={batch_size})...")
        query_embeds = self.encode_queries(queries, batch_size=batch_size).to(self.device)

        print("[*] Running MaxSim retrieval...")
        top_scores, top_indices = rank_documents_maxsim(
            query_embeddings=query_embeds,
            corpus_embeddings=self.corpus_embeddings,
            top_k=top_k,
        )

        results: Dict[str, List[Tuple[str, float]]] = {}
        for i, q_id in enumerate(query_ids):
            results[q_id] = [
                (self.corpus_page_ids[idx.item()], float(top_scores[i][j].item()))
                for j, idx in enumerate(top_indices[i])
            ]
        return results
