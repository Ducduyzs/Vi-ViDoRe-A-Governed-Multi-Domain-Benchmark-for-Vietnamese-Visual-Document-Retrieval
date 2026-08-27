from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import os

@dataclass
class PathConfig:
    root_dir: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = root_dir / "data"
    raw_pdf_dir: Path = root_dir / "data" / "raw_pdfs"
    processed_dir: Path = root_dir / "data" / "processed"
    pages_dir: Path = processed_dir / "pages"
    benchmark_dir: Path = root_dir / "data" / "benchmark"
    checkpoints_dir: Path = root_dir / "checkpoints"
    results_dir: Path = root_dir / "results"
    local_config_file: Path = root_dir / "config.local.json"

    def make_dirs(self):
        for path in [
            self.data_dir,
            self.raw_pdf_dir,
            self.processed_dir,
            self.pages_dir,
            self.benchmark_dir,
            self.checkpoints_dir,
            self.results_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

@dataclass
class LLMConfig:
    provider: str = "openai"  # "openai" | "gemini"
    openai_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    gemini_model: str = "gemini-2.0-flash"
    temperature: float = 0.3
    max_retries: int = 3
    requests_per_minute: int = 60

    @classmethod
    def load_from_file(cls, config_path: Path) -> "LLMConfig":
        config = cls()
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                config.provider = data.get("llm_provider", config.provider)
                config.openai_api_key = data.get("openai_api_key", os.getenv("OPENAI_API_KEY"))
                config.gemini_api_key = data.get("gemini_api_key", os.getenv("GEMINI_API_KEY"))
                config.openai_model = data.get("openai_model", config.openai_model)
                config.gemini_model = data.get("gemini_model", config.gemini_model)
            except Exception as e:
                print(f"[!] Warning: Could not read {config_path}: {e}")
        else:
            config.openai_api_key = os.getenv("OPENAI_API_KEY")
            config.gemini_api_key = os.getenv("GEMINI_API_KEY")
        return config

@dataclass
class ProcessingConfig:
    target_dpi: int = 150
    max_image_dim: int = 1024
    phash_threshold: int = 8  # Hamming distance threshold for near-duplicate images
    min_text_len: int = 20    # Minimum character count to classify as text-bearing page
    supported_domains: List[str] = field(
        default_factory=lambda: ["legal", "financial", "healthcare", "education", "infographic", "computer_science"]
    )

@dataclass
class ModelConfig:
    backbone_name: str = "vidore/colpali-v1.2"
    backbone_revision: Optional[str] = None  # Hugging Face model revision/commit hash
    device: str = "cuda"
    dtype: str = "bfloat16"
    embedding_dim: int = 128
    query_max_length: int = 64
    batch_size: int = 4
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

@dataclass
class TrainConfig:
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    num_epochs: int = 3
    gradient_accumulation_steps: int = 4
    temperature: float = 0.05
    num_hard_negatives: int = 5
    in_pdf_negative_weight: float = 1.5

@dataclass
class BenchmarkConfig:
    test_split_ratio: float = 0.6
    dev_split_ratio: float = 0.2
    train_split_ratio: float = 0.2
    top_k_eval: List[int] = field(default_factory=lambda: [1, 5, 10, 20])
    ci_bootstrap_samples: int = 1000
    ci_bootstrap_seed: int = 42
