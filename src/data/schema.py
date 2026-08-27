from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Any
import json
from pathlib import Path

class DomainType(str, Enum):
    LEGAL = "legal"
    FINANCIAL = "financial"
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    INFOGRAPHIC = "infographic"
    OTHER = "other"

class PageType(str, Enum):
    TEXT_HEAVY = "text_heavy"
    TABLE_HEAVY = "table_heavy"
    CHART_HEAVY = "chart_heavy"
    MIXED = "mixed"
    FORM_OR_TEMPLATE = "form_or_template"

class DocumentSourceType(str, Enum):
    BORN_DIGITAL = "born_digital"
    SCANNED = "scanned"

class QueryType(str, Enum):
    FACT_LOOKUP = "fact_lookup"
    LEGAL_CLAUSE = "legal_clause"
    NUMERIC_TABLE = "numeric_table"
    MULTI_CELL_COMPARISON = "multi_cell_comparison"
    CHART_INTERPRETATION = "chart_interpretation"
    PARAPHRASE_OR_ABBREVIATION = "paraphrase_or_abbreviation"

@dataclass
class PageMetadata:
    doc_id: str
    page_num: int  # 1-indexed
    page_id: str   # f"{doc_id}_p{page_num}"
    file_path: str
    image_path: str
    sha256: str
    phash: str
    domain: DomainType
    page_type: PageType
    source_type: DocumentSourceType
    native_text: str = ""
    char_count: int = 0
    estimated_dpi: int = 150
    blur_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["domain"] = self.domain.value
        data["page_type"] = self.page_type.value
        data["source_type"] = self.source_type.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PageMetadata:
        data = data.copy()
        data["domain"] = DomainType(data["domain"])
        data["page_type"] = PageType(data["page_type"])
        data["source_type"] = DocumentSourceType(data["source_type"])
        return cls(**data)

@dataclass
class QueryItem:
    query_id: str
    query_text: str
    domain: DomainType
    query_type: QueryType
    source: str  # "human_written" | "llm_assisted"
    target_page_ids: List[str]
    hardness_level: str = "medium"  # "easy" | "medium" | "hard"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["domain"] = self.domain.value
        data["query_type"] = self.query_type.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> QueryItem:
        data = data.copy()
        data["domain"] = DomainType(data["domain"])
        data["query_type"] = QueryType(data["query_type"])
        return cls(**data)

@dataclass
class QrelItem:
    query_id: str
    page_id: str
    relevance: int  # 0: Not relevant, 1: Partially relevant, 2: Fully relevant
    comment: Optional[str] = None

    def to_tsv_row(self) -> str:
        return f"{self.query_id}\t0\t{self.page_id}\t{self.relevance}"

    @classmethod
    def from_tsv_row(cls, row: str) -> QrelItem:
        parts = row.strip().split("\t")
        if len(parts) >= 4:
            return cls(query_id=parts[0], page_id=parts[2], relevance=int(parts[3]))
        elif len(parts) == 3:
            return cls(query_id=parts[0], page_id=parts[1], relevance=int(parts[2]))
        raise ValueError(f"Invalid qrel row: {row}")

@dataclass
class BenchmarkSplit:
    name: str  # "train" | "dev" | "test"
    queries: List[QueryItem]
    corpus_page_ids: List[str]
    qrels: List[QrelItem]
    doc_sources: List[str]

    def save(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        # Save queries
        with open(output_dir / "queries.jsonl", "w", encoding="utf-8") as f:
            for q in self.queries:
                f.write(json.dumps(q.to_dict(), ensure_ascii=False) + "\n")
        # Save qrels
        with open(output_dir / "qrels.tsv", "w", encoding="utf-8") as f:
            for qrel in self.qrels:
                f.write(qrel.to_tsv_row() + "\n")
        # Save corpus page list
        with open(output_dir / "corpus_pages.json", "w", encoding="utf-8") as f:
            json.dump(self.corpus_page_ids, f, ensure_ascii=False, indent=2)

