import os
from pathlib import Path
from typing import List, Optional, Tuple
from PIL import Image, ImageFilter
import numpy as np
import pypdfium2 as pdfium
from pypdf import PdfReader

from src.data.schema import PageMetadata, DomainType, PageType, DocumentSourceType
from src.data.deduplication import compute_file_sha256, compute_image_phash

class PDFProcessor:
    """
    Renders PDF documents into high-quality images, extracts native text,
    and computes image quality metrics.
    """
    def __init__(
        self,
        output_image_dir: Path,
        target_dpi: int = 150,
        max_image_dim: int = 1024,
        min_text_len_for_born_digital: int = 50,
    ):
        self.output_image_dir = Path(output_image_dir)
        self.output_image_dir.mkdir(parents=True, exist_ok=True)
        self.target_dpi = target_dpi
        self.max_image_dim = max_image_dim
        self.min_text_len = min_text_len_for_born_digital

    def compute_blur_score(self, image: Image.Image) -> float:
        """
        Calculates image sharpness score based on edge gradient variance.
        Higher score = sharper image.
        """
        gray = image.convert("L")
        edges = gray.filter(ImageFilter.FIND_EDGES)
        arr = np.array(edges, dtype=np.float32)
        return float(np.var(arr))

    def detect_page_type(self, text: str, image: Image.Image) -> PageType:
        """
        Heuristic classification for page type (Text, Table, Chart, Mixed).
        """
        text_len = len(text.strip())
        # Check for table characteristics: multiple vertical bars, tabs, or repeated digits
        digit_count = sum(c.isdigit() for c in text)
        line_count = text.count("\n") + 1

        if text_len > 800 and (digit_count / (text_len + 1)) < 0.15:
            return PageType.TEXT_HEAVY
        elif digit_count > 100 or "\t" in text or "  " in text and line_count > 15:
            return PageType.TABLE_HEAVY
        elif text_len < 150:
            return PageType.CHART_HEAVY
        else:
            return PageType.MIXED

    def process_pdf(
        self,
        pdf_path: Path,
        doc_id: Optional[str] = None,
        domain: DomainType = DomainType.LEGAL,
        max_pages: Optional[int] = None,
    ) -> List[PageMetadata]:
        """
        Processes a single PDF: extracts pages, native text, renders images,
        computes hashes and metadata.
        """
        pdf_path = Path(pdf_path)
        if doc_id is None:
            doc_id = pdf_path.stem

        file_sha256 = compute_file_sha256(pdf_path)
        pdf_doc = pdfium.PdfDocument(str(pdf_path))
        total_pages = len(pdf_doc)
        num_pages_to_process = min(total_pages, max_pages) if max_pages else total_pages

        # Also open with PdfReader for fast text extraction
        pdf_reader = None
        try:
            pdf_reader = PdfReader(str(pdf_path))
        except Exception:
            pass

        processed_pages: List[PageMetadata] = []

        for page_idx in range(num_pages_to_process):
            page_num = page_idx + 1
            page_id = f"{doc_id}_p{page_num:03d}"

            # 1. Render image using pdfium (target DPI = scale factor)
            scale = self.target_dpi / 72.0
            page = pdf_doc[page_idx]
            pil_image = page.render(scale=scale).to_pil()

            # Resize if exceeds max dimension
            if max(pil_image.size) > self.max_image_dim:
                pil_image.thumbnail((self.max_image_dim, self.max_image_dim), Image.Resampling.LANCZOS)

            # Save image
            img_filename = f"{page_id}.png"
            img_path = self.output_image_dir / img_filename
            pil_image.save(img_path, format="PNG", optimize=True)

            # 2. Extract native text
            native_text = ""
            if pdf_reader and page_idx < len(pdf_reader.pages):
                try:
                    native_text = pdf_reader.pages[page_idx].extract_text() or ""
                except Exception:
                    native_text = ""

            char_count = len(native_text.strip())
            source_type = (
                DocumentSourceType.BORN_DIGITAL
                if char_count >= self.min_text_len
                else DocumentSourceType.SCANNED
            )

            # 3. Compute metrics
            phash = compute_image_phash(pil_image)
            blur_score = self.compute_blur_score(pil_image)
            page_type = self.detect_page_type(native_text, pil_image)

            meta = PageMetadata(
                doc_id=doc_id,
                page_num=page_num,
                page_id=page_id,
                file_path=str(pdf_path.resolve()),
                image_path=str(img_path.resolve()),
                sha256=file_sha256,
                phash=phash,
                domain=domain,
                page_type=page_type,
                source_type=source_type,
                native_text=native_text,
                char_count=char_count,
                estimated_dpi=self.target_dpi,
                blur_score=blur_score,
            )
            processed_pages.append(meta)

        return processed_pages

