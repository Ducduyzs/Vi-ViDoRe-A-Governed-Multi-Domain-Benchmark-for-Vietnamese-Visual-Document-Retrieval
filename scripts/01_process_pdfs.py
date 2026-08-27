import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
from tqdm import tqdm

from src.config import PathConfig, ProcessingConfig
from src.data.schema import DomainType
from src.data.pdf_processor import PDFProcessor
from src.data.deduplication import DatasetAuditor

def infer_domain(pdf_path: Path) -> DomainType:
    """Infers document domain from folder name or filename keywords."""
    path_str = str(pdf_path).lower()
    name = pdf_path.stem.lower()

    if any(k in path_str for k in ["legal", "luat", "nghidinh", "thongtu", "tl4", "tongluan"]):
        return DomainType.LEGAL
    elif any(k in path_str for k in ["fin", "taichinh", "bctc", "ketoan", "nganhang", "kinhte"]):
        return DomainType.FINANCIAL
    elif any(k in path_str for k in ["health", "yte", "duoc", "thuoc", "benh"]):
        return DomainType.HEALTHCARE
    elif any(k in path_str for k in ["info", "chart", "thongke", "bieudo"]):
        return DomainType.INFOGRAPHIC
    elif any(k in path_str for k in ["tri_tue", "otomat", "cau_truc", "co_so_du_lieu", "mang_may_tinh", "dien_toan", "cam_nang", "he_dieu_hanh", "an_toan_thong_tin", "nhap_mon", "1706", "1810", "1512", "1406", "1409", "1206"]):
        return DomainType.EDUCATION
    return DomainType.EDUCATION

def main():
    parser = argparse.ArgumentParser(description="Step 1: Process raw PDFs into page images and metadata.")
    parser.add_argument("--data_dir", type=str, default=None, help="Root directory containing PDFs")
    parser.add_argument("--dpi", type=int, default=150, help="Target rendering DPI")
    parser.add_argument("--max_pages_per_doc", type=int, default=50, help="Max pages to process per PDF to balance dataset")
    args = parser.parse_args()

    paths = PathConfig()
    paths.make_dirs()

    data_dir = Path(args.data_dir) if args.data_dir else paths.data_dir
    print(f"[*] Scanning for PDFs in: {data_dir}")

    pdf_files = list(data_dir.glob("**/*.pdf"))
    if not pdf_files:
        print(f"[!] No PDF files found in {data_dir}.")
        return

    print(f"[*] Found {len(pdf_files)} PDF files to process:")
    for f in pdf_files[:10]:
        print(f"    - {f.relative_to(paths.root_dir)}")
    if len(pdf_files) > 10:
        print(f"    ... and {len(pdf_files) - 10} more files.")

    processor = PDFProcessor(
        output_image_dir=paths.pages_dir,
        target_dpi=args.dpi,
    )

    all_pages_meta = []
    for pdf_file in tqdm(pdf_files, desc="Rendering PDF pages"):
        domain = infer_domain(pdf_file)
        try:
            pages = processor.process_pdf(
                pdf_file,
                domain=domain,
                max_pages=args.max_pages_per_doc,
            )
            all_pages_meta.extend(pages)
        except Exception as e:
            print(f"[!] Error processing {pdf_file.name}: {e}")

    print(f"[+] Successfully processed {len(all_pages_meta)} pages from {len(pdf_files)} documents.")

    # Audit duplicates
    auditor = DatasetAuditor()
    pages_dict_list = [p.to_dict() for p in all_pages_meta]
    exact_dups, near_dups = auditor.find_duplicates(pages_dict_list)

    print(f"[*] Duplication Audit:")
    print(f"    - Exact duplicate clusters (SHA-256): {len(exact_dups)}")
    print(f"    - Near duplicate pairs (pHash <= {auditor.phash_threshold}): {len(near_dups)}")

    # Save metadata
    output_meta_path = paths.processed_dir / "all_pages_metadata.jsonl"
    with open(output_meta_path, "w", encoding="utf-8") as f:
        for p in all_pages_meta:
            f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")

    print(f"[+] Page metadata saved to: {output_meta_path}")

if __name__ == "__main__":
    main()
