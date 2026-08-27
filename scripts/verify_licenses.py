#!/usr/bin/env python3
"""
Verify license status for test PDFs by extracting license text from PDFs.
Checks for CC BY, CC0, public domain, or explicit redistribution permission.
"""

import csv
import hashlib
import re
from pathlib import Path
import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "data" / "governance" / "document_registry.csv"

# 6 test PDFs needing verification
TEST_DOCS = [
    "05_mang_may_tinh",
    "07_cam_nang_chuyen_doi_so",
    "08_he_dieu_hanh_unix_linux",
    "tl4_2021",
    "tl4_2021_1",
    "tongluan9_2024",
]

LICENSE_PATTERNS = [
    (r"CC\s*BY\s*3\.0\s*IGO", "CC BY 3.0 IGO"),
    (r"CC\s*BY\s*4\.0", "CC BY 4.0"),
    (r"CC\s*BY\s*3\.0", "CC BY 3.0"),
    (r"CC\s*BY\s*2\.0", "CC BY 2.0"),
    (r"CC\s*BY\s*1\.0", "CC BY 1.0"),
    (r"CC\s*0|CC\s*ZERO|public domain", "CC0 / Public Domain"),
    (r"Creative\s*Commons\s*Attribution", "Creative Commons Attribution"),
    (r"Open\s*Access", "Open Access"),
    (r"World\s*Bank.*license", "World Bank License"),
    (r"©.*World\s*Bank", "World Bank Copyright"),
    (r"NASATI|Viện.*Khoa\s*học.*Công\s*nghệ", "NASATI/Vietnam Gov"),
]

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def extract_text_from_pdf(pdf_path: Path, max_pages: int = 5) -> str:
    """Extract text from first N pages (license usually on first/last pages)."""
    text_parts = []
    try:
        doc = fitz.open(pdf_path)
        pages_to_check = min(max_pages, len(doc))
        # Check first 3 and last 2 pages
        check_pages = list(range(min(3, pages_to_check))) + list(range(max(0, len(doc)-2), len(doc)))
        check_pages = sorted(set(check_pages))
        for i in check_pages:
            page = doc[i]
            text_parts.append(page.get_text())
        doc.close()
    except Exception as e:
        return f"ERROR: {e}"
    return "\n".join(text_parts)

def detect_license(text: str) -> tuple[str, str]:
    """Return (license_name, evidence_snippet)."""
    text_lower = text.lower()
    for pattern, name in LICENSE_PATTERNS:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            m = matches[0]
            start = max(0, m.start() - 100)
            end = min(len(text), m.end() + 100)
            snippet = text[start:end].replace("\n", " ")
            return name, snippet
    return "unknown", ""

def main():
    print("=" * 80)
    print("LICENSE VERIFICATION FOR 6 TEST PDFs")
    print("=" * 80)

    # Load registry
    rows = []
    with REGISTRY.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    results = []
    for doc_id in TEST_DOCS:
        row = next((r for r in rows if r["doc_id"] == doc_id), None)
        if not row:
            print(f"[MISSING] {doc_id} not in registry")
            continue

        pdf_path = ROOT / row["local_pdf_path"]
        if not pdf_path.exists():
            print(f"[NOT FOUND] {doc_id}: {pdf_path}")
            continue

        print(f"\n--- {doc_id} ---")
        print(f"  Path: {pdf_path}")
        print(f"  Domain: {row['domain']}")
        print(f"  Publisher: {row['publisher']}")
        print(f"  Current license_status: {row['license_status']}")

        # SHA256
        file_hash = sha256_file(pdf_path)
        print(f"  SHA256: {file_hash}")

        # Extract text
        text = extract_text_from_pdf(pdf_path)
        if text.startswith("ERROR"):
            print(f"  Text extraction failed: {text}")
            results.append((doc_id, "extraction_failed", "", file_hash))
            continue

        # Detect license
        license_name, snippet = detect_license(text)
        print(f"  Detected license: {license_name}")
        if snippet:
            print(f"  Evidence: ...{snippet}...")

        # Check if verifiable
        verifiable = license_name != "unknown"
        redistribution = "true" if verifiable and license_name in ["CC BY 3.0 IGO", "CC BY 4.0", "CC BY 3.0", "CC0 / Public Domain", "Creative Commons Attribution", "Open Access"] else "false"

        results.append((doc_id, license_name, snippet, file_hash, verifiable, redistribution))

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'doc_id':<35} {'license':<25} {'verifiable':<12} {'redist'}")
    print("-" * 80)
    for doc_id, license_name, _, _, verifiable, redistribution in results:
        print(f"{doc_id:<35} {license_name:<25} {str(verifiable):<12} {redistribution}")

    # Save detailed report
    report_path = ROOT / "data" / "benchmark_governed_v0_1" / "license_verification_report.json"
    import json
    report = {
        "verified_at": "2026-08-27",
        "results": [
            {
                "doc_id": doc_id,
                "detected_license": license_name,
                "evidence_snippet": snippet,
                "sha256": file_hash,
                "verifiable": verifiable,
                "redistribution_allowed": redistribution,
            }
            for doc_id, license_name, snippet, file_hash, verifiable, redistribution in results
        ]
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[+] Detailed report saved to {report_path}")

    # Instructions for manual update
    print("\n" + "=" * 80)
    print("NEXT STEPS (MANUAL)")
    print("=" * 80)
    for doc_id, license_name, _, _, verifiable, redistribution in results:
        if not verifiable:
            row = next(r for r in rows if r["doc_id"] == doc_id)
            print(f"\n{doc_id}:")
            print(f"  - Current: license_status=unverified, redistribution_allowed=false, page_scope=first_10")
            print(f"  - Action needed: Find source URL, verify license, get redistribution permission")
            print(f"  - If CC BY found: Update registry -> license_status=verified, redistribution_allowed=true, page_scope=full")
            print(f"  - If NOT verifiable: Replace document or move to train/dev (not test)")

if __name__ == "__main__":
    main()