from typing import List, Dict, Any, Union, Optional
from pathlib import Path
import json
import platform
import torch
import sys
from datetime import datetime


def _fmt_metric(val: Union[float, str, None], default: str = "N/A") -> str:
    """Format metric value, returning default for missing/NaN values."""
    if val is None or (isinstance(val, float) and val != val):  # NaN check
        return default
    if isinstance(val, float):
        return f"{val:.3f}"
    return str(val)


def _fmt_metric_with_count(val: Union[float, str, None], count: int, default: str = "N/A") -> str:
    """Format metric with sample count."""
    if val is None or (isinstance(val, float) and val != val) or count == 0:
        return f"{default} (n={count})"
    if isinstance(val, float):
        return f"{val:.3f} (n={count})"
    return f"{val} (n={count})"


def _fmt_latex_metric(val: Union[float, str, None], count: int, default: str = "N/A") -> str:
    """Format metric for LaTeX table with sample count, showing N/A for missing/zero-count slices."""
    if val is None or (isinstance(val, float) and val != val) or count == 0:
        return default
    if isinstance(val, float):
        return f"{val:.3f}"
    return str(val)


def _get_runtime_info() -> Dict[str, Any]:
    """Collect runtime environment information for reproducibility."""
    info = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
    return info


def generate_markdown_report(results_list: List[Dict[str, Any]]) -> str:
    """
    Generates a Markdown comparison table across multiple evaluated models.
    """
    md = []
    md.append("# 📊 Bảng Tổng hợp Kết quả Thực nghiệm Vi-ViDoRe Benchmark\n")
    md.append("| Model | Macro nDCG@5 | Overall nDCG@5 (95% CI) | Legal nDCG@5 | Financial nDCG@5 | Health nDCG@5 | Edu/Info nDCG@5 | MRR@10 |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for res in results_list:
        name = res["model_name"]
        macro = res.get("macro_domain_ndcg@5", 0.0)
        overall_ndcg5 = res["overall"]["ndcg@5"]["mean"]
        ci = res["overall"]["ndcg@5"]["ci_95"]
        ci_str = f"{overall_ndcg5:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"

        doms = res.get("by_domain", {})
        legal = doms.get("legal", {}).get("ndcg@5")
        legal_count = doms.get("legal", {}).get("count", 0)
        financial = doms.get("financial", {}).get("ndcg@5")
        financial_count = doms.get("financial", {}).get("count", 0)
        health = doms.get("healthcare", {}).get("ndcg@5")
        health_count = doms.get("healthcare", {}).get("count", 0)
        edu_val = doms.get("education", {}).get("ndcg@5", doms.get("infographic", {}).get("ndcg@5"))
        edu_count = doms.get("education", {}).get("count", doms.get("infographic", {}).get("count", 0))
        mrr = res["overall"]["mrr@10"]["mean"]

        md.append(
            f"| **{name}** | **{macro:.3f}** | {ci_str} | {_fmt_metric_with_count(legal, legal_count)} | "
            f"{_fmt_metric_with_count(financial, financial_count)} | {_fmt_metric_with_count(health, health_count)} | "
            f"{_fmt_metric_with_count(edu_val, edu_count)} | {mrr:.3f} |"
        )

    md.append("\n\n### Phân rã theo Nguồn tài liệu (Born-digital vs Scanned):")
    md.append(r"| Model | Born-digital nDCG@5 | Scanned nDCG@5 | Gap ($\Delta$) |")
    md.append("| :--- | :---: | :---: | :---: |")

    for res in results_list:
        name = res["model_name"]
        by_src = res.get("by_source_type", {})
        born_val = by_src.get("born_digital", {}).get("ndcg@5")
        born_count = by_src.get("born_digital", {}).get("count", 0)
        scanned_val = by_src.get("scanned", {}).get("ndcg@5")
        scanned_count = by_src.get("scanned", {}).get("count", 0)

        born_str = _fmt_metric_with_count(born_val, born_count)
        scanned_str = _fmt_metric_with_count(scanned_val, scanned_count)

        if isinstance(born_val, float) and isinstance(scanned_val, float) and born_count > 0 and scanned_count > 0:
            gap = born_val - scanned_val
            gap_str = f"{gap:+.3f}"
        else:
            gap_str = "N/A"

        md.append(f"| **{name}** | {born_str} | {scanned_str} | {gap_str} |")

    return "\n".join(md)


def generate_latex_table(results_list: List[Dict[str, Any]]) -> str:
    """
    Generates a camera-ready LaTeX table for ACL/IEEE submission.
    """
    latex = []
    latex.append(r"\begin{table*}[t]")
    latex.append(r"\centering")
    latex.append(r"\small")
    latex.append(r"\begin{tabular}{l c c c c c c}")
    latex.append(r"\toprule")
    latex.append(r"\textbf{Model} & \textbf{Macro nDCG@5} & \textbf{Legal} & \textbf{Finance} & \textbf{Health} & \textbf{Edu} & \textbf{MRR@10} \\")
    latex.append(r"\midrule")

    for res in results_list:
        name = res["model_name"]
        macro = res.get("macro_domain_ndcg@5", 0.0)
        doms = res.get("by_domain", {})
        legal = _fmt_latex_metric(doms.get("legal", {}).get("ndcg@5"), doms.get("legal", {}).get("count", 0))
        financial = _fmt_latex_metric(doms.get("financial", {}).get("ndcg@5"), doms.get("financial", {}).get("count", 0))
        health = _fmt_latex_metric(doms.get("healthcare", {}).get("ndcg@5"), doms.get("healthcare", {}).get("count", 0))
        edu = _fmt_latex_metric(doms.get("education", {}).get("ndcg@5", doms.get("infographic", {}).get("ndcg@5")), 
                                doms.get("education", {}).get("count", doms.get("infographic", {}).get("count", 0)))
        mrr = res["overall"]["mrr@10"]["mean"]

        latex.append(
            f"{name} & {macro:.3f} & {legal} & {financial} & {health} & {edu} & {mrr:.3f} \\\\"
        )

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\caption{Main retrieval results on Vi-ViDoRe benchmark across domains. Missing slices (n=0) shown as N/A.}")
    latex.append(r"\label{tab:main_results}")
    latex.append(r"\end{table*}")
    return "\n".join(latex)


def save_evaluation_report(
    results_list: List[Dict[str, Any]],
    output_dir: Path,
    report_name: str = "benchmark_report",
    run_metadata: Optional[Dict[str, Any]] = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Add runtime info to results
    runtime_info = _get_runtime_info()
    if run_metadata:
        runtime_info.update(run_metadata)

    enriched_results = {
        "metadata": runtime_info,
        "results": results_list,
    }

    # 1. JSON (with full metadata)
    with open(output_dir / f"{report_name}.json", "w", encoding="utf-8") as f:
        json.dump(enriched_results, f, ensure_ascii=False, indent=2)

    # 2. Markdown
    md_content = generate_markdown_report(results_list)
    with open(output_dir / f"{report_name}.md", "w", encoding="utf-8") as f:
        f.write(md_content)

    # 3. LaTeX
    latex_content = generate_latex_table(results_list)
    with open(output_dir / f"{report_name}.tex", "w", encoding="utf-8") as f:
        f.write(latex_content)
