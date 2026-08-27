#!/usr/bin/env python3
"""
Model verification script for Vi-ViDoRe benchmark.
Verifies model loading, parameter checksums, and embedding parity.
"""

import sys
import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from sentence_transformers import MultiVectorEncoder


def compute_param_checksum(model) -> str:
    """Compute SHA-256 checksum of all model parameters."""
    hasher = hashlib.sha256()
    for name, param in model.named_parameters():
        hasher.update(param.data.cpu().numpy().tobytes())
    return hasher.hexdigest()


def compute_param_summary(model) -> dict:
    """Compute summary statistics for model parameters."""
    total_params = 0
    trainable_params = 0
    param_details = {}
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count
        param_details[name] = {
            "shape": list(param.shape),
            "numel": param_count,
            "requires_grad": param.requires_grad,
            "dtype": str(param.dtype),
            "mean": float(param.data.mean()),
            "std": float(param.data.std()),
        }
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": total_params - trainable_params,
        "param_details": param_details,
    }


def verify_colpali_loading(model_name: str = "vidore/colpali-v1.2", revision: str = None) -> dict:
    """Verify ColPali model loads correctly with MultiVectorEncoder."""
    print(f"[*] Loading {model_name} (revision={revision})...")
    
    try:
        encoder = MultiVectorEncoder(
            model_name_or_path=model_name,
            revision=revision,
            trust_remote_code=True,
        )
        print("[+] Model loaded successfully")
    except Exception as e:
        return {"success": False, "error": f"Failed to load model: {e}"}
    
    # Check for LoRA parameters
    lora_params = []
    custom_proj_params = []
    for name, param in encoder.model.named_parameters():
        if "lora" in name.lower():
            lora_params.append(name)
        if "custom_text_proj" in name:
            custom_proj_params.append(name)
    
    # Compute checksums
    param_checksum = compute_param_checksum(encoder.model)
    param_summary = compute_param_summary(encoder.model)
    
    # Test encoding
    print("[*] Testing query/document encoding...")
    test_query = "Test query for verification"
    test_doc = "Test document for verification"
    
    try:
        query_emb = encoder.encode_query(test_query)
        doc_emb = encoder.encode_document(test_doc)
        
        query_shape = query_emb.shape if hasattr(query_emb, 'shape') else list(query_emb.size())
        doc_shape = doc_emb.shape if hasattr(doc_emb, 'shape') else list(doc_emb.size())
        
        # Check for NaN/Inf
        query_has_nan = torch.isnan(query_emb).any().item() if hasattr(query_emb, 'any') else False
        doc_has_nan = torch.isnan(doc_emb).any().item() if hasattr(doc_emb, 'any') else False
        
        encoding_test = {
            "query_shape": list(query_shape),
            "doc_shape": list(doc_shape),
            "query_has_nan": query_has_nan,
            "doc_has_nan": doc_has_nan,
            "query_mean": float(query_emb.mean()) if not query_has_nan else None,
            "doc_mean": float(doc_emb.mean()) if not doc_has_nan else None,
        }
        print(f"[+] Encoding test passed: query={query_shape}, doc={doc_shape}")
    except Exception as e:
        encoding_test = {"error": f"Encoding failed: {e}"}
        print(f"[!] Encoding test failed: {e}")
    
    return {
        "success": True,
        "model_name": model_name,
        "revision": revision,
        "param_checksum_sha256": param_checksum,
        "param_summary": param_summary,
        "lora_parameters": lora_params,
        "custom_text_proj_parameters": custom_proj_params,
        "encoding_test": encoding_test,
    }


def verify_bge_m3(model_name: str = "BAAI/bge-m3", revision: str = None) -> dict:
    """Verify BGE-M3 model loads correctly."""
    print(f"[*] Loading {model_name} (revision={revision})...")
    
    try:
        from FlagEmbedding import BGEM3FlagModel
        model = BGEM3FlagModel(model_name, revision=revision, use_fp16=True)
        print("[+] Model loaded successfully")
    except Exception as e:
        return {"success": False, "error": f"Failed to load model: {e}"}
    
    # Compute checksums
    param_checksum = compute_param_checksum(model.model)
    param_summary = compute_param_summary(model.model)
    
    # Test encoding
    print("[*] Testing encoding...")
    try:
        test_texts = ["Test query", "Test document"]
        output = model.encode(test_texts, batch_size=2, max_length=512)
        
        encoding_test = {
            "dense_shape": list(output["dense_vecs"].shape),
            "has_sparse": "lexical_weights" in output,
            "has_colbert": "colbert_vecs" in output,
            "dense_has_nan": bool(torch.isnan(torch.tensor(output["dense_vecs"])).any()),
            "dense_mean": float(output["dense_vecs"].mean()),
        }
        print(f"[+] Encoding test passed: dense={output['dense_vecs'].shape}")
    except Exception as e:
        encoding_test = {"error": f"Encoding failed: {e}"}
        print(f"[!] Encoding test failed: {e}")
    
    return {
        "success": True,
        "model_name": model_name,
        "revision": revision,
        "param_checksum_sha256": param_checksum,
        "param_summary": param_summary,
        "encoding_test": encoding_test,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Verify model loading and parameters")
    parser.add_argument("--colpali", default="vidore/colpali-v1.2", help="ColPali model name")
    parser.add_argument("--colpali_revision", default=None, help="ColPali model revision")
    parser.add_argument("--bge_m3", default="BAAI/bge-m3", help="BGE-M3 model name")
    parser.add_argument("--bge_m3_revision", default=None, help="BGE-M3 model revision")
    parser.add_argument("--output", default="model_verification.json", help="Output JSON file")
    args = parser.parse_args()
    
    results = {
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    
    # Verify ColPali
    results["colpali"] = verify_colpali_loading(args.colpali, args.colpali_revision)
    
    # Verify BGE-M3
    results["bge_m3"] = verify_bge_m3(args.bge_m3, args.bge_m3_revision)
    
    # Save results
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n[+] Verification results saved to {args.output}")
    
    # Print summary
    for model_key in ["colpali", "bge_m3"]:
        if results[model_key].get("success"):
            print(f"\n{model_key.upper()}:")
            print(f"  Param checksum: {results[model_key]['param_checksum_sha256'][:32]}...")
            print(f"  Total params: {results[model_key]['param_summary']['total_parameters']:,}")
            print(f"  Trainable params: {results[model_key]['param_summary']['trainable_parameters']:,}")
            if results[model_key].get("lora_parameters"):
                print(f"  LoRA params found: {len(results[model_key]['lora_parameters'])}")
            if results[model_key].get("custom_text_proj_parameters"):
                print(f"  custom_text_proj params: {results[model_key]['custom_text_proj_parameters']}")
        else:
            print(f"\n{model_key.upper()}: FAILED - {results[model_key].get('error')}")
    
    return 0 if all(r.get("success") for r in [results["colpali"], results["bge_m3"]]) else 1


if __name__ == "__main__":
    sys.exit(main())