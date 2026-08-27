from typing import Optional, Tuple, Union
import torch
import numpy as np

def maxsim_pytorch(
    query_embeddings: torch.Tensor,
    doc_embeddings: torch.Tensor,
    query_mask: Optional[torch.Tensor] = None,
    doc_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Computes late interaction MaxSim score using PyTorch.

    Args:
        query_embeddings: Tensor of shape (B_q, N_q, D) or (N_q, D)
        doc_embeddings: Tensor of shape (B_d, N_d, D) or (N_d, D)
        query_mask: Optional bool tensor (B_q, N_q) where True indicates valid token
        doc_mask: Optional bool tensor (B_d, N_d) where True indicates valid token

    Returns:
        scores: Tensor of shape (B_q, B_d) or scalar if single pair.
    """
    is_single_q = query_embeddings.ndim == 2
    is_single_d = doc_embeddings.ndim == 2

    if is_single_q:
        query_embeddings = query_embeddings.unsqueeze(0)
    if is_single_d:
        doc_embeddings = doc_embeddings.unsqueeze(0)

    # Normalize vectors to unit length for cosine similarity
    query_norm = torch.nn.functional.normalize(query_embeddings, p=2, dim=-1)
    doc_norm = torch.nn.functional.normalize(doc_embeddings, p=2, dim=-1)

    # B_q, N_q, D  vs  B_d, N_d, D
    # Pairwise token similarity: (B_q, B_d, N_q, N_d)
    # Using einsum: 'b q d, c k d -> b c q k'
    sim_matrix = torch.einsum("bqd,ckd->bcqk", query_norm, doc_norm)

    if doc_mask is not None:
        # Mask out padding document tokens with a very negative value before max
        if doc_mask.ndim == 1:
            doc_mask = doc_mask.unsqueeze(0)
        # doc_mask: (B_d, N_d) -> expand to (1, B_d, 1, N_d)
        expanded_doc_mask = doc_mask.unsqueeze(0).unsqueeze(2)
        sim_matrix = sim_matrix.masked_fill(~expanded_doc_mask, -1e4)

    # Max over document tokens: (B_q, B_d, N_q)
    max_sim, _ = torch.max(sim_matrix, dim=-1)

    if query_mask is not None:
        if query_mask.ndim == 1:
            query_mask = query_mask.unsqueeze(0)
        # query_mask: (B_q, N_q) -> expand to (B_q, 1, N_q)
        expanded_q_mask = query_mask.unsqueeze(1)
        max_sim = max_sim.masked_fill(~expanded_q_mask, 0.0)
        scores = torch.sum(max_sim, dim=-1)
    else:
        scores = torch.sum(max_sim, dim=-1)

    if is_single_q and is_single_d:
        return scores.squeeze(0).squeeze(0)
    return scores

def maxsim_numpy(
    query_embeddings: np.ndarray,
    doc_embeddings: np.ndarray,
) -> float:
    """
    Numpy implementation for a single query (N_q, D) and single doc (N_d, D).
    """
    # Normalize
    q_norm = query_embeddings / (np.linalg.norm(query_embeddings, axis=-1, keepdims=True) + 1e-9)
    d_norm = doc_embeddings / (np.linalg.norm(doc_embeddings, axis=-1, keepdims=True) + 1e-9)

    # Cosine similarity matrix (N_q, N_d)
    sim = np.matmul(q_norm, d_norm.T)
    # Max along doc tokens, then sum over query tokens
    return float(np.sum(np.max(sim, axis=-1)))

def rank_documents_maxsim(
    query_embeddings: torch.Tensor,
    corpus_embeddings: torch.Tensor,
    top_k: int = 10,
    batch_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Ranks an entire corpus of multi-vector document representations for a batch of queries.

    Args:
        query_embeddings: Tensor of shape (B_q, N_q, D)
        corpus_embeddings: Tensor of shape (N_corpus, N_d, D)
        top_k: Number of top documents to retrieve

    Returns:
        top_scores: (B_q, top_k)
        top_indices: (B_q, top_k)
    """
    device = query_embeddings.device
    num_queries = query_embeddings.shape[0]
    num_docs = corpus_embeddings.shape[0]

    all_scores = []
    for i in range(0, num_docs, batch_size):
        batch_docs = corpus_embeddings[i : i + batch_size].to(device)
        batch_scores = maxsim_pytorch(query_embeddings, batch_docs)  # (B_q, batch_size)
        all_scores.append(batch_scores.cpu())

    full_scores = torch.cat(all_scores, dim=1)  # (B_q, N_corpus)
    k = min(top_k, num_docs)
    top_scores, top_indices = torch.topk(full_scores, k=k, dim=-1)
    return top_scores, top_indices

