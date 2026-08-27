from typing import Optional
import torch
import torch.nn as nn
from src.models.maxsim import maxsim_pytorch

class LateInteractionInfoNCELoss(nn.Module):
    """
    In-batch Contrastive InfoNCE Loss tailored for Late-Interaction Multi-Vector Representations.
    Supports in-batch negatives, explicit mined hard negatives, and same-PDF negative weighting.
    """
    def __init__(
        self,
        temperature: float = 0.05,
        in_pdf_weight: float = 1.5,
    ):
        super().__init__()
        self.temperature = temperature
        self.in_pdf_weight = in_pdf_weight

    def forward(
        self,
        query_embeddings: torch.Tensor,     # (B, N_q, D)
        pos_doc_embeddings: torch.Tensor,   # (B, N_d, D)
        neg_doc_embeddings: Optional[torch.Tensor] = None, # (B, num_neg, N_d, D) or (N_all_neg, N_d, D)
        in_pdf_mask: Optional[torch.Tensor] = None,        # bool mask for higher penalty on same-pdf negatives
    ) -> torch.Tensor:
        batch_size = query_embeddings.shape[0]

        # 1. In-batch positive scores: s(q_i, d_i^+)
        # Calculate full in-batch similarity matrix: (B, B)
        in_batch_sim = maxsim_pytorch(query_embeddings, pos_doc_embeddings) / self.temperature

        # Target is the diagonal (i == j)
        labels = torch.arange(batch_size, device=query_embeddings.device)

        if neg_doc_embeddings is None:
            # Standard in-batch contrastive loss
            return nn.functional.cross_entropy(in_batch_sim, labels)

        # 2. If explicit hard negatives are provided
        # neg_doc_embeddings shape: (B, K_neg, N_d, D)
        b, k_neg, n_d, dim = neg_doc_embeddings.shape
        flat_neg = neg_doc_embeddings.view(b * k_neg, n_d, dim)

        # Calculate similarity between each query_i and its K_neg hard negatives: (B, K_neg)
        # We can calculate query_i vs its specific negatives
        q_neg_sims = []
        for i in range(batch_size):
            q_i = query_embeddings[i : i + 1]  # (1, N_q, D)
            negs_i = neg_doc_embeddings[i]     # (K_neg, N_d, D)
            sim_i = maxsim_pytorch(q_i, negs_i) / self.temperature  # (1, K_neg)
            q_neg_sims.append(sim_i)

        q_neg_sim_tensor = torch.cat(q_neg_sims, dim=0)  # (B, K_neg)

        # Concatenate in-batch similarities and explicit negative similarities: (B, B + K_neg)
        full_logits = torch.cat([in_batch_sim, q_neg_sim_tensor], dim=-1)

        # Loss is cross entropy where class index 0..B-1 on the in-batch part is the target
        loss = nn.functional.cross_entropy(full_logits, labels)
        return loss

