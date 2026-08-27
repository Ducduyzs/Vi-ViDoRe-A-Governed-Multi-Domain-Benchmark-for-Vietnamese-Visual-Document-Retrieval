import pytest
import torch
import numpy as np

from src.models.maxsim import maxsim_pytorch, maxsim_numpy, rank_documents_maxsim
from src.models.visual_retriever import _concat_padded_embeddings


def test_concat_variable_length_query_batches_preserves_maxsim_scores():
    torch.manual_seed(42)
    first_batch = torch.randn(2, 3, 8)
    second_batch = torch.randn(1, 5, 8)
    documents = torch.randn(4, 6, 8)

    combined = _concat_padded_embeddings([first_batch, second_batch])

    assert combined.shape == (3, 5, 8)
    assert torch.count_nonzero(combined[:2, 3:]) == 0
    assert torch.allclose(
        maxsim_pytorch(combined[:2], documents),
        maxsim_pytorch(first_batch, documents),
    )

def test_maxsim_single_pair():
    # 3 query tokens, 4 doc tokens, dim = 8
    torch.manual_seed(42)
    q = torch.randn(3, 8)
    d = torch.randn(4, 8)

    score_torch = maxsim_pytorch(q, d).item()
    score_np = maxsim_numpy(q.numpy(), d.numpy())

    assert pytest.approx(score_torch, rel=1e-4) == score_np

def test_maxsim_batched():
    torch.manual_seed(42)
    # B_q = 2, N_q = 4, D = 16
    q_batch = torch.randn(2, 4, 16)
    # B_d = 5, N_d = 6, D = 16
    d_batch = torch.randn(5, 6, 16)

    scores = maxsim_pytorch(q_batch, d_batch)
    assert scores.shape == (2, 5)

    # Check pairwise match
    for i in range(2):
        for j in range(5):
            single_score = maxsim_pytorch(q_batch[i], d_batch[j]).item()
            assert pytest.approx(scores[i, j].item(), rel=1e-4) == single_score

def test_maxsim_padding_mask():
    torch.manual_seed(42)
    q = torch.randn(1, 4, 8)
    d = torch.randn(1, 5, 8)

    # Query mask: only first 2 tokens are valid
    q_mask = torch.tensor([[True, True, False, False]])
    score_masked = maxsim_pytorch(q, d, query_mask=q_mask).item()

    # Compare with explicit slice of valid query tokens
    score_explicit = maxsim_pytorch(q[:, :2, :], d).item()
    assert pytest.approx(score_masked, rel=1e-4) == score_explicit

def test_rank_documents_maxsim():
    torch.manual_seed(42)
    q = torch.randn(2, 3, 16)
    corpus = torch.randn(10, 5, 16)

    top_scores, top_indices = rank_documents_maxsim(q, corpus, top_k=3, batch_size=4)
    assert top_scores.shape == (2, 3)
    assert top_indices.shape == (2, 3)
    # Verify scores are sorted descending
    assert (top_scores[:, 0] >= top_scores[:, 1]).all()
    assert (top_scores[:, 1] >= top_scores[:, 2]).all()

