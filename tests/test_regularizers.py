import torch

from fermigates.losses import (
    group_sparsity_l21_loss,
    hoyer_sparsity_loss,
    hoyer_sparsity_score,
    kl_to_bernoulli_prior_loss,
)


def test_kl_to_bernoulli_prior_zero_when_matching_prior():
    probs = torch.full((8,), 0.3)
    loss = kl_to_bernoulli_prior_loss(probs, prior_prob=0.3, reduction="sum")
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


def test_group_sparsity_l21_matches_manual():
    values = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
    # sqrt(3^2 + 4^2) + sqrt(0) = 5
    loss = group_sparsity_l21_loss(values, group_dim=0, reduction="sum")
    assert torch.allclose(loss, torch.tensor(5.0), atol=1e-6)


def test_hoyer_score_prefers_sparse_vectors():
    dense = torch.ones(10)
    sparse = torch.tensor([1.0] + [0.0] * 9)

    dense_score = hoyer_sparsity_score(dense, reduction="mean")
    sparse_score = hoyer_sparsity_score(sparse, reduction="mean")
    assert sparse_score > dense_score

    dense_loss = hoyer_sparsity_loss(dense, normalized=False, reduction="mean")
    sparse_loss = hoyer_sparsity_loss(sparse, normalized=False, reduction="mean")
    assert sparse_loss < dense_loss
