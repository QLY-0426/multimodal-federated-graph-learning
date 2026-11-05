config = {
    "lambda": 0.0001,
    "gama": 1.0,
    "branch_weight": 0.5,
    "lr_feat": 0.005,
    "lr_adj": 0.01,
    "feat_alpha": 10.0,
    "smoothness_alpha": 0.1,
    "adj_threshold": 0.5,
    # real_induce: reuse the real subgraph's induced edges among the selected
    # nodes and skip the PGE, so only the paired features are optimized.
    # consistent_low: retain PGE weights >= 0.01 in training and validation;
    # soft: retain all weights in both; legacy: prune only at validation.
    "adj_mode": "real_induce",
    "alternating_optim": False,
}
