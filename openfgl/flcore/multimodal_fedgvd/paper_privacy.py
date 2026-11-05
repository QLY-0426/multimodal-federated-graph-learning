"""FedGVD+ manuscript mechanisms, NOT a certified full-vector DP release.

Uniform class permutations preserve the multiset of logits. Epsilon controls
the paper's retention probability; it is not a validated privacy budget here.
"""
import math

import torch
import torch.nn.functional as F


def retention_probability(epsilon, classes):
    if not math.isfinite(epsilon) or epsilon < 0 or classes < 1:
        raise ValueError('finite nonnegative mismatch epsilon and positive classes required')
    inverse = math.exp(-epsilon)
    return (1 - inverse) / (1 + (classes - 1) * inverse)


@torch.no_grad()
def randomized_mismatch(logits, epsilon, generator):
    if logits.ndim != 2 or not torch.isfinite(logits).all():
        raise ValueError('finite [nodes, classes] logits required')
    n, classes = logits.shape
    beta = retention_probability(epsilon, classes)
    selected = torch.rand(n, generator=generator) >= beta
    # Fisher-Yates gives a uniform permutation, including the identity.
    permutations = torch.arange(classes).expand(n, -1).clone()
    rows = torch.arange(n)
    for j in range(classes - 1, 0, -1):
        index = torch.randint(j + 1, (n,), generator=generator)
        old = permutations[:, j].clone()
        permutations[:, j] = permutations[rows, index]
        permutations[rows, index] = old
    permuted = logits.gather(1, permutations.to(logits.device))
    result = torch.where(selected.to(logits.device)[:, None], permuted, logits).detach().clone()
    return result, {'rows': n, 'beta': beta, 'selected_rows': int(selected.sum()),
                    'changed_rows': int((result != logits).any(1).sum()),
                    'changed_predictions': int((result.argmax(1) != logits.argmax(1)).sum())}


@torch.no_grad()
def prototype_filter(embeddings, labels, real_train_mask, real_nodes, teacher, quantile=.95):
    """Train-only class prototypes; pooled intra-class distance quantile.

The quantile definition is an explicit implementation choice, not specified
by the manuscript. Zero/missing prototypes and zero query embeddings reject.
"""
    if not math.isfinite(quantile) or not 0 <= quantile <= 1:
        raise ValueError('prototype quantile must be in [0, 1]')
    if len(embeddings) != real_nodes + len(teacher):
        raise ValueError('peer embeddings and teacher logits must be aligned')
    classes = teacher.size(1)
    h = embeddings.detach()
    finite = torch.isfinite(h).all(1) & (h.norm(dim=1) > 1e-12)
    mask = real_train_mask[:real_nodes] & finite[:real_nodes]
    labels_real = labels[:real_nodes][mask]
    if ((labels_real < 0) | (labels_real >= classes)).any():
        raise ValueError('training label outside public class range')
    prototypes = h.new_zeros(classes, h.size(1))
    counts = h.new_zeros(classes)
    prototypes.index_add_(0, labels_real, h[:real_nodes][mask])
    counts.index_add_(0, labels_real, h.new_ones(len(labels_real)))
    prototypes /= counts.clamp_min(1)[:, None]
    present = (counts > 0) & (prototypes.norm(dim=1) > 1e-12)
    usable = present[labels_real]
    intra = (1 - F.cosine_similarity(h[:real_nodes][mask][usable], prototypes[labels_real[usable]], dim=1)).clamp(0, 2)
    threshold = float(torch.quantile(intra.float(), quantile)) if len(intra) else 0.
    predicted = teacher.argmax(1)
    distances = (1 - F.cosine_similarity(h[real_nodes:], prototypes[predicted], dim=1)).clamp(0, 2)
    keep = present[predicted] & finite[real_nodes:] & (distances <= threshold + 1e-6)
    return keep, {'threshold': threshold, 'available_classes': int(present.sum()),
                  'candidate_rows': len(teacher), 'accepted_rows': int(keep.sum())}


def normalized_kl(student, teacher, keep, temperature=1.):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('distillation temperature must be finite and positive')
    if not keep.any():
        return student.sum() * 0
    def normalize(x):
        return (x - x.mean(1, keepdim=True)) / x.std(1, unbiased=False, keepdim=True).clamp_min(1e-6)
    log_student = F.log_softmax(normalize(student[keep]) / temperature, dim=1)
    log_teacher = F.log_softmax(normalize(teacher.detach()[keep]) / temperature, dim=1)
    return F.kl_div(log_student, log_teacher, log_target=True, reduction='batchmean')
