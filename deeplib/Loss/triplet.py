# Borrowed from this blog post: https://omoindrot.github.io/triplet-loss

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import deeplib.util as util


def batch_hard_triplet_loss(embeddings, labels, margin):
    """Build the triplet loss over a batch of embeddings.
    For each anchor, we get the hardest positive and hardest negative to form a triplet.
    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                If false, output is the pairwise euclidean distance matrix.
    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = pairwise_distances(embeddings)

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels).float()

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = mask_anchor_positive * pairwise_dist

    # shape (batch_size, 1)
    hardest_positive_dist, _ = anchor_positive_dist.max(1, keepdim=True)

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels).float()

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist, _ = pairwise_dist.max(1, keepdim=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # shape (batch_size,)
    hardest_negative_dist, _ = anchor_negative_dist.min(1, keepdim=True)

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    tl = hardest_positive_dist - hardest_negative_dist + margin
    tl[tl < 0] = 0
    triplet_loss = tl.mean()

    return triplet_loss

class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        return batch_hard_triplet_loss(embeddings, labels, self.margin)

def batch_all_triplet_loss(embeddings, labels, margin): 
    """Build the triplet loss over a batch of embeddings.
    We generate all the valid triplets and average the loss over the positive ones.
    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                If false, output is the pairwise euclidean distance matrix.
    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = pairwise_distances(embeddings)

    anchor_positive_dist = pairwise_dist.unsqueeze(2)
    anchor_negative_dist = pairwise_dist.unsqueeze(1)

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = _get_triplet_mask(labels)
    triplet_loss = mask.float() * triplet_loss

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss[triplet_loss < 0] = 0

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = triplet_loss[triplet_loss > 1e-16]
    num_positive_triplets = valid_triplets.size(0)
    num_valid_triplets = mask.sum()

    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets.float() + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)

    return triplet_loss

class BatchAllTripletLoss(nn.Module):
    def __init__(self, margin):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        return batch_all_triplet_loss(embeddings, labels, self.margin)


def select_pos_neg_dists(embeddings, labels, cutoff, nonzero_loss_cutoff):
    n, d = embeddings.shape

    distance = pairwise_distances(embeddings)
    distance = distance.clamp(min=cutoff)

    mask_anc hor_positive = _get_anchor_positive_triplet_mask(labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels)

    log_weights = ((2.0 - float(d)) * distance.log() - (float(d-3)/2)*torch.log(torch.clamp(1.0 - 0.25*(distance*distance), min=1e-8)))
    log_weights = (log_weights - log_weights.min()) / (log_weights.max() - log_weights.min() + 1e-8)

    weights = torch.exp(log_weights - torch.max(log_weights))

    weights = weights * mask_anchor_negative.float() * (distance < nonzero_loss_cutoff).float()

    weights_sum = torch.sum(weights, dim=1, keepdim=True)
    weights = weights / weights_sum

    weights = util.to_cpu(weights).detach().numpy()
 
    pos_dist = []
    neg_dist = []
    anc_labels = []

    for i, (y, distances, pos_mask, neg_mask) in enumerate(zip(labels, distance, mask_anchor_positive, mask_anchor_negative.cpu())):
      num_triplets = min(pos_mask.sum().int().item(), neg_mask.sum().int().item())
      pos_dists = distances[pos_mask][:num_triplets]

      if weights_sum[i] != 0:
        neg_dist_indicies = np.random.choice(n, num_triplets, p=weights[i])
      else:
        neg_dist_indicies = np.random.choice(n, num_triplets, p=neg_mask.double()/neg_mask.sum())

      neg_dists = distances[neg_dist_indicies]

      pos_dist.append(pos_dists)
      neg_dist.append(neg_dists)
      anc_labels.append(y.expand_as(pos_dists))

    pos_dist = torch.cat(pos_dist)
    neg_dist = torch.cat(neg_dist)
    anc_labels = torch.cat(anc_labels)

    return pos_dist, neg_dist, anc_labels
 
def distance_weighted_triplet_loss(embeddings, labels, margin, nonzero_loss_cutoff=1.4, cutoff=0.5):
    pos_dist, neg_dist, _ = select_pos_neg_dists(embeddings, labels, cutoff, nonzero_loss_cutoff)

    loss = pos_dist - neg_dist + margin
    loss[loss < 0] = 0

    return loss.mean()

class DistanceWeightedTripletLoss(nn.Module):
  def __init__(self, margin, nonzero_loss_cutoff=1.4, cutoff=0.5):
    super(DistanceWeightedTripletLoss, self).__init__()
    self.margin = margin
    self.nonzero_loss_cutoff = nonzero_loss_cutoff
    self.cutoff = cutoff

  def forward(self, x, y):
    return distance_weighted_triplet_loss(x, y, self.margin, self.nonzero_loss_cutoff, self.cutoff) 

def distance_weighted_margin_loss(embeddings, labels, margin, beta, nonzero_loss_cutoff=1.4, cutoff=0.5):
    pos_dist, neg_dist, anc_labels = select_pos_neg_dists(embeddings, labels, cutoff, nonzero_loss_cutoff)
    betas = beta[anc_labels]

    pos_loss = torch.clamp(pos_dist - betas + margin, min=0.0)
    neg_loss = torch.clamp(betas - neg_dist + margin, min=0.0)

    pair_cnt = int(torch.sum((pos_loss > 0.0) + (neg_loss > 0.0)))

    loss = (torch.sum(pos_loss + neg_loss)) / pair_cnt
    return loss

class DistanceWeightedMarginLoss(nn.Module):
  def __init__(self, num_classes, margin=.2, beta=1.2, nonzero_loss_cutoff=1.4, cutoff=0.5):
    super(DistanceWeightedMarginLoss, self).__init__()
    self.nonzero_loss_cutoff = nonzero_loss_cutoff
    self.margin = margin
    self.cutoff = cutoff
    self.beta = nn.Parameter(torch.ones((num_classes,), dtype=torch.float32, device=torch.device('cuda'))*beta)
    self.optimizer_beta = torch.optim.SGD([self.beta], .1, momentum=.9, weight_decay=0.0001)

  def forward(self, x, y):
    margin_loss = distance_weighted_margin_loss(x, y, self.margin, self.beta, self.nonzero_loss_cutoff, self.cutoff) 
    return margin_loss

def pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.
    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    dot_product = torch.matmul(embeddings, embeddings.t())

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = torch.diag(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances[distances < 0] = 0

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = distances.eq(0).float()
        distances = distances + mask * 1e-16

        distances = (1.0 -mask) * torch.sqrt(distances)

    return distances

def _get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = util.to_gpu(torch.eye(labels.size(0)))
    indices_not_equal = indices_equal != 1
    i_not_equal_j = indices_not_equal.unsqueeze(2)
    i_not_equal_k = indices_not_equal.unsqueeze(1)
    j_not_equal_k = indices_not_equal.unsqueeze(0)

    distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k


    label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    i_equal_j = label_equal.unsqueeze(2)
    i_equal_k = label_equal.unsqueeze(1)

    valid_labels = ~i_equal_k & i_equal_j

    return valid_labels & distinct_indices


def _get_anchor_positive_triplet_mask(labels):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check that i and j are distinct
    indices_equal = util.to_gpu(torch.eye(labels.size(0)))
    indices_not_equal = indices_equal != 1

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)

    return labels_equal & indices_not_equal


def _get_anchor_negative_triplet_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)

    return ~(labels.unsqueeze(0) == labels.unsqueeze(1))
