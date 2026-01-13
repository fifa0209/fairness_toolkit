# New file: measurement_module/src/ranking_metrics.py

import numpy as np
from typing import Dict, Tuple, Optional

def exposure_parity_difference(
    rankings: np.ndarray,
    sensitive_features: np.ndarray,
    top_k: Optional[int] = None,
    position_weights: Optional[np.ndarray] = None
) -> Tuple[float, Dict[str, float], Dict[str, int]]:
    """
    Compute exposure parity difference for ranked results.
    
    Exposure measures how visible items from different groups are
    in the ranking, accounting for position bias.
    
    Args:
        rankings: Array of item IDs in ranked order (higher = better)
        sensitive_features: Protected attribute for each item
        top_k: Only consider top-k positions (None = all)
        position_weights: Custom weights per position (None = 1/log2(pos+1))
    
    Returns:
        Tuple of (difference, group_exposures, group_sizes)
    
    Example:
        >>> rankings = np.array([0, 1, 2, 3, 4])  # Item IDs
        >>> groups = np.array([0, 1, 0, 1, 0])     # Group membership
        >>> diff, exposures, sizes = exposure_parity_difference(rankings, groups)
    """
    # Validate inputs
    if len(rankings) == 0:
        raise ValueError("Rankings array cannot be empty")
    
    # Check that all ranking IDs are valid indices for sensitive_features
    max_id = np.max(rankings) if len(rankings) > 0 else -1
    if max_id >= len(sensitive_features):
        raise IndexError(
            f"Ranking contains item ID {max_id} but sensitive_features only has {len(sensitive_features)} items"
        )
    
    if top_k is None:
        top_k = len(rankings)
    else:
        top_k = min(top_k, len(rankings))
    
    # Default position weights: 1/log2(position+1) (DCG-style)
    if position_weights is None:
        positions = np.arange(1, top_k + 1)
        position_weights = 1.0 / np.log2(positions + 1)
    
    # Get groups
    groups = np.unique(sensitive_features)
    if len(groups) != 2:
        raise ValueError(f"Expected 2 groups, got {len(groups)}")
    
    # Compute exposure for each group
    group_exposures = {}
    group_sizes = {}
    
    for group in groups:
        # Items belonging to this group
        group_mask = sensitive_features == group
        group_size = np.sum(group_mask)
        group_sizes[f"Group_{group}"] = int(group_size)
        
        if group_size == 0:
            group_exposures[f"Group_{group}"] = 0.0
            continue
        
        # Compute exposure: sum of position weights for this group's items
        exposure = 0.0
        for pos_idx in range(top_k):
            item_id = rankings[pos_idx]
            if item_id < len(sensitive_features) and sensitive_features[item_id] == group:
                exposure += position_weights[pos_idx]
        
        # Normalize by group size
        normalized_exposure = exposure / group_size
        group_exposures[f"Group_{group}"] = normalized_exposure
    
    # Compute difference
    exposures = list(group_exposures.values())
    difference = abs(exposures[0] - exposures[1])
    
    return difference, group_exposures, group_sizes


def normalized_discounted_cumulative_fairness(
    rankings: np.ndarray,
    sensitive_features: np.ndarray,
    relevance_scores: Optional[np.ndarray] = None,
    top_k: Optional[int] = None
) -> Tuple[float, Dict[str, float]]:
    """
    Compute normalized DCG-based fairness metric.
    
    Measures whether high-relevance items from different groups
    receive proportional visibility in top positions.
    
    Args:
        rankings: Ranked item IDs
        sensitive_features: Group membership
        relevance_scores: True relevance (None = assume all equal)
        top_k: Consider top-k positions
    
    Returns:
        Tuple of (fairness_score, group_metrics)
    """
    if top_k is None:
        top_k = len(rankings)
    
    if relevance_scores is None:
        relevance_scores = np.ones(len(rankings))
    
    groups = np.unique(sensitive_features)
    
    # Compute DCG per group
    group_dcg = {}
    group_ideal_dcg = {}
    
    for group in groups:
        group_mask = sensitive_features == group
        
        # Actual DCG
        dcg = 0.0
        for pos in range(top_k):
            item_id = rankings[pos]
            if item_id < len(sensitive_features) and sensitive_features[item_id] == group:
                rel = relevance_scores[item_id]
                dcg += rel / np.log2(pos + 2)
        
        # Ideal DCG (if all group items were perfectly ranked)
        group_relevances = relevance_scores[group_mask]
        sorted_rel = np.sort(group_relevances)[::-1]
        
        idcg = 0.0
        for pos, rel in enumerate(sorted_rel[:top_k]):
            idcg += rel / np.log2(pos + 2)
        
        group_dcg[f"Group_{group}"] = dcg
        group_ideal_dcg[f"Group_{group}"] = idcg if idcg > 0 else 1.0
    
    # Compute nDCG per group
    group_ndcg = {}
    for group in groups:
        group_name = f"Group_{group}"
        ndcg = group_dcg[group_name] / group_ideal_dcg[group_name]
        group_ndcg[group_name] = ndcg
    
    # Fairness = max difference in nDCG
    ndcgs = list(group_ndcg.values())
    fairness_score = abs(ndcgs[0] - ndcgs[1])
    
    return fairness_score, group_ndcg


def attention_weighted_fairness(
    rankings: np.ndarray,
    sensitive_features: np.ndarray,
    attention_model: str = 'exponential',
    top_k: Optional[int] = None
) -> Tuple[float, Dict[str, float]]:
    """
    Compute fairness using attention-weighted visibility.
    
    Models user attention decay as they scan down a ranked list.
    
    Args:
        rankings: Ranked item IDs
        sensitive_features: Group membership
        attention_model: 'exponential', 'linear', or 'log'
        top_k: Consider top-k positions
    
    Returns:
        Tuple of (difference, group_attention)
    """
    if top_k is None:
        top_k = len(rankings)
    
    # Attention weights based on model
    positions = np.arange(1, top_k + 1)
    
    if attention_model == 'exponential':
        # Exponential decay: attention_weight = exp(-decay_rate * position)
        decay_rate = 0.1
        weights = np.exp(-decay_rate * positions)
    elif attention_model == 'linear':
        # Linear decay
        weights = 1.0 - (positions - 1) / top_k
    elif attention_model == 'log':
        # Logarithmic (DCG-style)
        weights = 1.0 / np.log2(positions + 1)
    else:
        raise ValueError(f"Unknown attention model: {attention_model}")
    
    # Normalize weights
    weights = weights / np.sum(weights)
    
    # Compute attention per group
    groups = np.unique(sensitive_features)
    group_attention = {}
    
    for group in groups:
        attention = 0.0
        group_count = 0
        
        for pos in range(top_k):
            item_id = rankings[pos]
            if item_id < len(sensitive_features) and sensitive_features[item_id] == group:
                attention += weights[pos]
                group_count += 1
        
        # Normalize by group representation in top-k
        if group_count > 0:
            attention = attention / group_count
        
        group_attention[f"Group_{group}"] = attention
    
    # Difference in attention
    attentions = list(group_attention.values())
    difference = abs(attentions[0] - attentions[1])
    
    return difference, group_attention