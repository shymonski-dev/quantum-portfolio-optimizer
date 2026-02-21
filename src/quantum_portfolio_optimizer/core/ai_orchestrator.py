"""AI Orchestrator for portfolio partitioning and hardware-aware mapping.

This module implements 'Min-Cut' clustering to automatically group assets into
sectors that minimize quantum information loss when split across modular hardware.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering

logger = logging.getLogger(__name__)


class PortfolioClusturer:
    """AI-driven asset grouping using graph partitioning (Normalized Min-Cut)."""

    def __init__(self, n_clusters: int = 2):
        self.n_clusters = n_clusters
        self.labels_: Optional[np.ndarray] = None

    def cluster_assets(self, returns: pd.DataFrame) -> Dict[str, List[int]]:
        """Group assets into sectors based on correlation min-cut.

        Args:
            returns: DataFrame of historical asset returns.

        Returns:
            Dictionary mapping cluster names to lists of asset indices.
        """
        # 1. Calculate Correlation Matrix
        corr_matrix = returns.corr().values
        
        # 2. Transform into Adjacency Matrix (Similarity Graph)
        # We use absolute correlation as edge weight. 
        # Assets with higher correlation have 'stronger' grips.
        adjacency = np.abs(corr_matrix)
        np.fill_diagonal(adjacency, 0) # No self-loops

        # 3. Apply Spectral Clustering (Normalized Min-Cut)
        # Spectral clustering find the partition that minimizes the cut edges
        # while keeping the cluster sizes balanced.
        model = SpectralClustering(
            n_clusters=self.n_clusters,
            affinity='precomputed',
            assign_labels='kmeans',
            random_state=42
        )
        
        self.labels_ = model.fit_predict(adjacency)
        
        # 4. Format as Sectors
        sectors = {}
        for cluster_id in range(self.n_clusters):
            indices = np.where(self.labels_ == cluster_id)[0].tolist()
            sectors[f"AI_Sector_{cluster_id + 1}"] = indices
            
        logger.info(f"AI Clustering complete: {len(sectors)} sectors identified.")
        return sectors


def get_optimal_cut_points(
    returns: pd.DataFrame, 
    max_cluster_size: int = 25
) -> Dict[str, List[int]]:
    """Determine optimal sectors based on hardware constraints.

    Args:
        returns: DataFrame of returns.
        max_cluster_size: Maximum qubits (assets) per modular chip.

    Returns:
        Sector map for partitioning.
    """
    num_assets = returns.shape[1]
    n_clusters = int(np.ceil(num_assets / max_cluster_size))
    
    if n_clusters <= 1:
        return {"Global": list(range(num_assets))}
        
    clusturer = PortfolioClusturer(n_clusters=n_clusters)
    return clusturer.cluster_assets(returns)
