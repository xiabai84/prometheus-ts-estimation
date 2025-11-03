import pandas as pd
import numpy as np
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMinMax, TimeSeriesScalerMeanVariance
from tslearn.metrics import cdist_dtw, cdist_soft_dtw, dtw
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import warnings
import json
import os
from typing import List, Dict, Any
warnings.filterwarnings('ignore')


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy data types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

class DTWClustering:
    def __init__(self, n_clusters=3, metric="dtw", max_iter=50, 
                 min_non_zero_ratio=0.1, min_variance=0.001,
                 scaler_type="minmax", dtw_params=None, 
                 n_init=3, random_state=42, verbose=False,
                 window_constraint=None, warp_penalty=0.0):
        """
        Robust Time Series Clustering using tslearn library
        
        Parameters:
        - n_clusters: Number of clusters (TimeSeriesKMeans)
        - metric: Distance metric ('dtw', 'softdtw', 'euclidean')
        - max_iter: Maximum number of iterations
        - min_non_zero_ratio: Minimum ratio of non-zero/non-null values
        - min_variance: Minimum variance threshold
        - scaler_type: Normalization type ('minmax', 'meanvar')
        - dtw_params: DTW parameters dictionary
        - n_init: Number of K-Means initializations (TimeSeriesKMeans)
        - random_state: Random seed for reproducibility
        - verbose: Whether to show detailed output
        - window_constraint: Window size for DTW constraint (None, 'sakoe_chiba', 'itakura')
        - warp_penalty: Penalty for warping in soft-DTW (higher = less warping allowed)
        """
        self.n_clusters = n_clusters
        self.metric = metric
        self.max_iter = max_iter
        self.min_non_zero_ratio = min_non_zero_ratio
        self.min_variance = min_variance
        self.scaler_type = scaler_type
        self.dtw_params = dtw_params or {}
        self.n_init = n_init
        self.random_state = random_state
        self.verbose = verbose
        self.window_constraint = window_constraint
        self.warp_penalty = warp_penalty
        
        self.model = None
        self.labels_ = None
        self.distance_matrix_ = None
        self.valid_columns_ = None
        self.removed_columns_ = None
        self.final_n_clusters_ = None
        self.normalized_data_ = None
        self.original_data_ = None
        self.similar_metrics_dict_ = None
        self.cluster_centers_ = None
        self.clustering_scores_ = None
        self.column_to_index_ = None
        self.pairwise_distances_ = None
        self.window_size_ = None
        
    def _clean_data(self, data):
        """
        Clean input data by removing metrics with insufficient data
        """
        if self.verbose:
            print("Cleaning data...")
        
        original_columns = data.columns.tolist()
        
        # Convert to numeric, coercing errors to NaN
        data_clean = data.apply(pd.to_numeric, errors='coerce')
        
        # Remove columns with too many null/infinite values
        null_ratio = data_clean.isnull().mean()
        infinite_mask = data_clean.apply(lambda x: np.isinf(x).any())
        
        # Calculate non-zero ratio (excluding NaN and zeros)
        non_zero_ratio = (data_clean.notnull() & (data_clean != 0)).mean()
        
        # Calculate variance (excluding NaN)
        variances = data_clean.var()
        
        # Filter conditions
        valid_mask = (
            (null_ratio < 0.8) &  # Less than 80% null values
            (~infinite_mask) &     # No infinite values
            (non_zero_ratio >= self.min_non_zero_ratio) &  # Enough non-zero values
            (variances >= self.min_variance)  # Enough variance
        )
        
        self.valid_columns_ = data_clean.columns[valid_mask].tolist()
        self.removed_columns_ = data_clean.columns[~valid_mask].tolist()
        
        if len(self.valid_columns_) == 0:
            # If no columns pass initial filter, apply more lenient filtering
            if self.verbose:
                print("No columns passed initial filter. Applying lenient filtering...")
            valid_mask = (
                (null_ratio < 0.9) &  # Less than 90% null values
                (~infinite_mask) &     # No infinite values
                (non_zero_ratio >= 0.05)  # At least 5% non-zero values
            )
            self.valid_columns_ = data_clean.columns[valid_mask].tolist()
            self.removed_columns_ = data_clean.columns[~valid_mask].tolist()
        
        # Apply filtering
        cleaned_data = data_clean[self.valid_columns_].copy()
        
        if len(cleaned_data.columns) == 0:
            raise ValueError("No valid metrics remaining after data cleaning!")
        
        # Fill remaining null values
        cleaned_data = cleaned_data.ffill().bfill().fillna(0)
        
        # Remove any remaining rows with all null values
        cleaned_data = cleaned_data.dropna(how='all')
        
        if self.verbose:
            print(f"Original metrics: {len(original_columns)}")
            print(f"Valid metrics after cleaning: {len(self.valid_columns_)}")
            print(f"Removed metrics: {len(self.removed_columns_)}")
        
        if self.verbose and len(self.removed_columns_) > 0:
            print("Removed metrics due to insufficient data:")
            for col in self.removed_columns_[:10]:
                col_null_ratio = float(null_ratio[col])
                col_non_zero_ratio = float(non_zero_ratio[col])
                col_variance = float(variances[col]) if not np.isnan(variances[col]) else 0.0
                print(f"  - {col}: null_ratio={col_null_ratio:.3f}, "
                      f"non_zero_ratio={col_non_zero_ratio:.3f}, "
                      f"variance={col_variance:.3f}")
            if len(self.removed_columns_) > 10:
                print(f"  ... and {len(self.removed_columns_) - 10} more")
        
        # Adjust number of clusters if necessary
        self.final_n_clusters_ = min(self.n_clusters, len(cleaned_data.columns))
        if self.final_n_clusters_ < self.n_clusters:
            print(f"Warning: Reduced number of clusters from {self.n_clusters} to {self.final_n_clusters_} "
                  f"due to insufficient valid metrics.")
        
        if self.final_n_clusters_ < 2:
            raise ValueError(f"Only {len(cleaned_data.columns)} valid metrics remaining. Need at least 2 for clustering.")
        
        return cleaned_data
    
    def _prepare_tslearn_data(self, data):
        """
        Prepare data format for tslearn
        tslearn expects shape: (n_ts, sz, d) where d is dimension
        """
        # Transpose to (n_metrics, n_timesteps)
        ts_data = data.values.T
        
        # Reshape to tslearn format (n_ts, sz, d)
        if len(ts_data.shape) == 2:
            ts_data = ts_data.reshape(ts_data.shape[0], ts_data.shape[1], 1)
        
        if self.verbose:
            print(f"Data shape for tslearn: {ts_data.shape}")
        return ts_data
    
    def _compute_window_size(self, series_length):
        """
        Compute appropriate window size based on constraint type and series length
        """
        if self.window_constraint is None:
            return None
        
        if isinstance(self.window_constraint, (int, float)):
            # Direct window size specification
            if 0 < self.window_constraint <= 1:
                # Percentage of series length
                window_size = int(self.window_constraint * series_length)
            else:
                # Absolute window size
                window_size = int(self.window_constraint)
            
            # Ensure window size is reasonable
            window_size = max(1, min(window_size, series_length - 1))
            return window_size
        
        elif self.window_constraint == 'sakoe_chiba':
            # Sakoe-Chiba band: typically 10-20% of series length
            return max(1, int(0.1 * series_length))
        
        elif self.window_constraint == 'itakura':
            # Itakura parallelogram: more restrictive
            return max(1, int(0.05 * series_length))
        
        else:
            raise ValueError(f"Unknown window constraint: {self.window_constraint}")
    
    def _compute_distance_matrix(self, ts_data):
        """
        Compute distance matrix using tslearn with window constraints
        """
        if self.verbose:
            print(f"Computing {self.metric} distance matrix...")
        
        series_length = ts_data.shape[1]
        self.window_size_ = self._compute_window_size(series_length)
        
        if self.verbose and self.window_size_:
            print(f"Using window constraint: {self.window_size_} ({(self.window_size_/series_length)*100:.1f}% of series length)")
        
        if self.metric == 'dtw':
            try:
                # Build DTW parameters with window constraints
                valid_dtw_params = {}
                
                # Add window constraints if specified
                if self.window_size_:
                    valid_dtw_params['global_constraint'] = 'sakoe_chiba'
                    valid_dtw_params['sakoe_chiba_radius'] = self.window_size_
                
                # Add any user-specified DTW parameters
                if 'global_constraint' in self.dtw_params:
                    valid_dtw_params['global_constraint'] = self.dtw_params['global_constraint']
                if 'sakoe_chiba_radius' in self.dtw_params:
                    valid_dtw_params['sakoe_chiba_radius'] = self.dtw_params['sakoe_chiba_radius']
                if 'itakura_max_slope' in self.dtw_params:
                    valid_dtw_params['itakura_max_slope'] = self.dtw_params['itakura_max_slope']
                
                self.distance_matrix_ = cdist_dtw(ts_data, **valid_dtw_params)
                if self.verbose:
                    print("DTW distance matrix computed successfully")
            except Exception as e:
                print(f"DTW computation failed: {e}")
                raise
                
        elif self.metric == 'softdtw':
            try:
                # For soft-DTW, use gamma as warp penalty
                gamma = self.dtw_params.get('gamma', max(0.1, self.warp_penalty))
                self.distance_matrix_ = cdist_soft_dtw(ts_data, gamma=gamma)
                if self.verbose:
                    print(f"Soft-DTW distance matrix computed successfully (gamma={gamma})")
            except Exception as e:
                print(f"Soft-DTW computation failed: {e}")
                raise
                
        else:
            # For Euclidean distance, tslearn handles internally
            self.distance_matrix_ = None
            
        return self.distance_matrix_
    
    def _create_tslearn_model(self):
        """
        Create tslearn model with proper parameter handling
        """
        model_params = {
            'n_clusters': self.final_n_clusters_,
            'metric': self.metric,
            'max_iter': self.max_iter,
            'n_init': self.n_init,
            'random_state': self.random_state,
            'verbose': self.verbose
        }
        
        # Add metric-specific parameters
        if self.metric == 'dtw':
            model_params['metric_params'] = {}
            
            # Add window constraints if specified
            if self.window_size_:
                model_params['metric_params']['global_constraint'] = 'sakoe_chiba'
                model_params['metric_params']['sakoe_chiba_radius'] = self.window_size_
            
            # Add user-specified DTW parameters
            valid_dtw_params = {}
            if 'global_constraint' in self.dtw_params:
                valid_dtw_params['global_constraint'] = self.dtw_params['global_constraint']
            if 'sakoe_chiba_radius' in self.dtw_params:
                valid_dtw_params['sakoe_chiba_radius'] = self.dtw_params['sakoe_chiba_radius']
            if 'itakura_max_slope' in self.dtw_params:
                valid_dtw_params['itakura_max_slope'] = self.dtw_params['itakura_max_slope']
            
            if valid_dtw_params:
                model_params['metric_params'].update(valid_dtw_params)
                
        elif self.metric == 'softdtw':
            model_params['metric_params'] = {}
            
            # Use warp penalty for soft-DTW
            gamma = self.dtw_params.get('gamma', max(0.1, self.warp_penalty))
            model_params['metric_params']['gamma'] = gamma
        
        return TimeSeriesKMeans(**model_params)
    
    def fit(self, data):
        """
        Fit clustering model using tslearn
        """
        # Clean and preprocess data
        cleaned_data = self._clean_data(data)
        self.original_data_ = cleaned_data
        
        if self.verbose:
            print(f"Data shape after cleaning: {cleaned_data.shape}")
            print(f"Final number of clusters: {self.final_n_clusters_}")
        
        # Prepare tslearn data format
        ts_data = self._prepare_tslearn_data(cleaned_data)
        
        # Data normalization
        if self.scaler_type == "minmax":
            scaler = TimeSeriesScalerMinMax()
        else:  # "meanvar"
            scaler = TimeSeriesScalerMeanVariance()
            
        ts_data_scaled = scaler.fit_transform(ts_data)
        self.normalized_data_ = ts_data_scaled
        
        # Create column name to index mapping
        self.column_to_index_ = {col: i for i, col in enumerate(self.valid_columns_)}
        
        # Compute distance matrix (if needed)
        if self.metric in ['dtw', 'softdtw']:
            self._compute_distance_matrix(ts_data_scaled)
            self.pairwise_distances_ = self.distance_matrix_
        else:
            # For Euclidean distance, compute pairwise distances
            self._compute_euclidean_pairwise_distances(ts_data_scaled)
        
        # Create and train tslearn model
        if self.verbose:
            print(f"Training TimeSeriesKMeans with {self.metric} metric...")
        
        self.model = self._create_tslearn_model()
        self.labels_ = self.model.fit_predict(ts_data_scaled)
        self.cluster_centers_ = self.model.cluster_centers_
        
        # Compute clustering quality scores
        self._compute_clustering_scores(ts_data_scaled)
        
        if self.verbose:
            print(f"Clustering completed: {self.final_n_clusters_} clusters formed")
        
        # Create similar metrics dictionary
        self._create_similar_metrics_dict()
        
        return self
    
    def _compute_euclidean_pairwise_distances(self, ts_data):
        """Compute pairwise distance matrix for Euclidean metric"""
        if self.verbose:
            print("Computing Euclidean pairwise distances...")
        
        n_series = ts_data.shape[0]
        self.pairwise_distances_ = np.zeros((n_series, n_series))
        
        for i in range(n_series):
            for j in range(i + 1, n_series):
                dist = np.sqrt(np.sum((ts_data[i] - ts_data[j]) ** 2))
                self.pairwise_distances_[i, j] = dist
                self.pairwise_distances_[j, i] = dist
        
        if self.verbose:
            print("Euclidean pairwise distances computed successfully")
    
    def _compute_clustering_scores(self, ts_data):
        """
        Compute clustering quality scores
        """
        try:
            # Flatten time series data for score computation
            if len(ts_data.shape) == 3:
                flattened_data = ts_data.reshape(ts_data.shape[0], -1)
            else:
                flattened_data = ts_data
                
            # Use PCA for dimensionality reduction to improve stability
            n_components = min(10, flattened_data.shape[0], flattened_data.shape[1])
            if n_components > 1:
                pca = PCA(n_components=n_components)
                reduced_data = pca.fit_transform(flattened_data)
                
                silhouette_avg = silhouette_score(reduced_data, self.labels_)
                
                self.clustering_scores_ = {
                    'silhouette_score': float(silhouette_avg),
                    'inertia': float(self.model.inertia_) if hasattr(self.model, 'inertia_') else None,
                    'n_iter': int(self.model.n_iter_) if hasattr(self.model, 'n_iter_') else None,
                    'window_size': self.window_size_
                }
                if self.verbose:
                    print(f"Clustering scores - Silhouette: {silhouette_avg:.3f}")
            else:
                self.clustering_scores_ = {
                    'inertia': float(self.model.inertia_) if hasattr(self.model, 'inertia_') else None,
                    'n_iter': int(self.model.n_iter_) if hasattr(self.model, 'n_iter_') else None,
                    'window_size': self.window_size_
                }
                
        except Exception as e:
            if self.verbose:
                print(f"Could not compute clustering scores: {e}")
            self.clustering_scores_ = {'window_size': self.window_size_}
    
    def _create_similar_metrics_dict(self):
        """
        Create dictionary where key is metric name and value is list of similar metrics
        """
        if self.labels_ is None or self.valid_columns_ is None:
            raise ValueError("Model must be fitted first")
            
        # Group metrics by cluster
        cluster_groups = {}
        for i, (metric, cluster_id) in enumerate(zip(self.valid_columns_, self.labels_)):
            if cluster_id not in cluster_groups:
                cluster_groups[cluster_id] = []
            cluster_groups[cluster_id].append(metric)
        
        # Create final dictionary
        self.similar_metrics_dict_ = {}
        for cluster_id, metrics in cluster_groups.items():
            for metric in metrics:
                # For each metric, include all other metrics in the same cluster (excluding itself)
                similar_metrics = [m for m in metrics if m != metric]
                self.similar_metrics_dict_[metric] = similar_metrics
        
        if self.verbose:
            print(f"Created similar metrics dictionary with {len(self.similar_metrics_dict_)} entries")
    
    def get_metric_similarity_groups(self) -> List[List[str]]:
        """
        Get similarity groups as lists of similar metrics
        
        Returns:
        - List of lists, where each inner list contains metrics that are similar to each other
        """
        if self.labels_ is None:
            raise ValueError("Model must be fitted first")
            
        # Group metrics by cluster
        cluster_groups = {}
        for i, (metric, cluster_id) in enumerate(zip(self.valid_columns_, self.labels_)):
            if cluster_id not in cluster_groups:
                cluster_groups[cluster_id] = []
            cluster_groups[cluster_id].append(metric)
        
        # Convert to list of lists
        similarity_groups = list(cluster_groups.values())
        
        if self.verbose:
            print(f"Created {len(similarity_groups)} similarity groups")
            for i, group in enumerate(similarity_groups):
                print(f"  Group {i}: {len(group)} metrics")
        
        return similarity_groups
    
    def get_topk_similar_columns(self, column_name: str, top_k: int = 5, 
                               method: str = 'cluster') -> List[Dict[str, Any]]:
        """
        Get top K most similar columns to the specified column
        
        Parameters:
        - column_name: Target column name
        - top_k: Number of similar columns to return
        - method: Similarity computation method ('cluster', 'distance', 'hybrid')
        
        Returns:
        - List of dictionaries with similar columns and their similarity scores
        """
        if self.model is None:
            raise ValueError("Model must be fitted first")
        
        if column_name not in self.valid_columns_:
            raise ValueError(f"Column '{column_name}' not found in valid columns. "
                           f"Available columns: {self.valid_columns_[:10]}...")
        
        if method == 'cluster':
            return self._get_similar_by_cluster(column_name, top_k)
        elif method == 'distance':
            return self._get_similar_by_distance(column_name, top_k)
        elif method == 'hybrid':
            return self._get_similar_hybrid(column_name, top_k)
        else:
            raise ValueError("Method must be 'cluster', 'distance', or 'hybrid'")
    
    def _get_similar_by_cluster(self, column_name: str, top_k: int) -> List[Dict[str, Any]]:
        """Get similar columns based on clustering results"""
        # Get other columns in the same cluster
        cluster_mates = self.similar_metrics_dict_.get(column_name, [])
        
        # If not enough columns in the same cluster, supplement from other clusters
        if len(cluster_mates) < top_k:
            # Get target column's cluster
            target_idx = self.column_to_index_[column_name]
            target_cluster = self.labels_[target_idx]
            
            # Calculate average distance to other clusters
            cluster_distances = {}
            for cluster_id in set(self.labels_):
                if cluster_id != target_cluster:
                    cluster_indices = [i for i, c in enumerate(self.labels_) if c == cluster_id]
                    avg_distance = np.mean([self.pairwise_distances_[target_idx, i] for i in cluster_indices])
                    cluster_distances[cluster_id] = avg_distance
            
            # Sort by distance, select closest clusters
            sorted_clusters = sorted(cluster_distances.items(), key=lambda x: x[1])
            
            # Add columns from closest clusters
            for cluster_id, _ in sorted_clusters:
                additional_mates = [col for i, col in enumerate(self.valid_columns_) 
                                  if self.labels_[i] == cluster_id and col != column_name]
                cluster_mates.extend(additional_mates)
                if len(cluster_mates) >= top_k:
                    break
        
        # Take first top_k
        result = []
        for i, mate in enumerate(cluster_mates[:top_k]):
            mate_idx = self.column_to_index_[mate]
            distance = self.pairwise_distances_[self.column_to_index_[column_name], mate_idx]
            
            result.append({
                'column_name': mate,
                'similarity_score': 1.0 / (1.0 + distance),  # Convert distance to similarity
                'distance': distance,
                'same_cluster': self.labels_[self.column_to_index_[column_name]] == self.labels_[mate_idx],
                'rank': i + 1
            })
        
        return result
    
    def _get_similar_by_distance(self, column_name: str, top_k: int) -> List[Dict[str, Any]]:
        """Get similar columns based on distance matrix"""
        target_idx = self.column_to_index_[column_name]
        
        # Get distances to all other columns
        distances = []
        for i, col in enumerate(self.valid_columns_):
            if col != column_name:
                distance = self.pairwise_distances_[target_idx, i]
                distances.append((col, distance, i))
        
        # Sort by distance (ascending)
        distances.sort(key=lambda x: x[1])
        
        # Build result
        result = []
        for i, (col, distance, col_idx) in enumerate(distances[:top_k]):
            similarity = 1.0 / (1.0 + distance)  # Convert distance to similarity
            same_cluster = self.labels_[target_idx] == self.labels_[col_idx]
            
            result.append({
                'column_name': col,
                'similarity_score': similarity,
                'distance': distance,
                'same_cluster': same_cluster,
                'rank': i + 1
            })
        
        return result
    
    def _get_similar_hybrid(self, column_name: str, top_k: int) -> List[Dict[str, Any]]:
        """Hybrid method: prioritize same cluster, then supplement by distance"""
        target_idx = self.column_to_index_[column_name]
        target_cluster = self.labels_[target_idx]
        
        # Get columns in same cluster and other clusters
        same_cluster_cols = []
        other_cluster_cols = []
        
        for i, col in enumerate(self.valid_columns_):
            if col != column_name:
                distance = self.pairwise_distances_[target_idx, i]
                item = {
                    'column_name': col,
                    'similarity_score': 1.0 / (1.0 + distance),
                    'distance': distance,
                    'same_cluster': self.labels_[i] == target_cluster,
                    'cluster_id': self.labels_[i]
                }
                
                if item['same_cluster']:
                    same_cluster_cols.append(item)
                else:
                    other_cluster_cols.append(item)
        
        # Sort same cluster by distance
        same_cluster_cols.sort(key=lambda x: x['distance'])
        
        # Sort other clusters by distance
        other_cluster_cols.sort(key=lambda x: x['distance'])
        
        # Merge results: same cluster first, then other clusters
        result = same_cluster_cols + other_cluster_cols
        result = result[:top_k]
        
        # Add ranking
        for i, item in enumerate(result):
            item['rank'] = i + 1
        
        return result
    
    def save_results_to_json(self, filename_prefix: str = 'robust_clustering_results', 
                           output_dir: str = './clustering_results') -> Dict[str, str]:
        """
        Save all clustering results to JSON files
        
        Parameters:
        - filename_prefix: Prefix for output files
        - output_dir: Directory to save files
        
        Returns:
        - Dictionary with paths to saved files
        """
        if self.model is None:
            raise ValueError("Model must be fitted first")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = {}
        
        try:
            # 1. Save similar metrics dictionary
            similar_metrics_file = os.path.join(output_dir, f'{filename_prefix}_similar_metrics.json')
            with open(similar_metrics_file, 'w') as f:
                json.dump(self.similar_metrics_dict_, f, indent=2, cls=NumpyEncoder)
            saved_files['similar_metrics'] = similar_metrics_file
            print(f"💾 Saved similar metrics dictionary to '{similar_metrics_file}'")
            
            # 2. Save cluster assignments
            assignments = self.get_cluster_assignments()
            assignments_file = os.path.join(output_dir, f'{filename_prefix}_cluster_assignments.json')
            with open(assignments_file, 'w') as f:
                json.dump(assignments, f, indent=2, cls=NumpyEncoder)
            saved_files['cluster_assignments'] = assignments_file
            print(f"💾 Saved cluster assignments to '{assignments_file}'")
            
            # 3. Save similarity groups
            similarity_groups = self.get_metric_similarity_groups()
            similarity_groups_file = os.path.join(output_dir, f'{filename_prefix}_similarity_groups.json')
            with open(similarity_groups_file, 'w') as f:
                json.dump(similarity_groups, f, indent=2, cls=NumpyEncoder)
            saved_files['similarity_groups'] = similarity_groups_file
            print(f"💾 Saved similarity groups to '{similarity_groups_file}'")
            
            # 4. Save clustering scores and configuration
            config_and_scores = {
                'clustering_configuration': {
                    'n_clusters': self.n_clusters,
                    'final_n_clusters': self.final_n_clusters_,
                    'metric': self.metric,
                    'max_iter': self.max_iter,
                    'scaler_type': self.scaler_type,
                    'window_constraint': self.window_constraint,
                    'window_size': self.window_size_,
                    'warp_penalty': self.warp_penalty
                },
                'clustering_scores': self.clustering_scores_,
                'data_statistics': {
                    'original_metrics_count': len(self.valid_columns_) + len(self.removed_columns_),
                    'valid_metrics_count': len(self.valid_columns_),
                    'removed_metrics_count': len(self.removed_columns_),
                    'removed_metrics': self.removed_columns_
                }
            }
            config_file = os.path.join(output_dir, f'{filename_prefix}_configuration.json')
            with open(config_file, 'w') as f:
                json.dump(config_and_scores, f, indent=2, cls=NumpyEncoder)
            saved_files['configuration'] = config_file
            print(f"💾 Saved configuration and scores to '{config_file}'")
            
            # 5. Save detailed pairwise similarities for top columns
            top_similarities = {}
            for column in self.valid_columns_[:10]:  # Save for first 10 columns to avoid large files
                try:
                    similar_cols = self.get_topk_similar_columns(column, top_k=5, method='hybrid')
                    top_similarities[column] = similar_cols
                except Exception as e:
                    print(f"Warning: Could not compute similarities for {column}: {e}")
            
            similarities_file = os.path.join(output_dir, f'{filename_prefix}_top_similarities.json')
            with open(similarities_file, 'w') as f:
                json.dump(top_similarities, f, indent=2, cls=NumpyEncoder)
            saved_files['top_similarities'] = similarities_file
            print(f"💾 Saved top similarities to '{similarities_file}'")
            
            # 6. Save cluster centers if available
            if self.cluster_centers_ is not None:
                centers_file = os.path.join(output_dir, f'{filename_prefix}_cluster_centers.json')
                # Convert cluster centers to list format for JSON serialization
                centers_list = [center.tolist() for center in self.cluster_centers_]
                with open(centers_file, 'w') as f:
                    json.dump(centers_list, f, indent=2, cls=NumpyEncoder)
                saved_files['cluster_centers'] = centers_file
                print(f"💾 Saved cluster centers to '{centers_file}'")
            
            print(f"\n✅ All results saved to directory: {output_dir}")
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            raise
        
        return saved_files
    
    def get_similar_metrics_dict(self):
        """
        Return dictionary:
        - Key: metric name
        - Value: list of similar metric names (from same cluster)
        """
        if self.similar_metrics_dict_ is None:
            raise ValueError("Model must be fitted first")
            
        return self.similar_metrics_dict_
    
    def get_cluster_assignments(self):
        """
        Get cluster assignments for each metric
        """
        if self.labels_ is None:
            raise ValueError("Model must be fitted first")
            
        assignments = {}
        for i, column in enumerate(self.valid_columns_):
            assignments[column] = int(self.labels_[i])
        return assignments
    
    def get_clustering_scores(self):
        """
        Get clustering quality scores
        """
        return self.clustering_scores_
    
    def get_column_info(self, column_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a column
        """
        if column_name not in self.valid_columns_:
            raise ValueError(f"Column '{column_name}' not found in valid columns")
        
        idx = self.column_to_index_[column_name]
        cluster_id = self.labels_[idx]
        
        # Get other columns in the same cluster
        cluster_mates = [col for i, col in enumerate(self.valid_columns_) 
                        if self.labels_[i] == cluster_id and col != column_name]
        
        return {
            'column_name': column_name,
            'cluster_id': cluster_id,
            'cluster_size': len(cluster_mates) + 1,
            'cluster_mates': cluster_mates,
            'column_index': idx,
            'window_constraint': self.window_constraint,
            'window_size': self.window_size_
        }

    def visualize_similar_columns(self, column_name: str, top_k: int = 5, 
                                method: str = 'hybrid', figsize: tuple = (12, 8)):
        """
        Visualize similar columns
        """
        similar_cols = self.get_topk_similar_columns(column_name, top_k, method)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Plot 1: Time series comparison
        colors = plt.cm.Set3(np.linspace(0, 1, len(similar_cols) + 1))
        
        # Plot target column
        target_data = self.original_data_[column_name]
        ax1.plot(target_data.index, target_data.values, 
                color='red', linewidth=3, label=f'Target: {column_name}', alpha=0.8)
        
        # Plot similar columns
        for i, similar in enumerate(similar_cols):
            similar_data = self.original_data_[similar['column_name']]
            ax1.plot(similar_data.index, similar_data.values,
                    color=colors[i], linewidth=2, alpha=0.7,
                    label=f"{similar['column_name'][:20]} (sim: {similar['similarity_score']:.3f})")
        
        ax1.set_title(f'Top {top_k} Similar Columns to "{column_name}"\n(Window constraint: {self.window_constraint})')
        ax1.set_xlabel('Time', fontweight='bold', fontsize=8)
        ax1.set_ylabel('Value', fontweight='bold', fontsize=8)
        ax1.legend(fontsize=5)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Similarity bar chart
        names = [f'{similar["column_name"][:20]}...{similar["column_name"][-10:]}' for similar in similar_cols]
        scores = [similar['similarity_score'] for similar in similar_cols]
        same_cluster = [similar['same_cluster'] for similar in similar_cols]
        
        colors_bar = ['green' if sc else 'blue' for sc in same_cluster]
        
        bars = ax2.barh(names, scores, color=colors_bar, alpha=0.5)
        ax2.set_xlabel('Similarity Score', fontweight='bold', fontsize=8)
        ax2.set_title('Similarity Scores', fontweight='bold', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, score in zip(bars, scores):
            ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{score:.3f}', va='center', ha='left', fontsize=8)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, label='Same Cluster'),
            Patch(facecolor='blue', alpha=0.7, label='Different Cluster')
        ]
        ax2.legend(handles=legend_elements, loc='lower right', fontsize=8)
        
        plt.tight_layout()
        plt.show()
        
        return similar_cols
