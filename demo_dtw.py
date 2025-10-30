import pandas as pd
import numpy as np
from dtaidistance import dtw
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
import json
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

class RobustTimeSeriesClustering:
    def __init__(self, n_clusters=3, method='kmeans', metric='dtw', 
                 min_non_zero_ratio=0.1, min_variance=0.001,
                 dtw_window=None, dtw_use_c=False, random_state=42):
        """
        Initialize the Robust Time Series Clustering model
        
        Parameters:
        - n_clusters: Number of clusters to form
        - method: Clustering method ('kmeans' or 'hierarchical')
        - metric: Distance metric ('dtw' or 'euclidean')
        - min_non_zero_ratio: Minimum ratio of non-zero/non-null values to keep a metric
        - min_variance: Minimum variance threshold to keep a metric
        - dtw_window: DTW window size (None for no constraint, or integer for Sakoe-Chiba band)
        - dtw_use_c: Whether to use fast C implementation
        - random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.method = method
        self.metric = metric
        self.min_non_zero_ratio = min_non_zero_ratio
        self.min_variance = min_variance
        self.dtw_window = dtw_window
        self.dtw_use_c = dtw_use_c
        self.random_state = random_state
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
        
    def _clean_data(self, data):
        """
        Clean the input data by removing metrics with insufficient data
        """
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
            # If no columns pass initial filter, be more lenient
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
        
        # Fill remaining null values with forward fill, then backward fill, then 0
        cleaned_data = cleaned_data.ffill().bfill().fillna(0)
        
        # Remove any remaining rows with all null values
        cleaned_data = cleaned_data.dropna(how='all')
        
        print(f"Original metrics: {len(original_columns)}")
        print(f"Valid metrics after cleaning: {len(self.valid_columns_)}")
        print(f"Removed metrics: {len(self.removed_columns_)}")
        
        if len(self.removed_columns_) > 0:
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
    
    def compute_distance_matrix(self, data):
        """
        Compute distance matrix using DTW or Euclidean distance with window constraint
        """
        # Convert DataFrame to numpy array (metrics as columns)
        ts_data = data.values.T
        
        print(f"Computing {self.metric} distance matrix for {ts_data.shape[0]} metrics...")
        
        if self.metric == 'dtw':
            try:
                if self.dtw_use_c:
                    # Using fast C implementation with window
                    print(f"Using DTW with window={self.dtw_window}")
                    self.distance_matrix_ = dtw.distance_matrix_fast(
                        ts_data, 
                        window=self.dtw_window,
                        use_c=True
                    )
                else:
                    # Using pure Python implementation with window
                    print(f"Using DTW (Python) with window={self.dtw_window}")
                    self.distance_matrix_ = dtw.distance_matrix(
                        ts_data, 
                        window=self.dtw_window,
                        use_c=False
                    )
                
                # Convert to proper matrix format
                self.distance_matrix_ = np.array(self.distance_matrix_)
                
                # Replace any NaN or inf values with large distance
                self.distance_matrix_ = np.nan_to_num(self.distance_matrix_, nan=1e6, posinf=1e6, neginf=1e6)
                
                print("DTW distance matrix computed successfully")
                
            except Exception as e:
                print(f"DTW computation failed: {e}. Falling back to Euclidean distance.")
                self.metric = 'euclidean'
                
        if self.metric == 'euclidean':
            from scipy.spatial.distance import pdist, squareform
            try:
                distances = pdist(ts_data.T, metric='euclidean')
                self.distance_matrix_ = squareform(distances)
                print("Euclidean distance matrix computed successfully")
            except Exception as e:
                print(f"Euclidean distance computation failed: {e}")
                raise
            
        return self.distance_matrix_
    
    def _perform_kmeans_clustering(self, distance_matrix):
        """
        Perform K-Means clustering on distance matrix using kernel trick
        """
        print("Performing K-Means clustering...")
        
        # Convert distance matrix to similarity matrix using Gaussian kernel
        gamma = 1.0 / (np.median(distance_matrix[distance_matrix > 0]) ** 2)
        similarity_matrix = np.exp(-gamma * distance_matrix ** 2)
        
        # Perform K-Means on the similarity matrix
        kmeans = KMeans(
            n_clusters=self.final_n_clusters_,
            random_state=self.random_state,
            n_init=10
        )
        
        # Use PCA to reduce dimensionality for better K-Means performance
        pca = PCA(n_components=min(10, similarity_matrix.shape[0]))
        features = pca.fit_transform(similarity_matrix)
        
        self.labels_ = kmeans.fit_predict(features)
        self.cluster_centers_ = kmeans.cluster_centers_
        
        # Calculate clustering quality scores
        try:
            silhouette_avg = silhouette_score(features, self.labels_)
            calinski_harabasz = calinski_harabasz_score(features, self.labels_)
            self.clustering_scores_ = {
                'silhouette_score': float(silhouette_avg),
                'calinski_harabasz_score': float(calinski_harabasz)
            }
            print(f"Clustering scores - Silhouette: {silhouette_avg:.3f}, Calinski-Harabasz: {calinski_harabasz:.3f}")
        except Exception as e:
            print(f"Could not compute clustering scores: {e}")
            self.clustering_scores_ = None
        
        return self.labels_
    
    def _perform_hierarchical_clustering(self, distance_matrix):
        """
        Perform hierarchical clustering on distance matrix
        """
        print("Performing hierarchical clustering...")
        
        clustering = AgglomerativeClustering(
            n_clusters=self.final_n_clusters_,
            metric='precomputed',
            linkage='average'
        )
        
        self.labels_ = clustering.fit_predict(distance_matrix)
        return self.labels_
    
    def fit(self, data):
        """
        Fit the clustering model to the data with robust data handling
        """
        # Clean and preprocess data
        cleaned_data = self._clean_data(data)
        self.original_data_ = cleaned_data
        
        print(f"Data shape after cleaning: {cleaned_data.shape}")
        print(f"Time series length: {len(cleaned_data)}")
        print(f"Clustering method: {self.method}")
        
        # Normalize the data
        scaler = StandardScaler()
        normalized_data = pd.DataFrame(
            scaler.fit_transform(cleaned_data),
            index=cleaned_data.index,
            columns=cleaned_data.columns
        )
        self.normalized_data_ = normalized_data
        
        # Compute distance matrix
        distance_matrix = self.compute_distance_matrix(normalized_data)
        
        if distance_matrix is None or distance_matrix.shape[0] == 0:
            raise ValueError("Failed to compute distance matrix")
        
        print(f"Distance matrix shape: {distance_matrix.shape}")
        
        # Perform clustering based on selected method
        if self.method == 'kmeans':
            self.labels_ = self._perform_kmeans_clustering(distance_matrix)
        elif self.method == 'hierarchical':
            self.labels_ = self._perform_hierarchical_clustering(distance_matrix)
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")
        
        # Convert labels to Python integers
        self.labels_ = [int(label) for label in self.labels_]
        
        print(f"Clustering completed: {self.final_n_clusters_} clusters formed")
        
        # Create the similar metrics dictionary
        self._create_similar_metrics_dict()
        
        return self
    
    def _create_similar_metrics_dict(self):
        """
        Create dictionary where key is metric name and value is list of similar metrics
        """
        if self.labels_ is None or self.valid_columns_ is None:
            raise ValueError("Model must be fitted first")
            
        # Group metrics by cluster
        cluster_groups = {}
        for i, (metric, cluster_id) in enumerate(zip(self.valid_columns_, self.labels_)):
            if i < len(self.labels_):
                if cluster_id not in cluster_groups:
                    cluster_groups[cluster_id] = []
                cluster_groups[cluster_id].append(metric)
        
        # Create the final dictionary
        self.similar_metrics_dict_ = {}
        for cluster_id, metrics in cluster_groups.items():
            for metric in metrics:
                # For each metric, include all other metrics in the same cluster (excluding itself)
                similar_metrics = [m for m in metrics if m != metric]
                self.similar_metrics_dict_[metric] = similar_metrics
        
        print(f"Created similar metrics dictionary with {len(self.similar_metrics_dict_)} entries")
    
    def get_similar_metrics_dict(self):
        """
        Return dictionary where:
        - Key: metric name
        - Value: list of similar metric names (from the same cluster)
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
            if i < len(self.labels_):
                assignments[column] = int(self.labels_[i])
            else:
                assignments[column] = -1
        return assignments
    
    def get_clustering_scores(self):
        """
        Get clustering quality scores
        """
        return self.clustering_scores_
    
    def find_optimal_clusters(self, data, max_clusters=10):
        """
        Find optimal number of clusters using elbow method and silhouette analysis
        """
        print("Finding optimal number of clusters...")
        
        # Clean data first
        cleaned_data = self._clean_data(data)
        self.original_data_ = cleaned_data
        
        # Normalize the data
        scaler = StandardScaler()
        normalized_data = pd.DataFrame(
            scaler.fit_transform(cleaned_data),
            index=cleaned_data.index,
            columns=cleaned_data.columns
        )
        
        # Compute distance matrix
        distance_matrix = self.compute_distance_matrix(normalized_data)
        
        # Convert to similarity matrix for K-Means
        gamma = 1.0 / (np.median(distance_matrix[distance_matrix > 0]) ** 2)
        similarity_matrix = np.exp(-gamma * distance_matrix ** 2)
        
        # Reduce dimensionality
        pca = PCA(n_components=min(10, similarity_matrix.shape[0]))
        features = pca.fit_transform(similarity_matrix)
        
        # Test different numbers of clusters
        k_range = range(2, min(max_clusters + 1, features.shape[0]))
        inertia = []
        silhouette_scores = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(features)
            
            inertia.append(kmeans.inertia_)
            if k > 1:  # Silhouette score requires at least 2 clusters
                try:
                    silhouette_scores.append(silhouette_score(features, labels))
                except:
                    silhouette_scores.append(-1)
            else:
                silhouette_scores.append(-1)
            
            print(f"K={k}: Inertia={inertia[-1]:.2f}, Silhouette={silhouette_scores[-1]:.3f}")
        
        # Find optimal K (elbow method + silhouette)
        optimal_k = self._find_elbow_point(inertia, k_range)
        
        # Also consider silhouette scores
        if len(silhouette_scores) > 0:
            best_silhouette_k = k_range[np.argmax(silhouette_scores)]
            # Prefer the one with better silhouette score if close
            if abs(optimal_k - best_silhouette_k) <= 2:
                optimal_k = best_silhouette_k
        
        print(f"Optimal number of clusters: {optimal_k}")
        
        # Plot results
        self._plot_optimal_clusters(k_range, inertia, silhouette_scores, optimal_k)
        
        return optimal_k
    
    def _find_elbow_point(self, inertia, k_range):
        """Find elbow point using the kneedle algorithm"""
        try:
            from kneed import KneeLocator
            kneedle = KneeLocator(list(k_range), inertia, curve='convex', direction='decreasing')
            return kneedle.elbow if kneedle.elbow else 3
        except:
            # Fallback: simple second derivative method
            differences = np.diff(inertia)
            second_diff = np.diff(differences)
            if len(second_diff) > 0:
                return k_range[np.argmax(second_diff) + 1]
            return 3
    
    def _plot_optimal_clusters(self, k_range, inertia, silhouette_scores, optimal_k):
        """Plot optimal cluster analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Elbow curve
        ax1.plot(k_range, inertia, 'bo-')
        ax1.axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimal K={optimal_k}')
        ax1.set_xlabel('Number of Clusters')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Silhouette scores
        ax2.plot(k_range, silhouette_scores, 'go-')
        ax2.axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimal K={optimal_k}')
        ax2.set_xlabel('Number of Clusters')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Analysis')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def visualize_clusters(self, figsize=(15, 12)):
        """
        Visualize the clustering results
        """
        if self.labels_ is None:
            raise ValueError("Model must be fitted first")
            
        cluster_assignments = self.get_cluster_assignments()
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: All time series colored by cluster
        ax1 = axes[0, 0]
        colors = plt.cm.Set3(np.linspace(0, 1, self.final_n_clusters_))
        
        for metric in self.valid_columns_:
            cluster_id = cluster_assignments.get(metric, -1)
            if cluster_id >= 0 and metric in self.normalized_data_.columns:
                color = colors[cluster_id % len(colors)]
                ax1.plot(self.normalized_data_.index, 
                        self.normalized_data_[metric], 
                        color=color, alpha=0.7,
                        label=f'Cluster {cluster_id}' if metric == self.valid_columns_[0] else "")
        
        ax1.set_title(f'Time Series Clusters ({self.method}, DTW window={self.dtw_window})')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Normalized Value')
        ax1.legend()
        
        # Plot 2: Show similar metrics for a sample metric
        ax2 = axes[0, 1]
        if self.similar_metrics_dict_:
            sample_metric = None
            for metric, similar_metrics in self.similar_metrics_dict_.items():
                if similar_metrics:
                    sample_metric = metric
                    break
            
            if sample_metric:
                similar_metrics = self.similar_metrics_dict_[sample_metric]
                ax2.plot(self.normalized_data_.index, 
                        self.normalized_data_[sample_metric], 
                        'b-', linewidth=3, label=f'Target: {sample_metric}')
                
                for similar_metric in similar_metrics[:3]:
                    ax2.plot(self.normalized_data_.index, 
                            self.normalized_data_[similar_metric], 
                            'r--', alpha=0.7, label=f'Similar: {similar_metric}')
                
                ax2.set_title(f'Similar Metrics to: {sample_metric}')
                ax2.set_xlabel('Time')
                ax2.set_ylabel('Normalized Value')
                ax2.legend()
        
        # Plot 3: Distance matrix heatmap
        ax3 = axes[1, 0]
        if self.distance_matrix_ is not None:
            sns.heatmap(self.distance_matrix_, ax=ax3, cmap='viridis')
            ax3.set_title('Distance Matrix Heatmap')
            ax3.set_xlabel('Metric Index')
            ax3.set_ylabel('Metric Index')
        
        # Plot 4: Cluster sizes
        ax4 = axes[1, 1]
        cluster_sizes = {}
        for metric, cluster_id in cluster_assignments.items():
            if cluster_id not in cluster_sizes:
                cluster_sizes[cluster_id] = 0
            cluster_sizes[cluster_id] += 1
        
        clusters = sorted(cluster_sizes.keys())
        sizes = [cluster_sizes[cluster_id] for cluster_id in clusters]
        
        ax4.bar([str(c) for c in clusters], sizes, color=colors[:len(clusters)])
        ax4.set_title('Cluster Sizes')
        ax4.set_xlabel('Cluster ID')
        ax4.set_ylabel('Number of Metrics')
        for i, size in enumerate(sizes):
            ax4.text(i, size, str(size), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()

def generate_sample_data(n_timesteps=100, n_metrics=20):
    """
    Generate sample time series data with clear patterns for testing
    """
    np.random.seed(42)
    time_index = pd.date_range('2023-01-01', periods=n_timesteps, freq='D')
    
    data = {}
    
    patterns = [
        lambda i: np.sin(np.linspace(0 + i*0.5, 4*np.pi + i*0.5, n_timesteps)) + np.random.normal(0, 0.1, n_timesteps),
        lambda i: np.linspace(0, 10 + i, n_timesteps) + np.random.normal(0, 0.2, n_timesteps),
        lambda i: np.sin(np.linspace(0, 8*np.pi, n_timesteps)) * (1 + i*0.1) + np.random.normal(0, 0.15, n_timesteps),
        lambda i: np.cumsum(np.random.normal(0, 0.1 + i*0.01, n_timesteps)),
        lambda i: np.exp(np.linspace(0, 2, n_timesteps)) + np.random.normal(0, 0.3, n_timesteps)
    ]
    
    for i in range(n_metrics):
        pattern_type = i % len(patterns)
        data[f'metric_pattern{pattern_type}_{i}'] = patterns[pattern_type](i)
    
    df = pd.DataFrame(data, index=time_index)
    return df

def compare_clustering_methods(data):
    """
    Compare K-Means vs Hierarchical clustering
    """
    print("=== Comparing Clustering Methods ===")
    
    methods = ['kmeans', 'hierarchical']
    results = {}
    
    for method in methods:
        print(f"\n🧪 Testing {method.upper()} clustering...")
        
        try:
            ts_cluster = RobustTimeSeriesClustering(
                n_clusters=5,
                method=method,
                metric='dtw',
                dtw_window=2,
                dtw_use_c=True,
                random_state=42
            )
            
            ts_cluster.fit(data)
            
            similar_metrics_dict = ts_cluster.get_similar_metrics_dict()
            assignments = ts_cluster.get_cluster_assignments()
            scores = ts_cluster.get_clustering_scores()
            
            # Calculate statistics
            metrics_with_similar = sum(1 for similar_list in similar_metrics_dict.values() if len(similar_list) > 0)
            avg_similar_metrics = np.mean([len(similar_list) for similar_list in similar_metrics_dict.values()])
            
            cluster_counts = {}
            for cluster_id in assignments.values():
                cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
            
            results[method] = {
                'similar_metrics_dict': similar_metrics_dict,
                'cluster_counts': cluster_counts,
                'metrics_with_similar': metrics_with_similar,
                'avg_similar_metrics': avg_similar_metrics,
                'clustering_scores': scores
            }
            
            print(f"   ✅ Success - {metrics_with_similar} metrics have similar counterparts")
            print(f"   📊 Cluster distribution: {cluster_counts}")
            if scores:
                print(f"   📈 Clustering scores: {scores}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results[method] = {'error': str(e)}
    
    return results

def main():
    """
    Main function demonstrating K-Means clustering
    """
    print("=== K-Means Time Series Clustering ===")
    
    # Generate sample data
    data = generate_sample_data(n_timesteps=100, n_metrics=20)
    print(f"Generated data shape: {data.shape}")
    
    # Example 1: K-Means with automatic optimal cluster detection
    print("\n" + "="*50)
    print("Example 1: K-Means with optimal cluster detection")
    
    ts_cluster_kmeans = RobustTimeSeriesClustering(
        method='kmeans',
        metric='dtw',
        dtw_window=2,
        dtw_use_c=True,
        random_state=42
    )
    
    # Find optimal number of clusters
    optimal_k = ts_cluster_kmeans.find_optimal_clusters(data, max_clusters=8)
    
    # Fit with optimal K
    ts_cluster_kmeans.n_clusters = optimal_k
    ts_cluster_kmeans.fit(data)
    
    # Get results
    similar_metrics_kmeans = ts_cluster_kmeans.get_similar_metrics_dict()
    assignments_kmeans = ts_cluster_kmeans.get_cluster_assignments()
    scores_kmeans = ts_cluster_kmeans.get_clustering_scores()
    
    print(f"\n🎯 K-Means Results (K={optimal_k}):")
    print(f"   Metrics with similar counterparts: {sum(1 for lst in similar_metrics_kmeans.values() if len(lst) > 0)}")
    print(f"   Clustering scores: {scores_kmeans}")
    
    # Example 2: Hierarchical clustering for comparison
    print("\n" + "="*50)
    print("Example 2: Hierarchical clustering")
    
    ts_cluster_hierarchical = RobustTimeSeriesClustering(
        n_clusters=optimal_k,
        method='hierarchical',
        metric='dtw',
        dtw_window=2,
        dtw_use_c=True
    )
    
    ts_cluster_hierarchical.fit(data)
    similar_metrics_hierarchical = ts_cluster_hierarchical.get_similar_metrics_dict()
    
    print(f"\n🎯 Hierarchical Results (K={optimal_k}):")
    print(f"   Metrics with similar counterparts: {sum(1 for lst in similar_metrics_hierarchical.values() if len(lst) > 0)}")
    
    # Compare methods
    print("\n" + "="*50)
    comparison_results = compare_clustering_methods(data)
    
    # Display comparison
    print("\n📊 METHOD COMPARISON:")
    print("=" * 60)
    for method, result in comparison_results.items():
        if 'error' not in result:
            print(f"\n{method.upper()}:")
            print(f"  Metrics with similar counterparts: {result['metrics_with_similar']}")
            print(f"  Average similar metrics: {result['avg_similar_metrics']:.1f}")
            print(f"  Cluster distribution: {result['cluster_counts']}")
            if result['clustering_scores']:
                print(f"  Clustering scores: {result['clustering_scores']}")
    
    # Visualize K-Means results
    print("\n" + "="*50)
    print("Generating K-Means visualization...")
    ts_cluster_kmeans.visualize_clusters()
    
    return comparison_results

if __name__ == "__main__":
    results = main()
    
    print(f"\n🎯 Key advantages of K-Means:")
    print("1. ⚡ Much faster than hierarchical clustering (O(n) vs O(n²))")
    print("2. 📊 Better scalability for large datasets")
    print("3. 🎯 Automatic cluster center computation")
    print("4. 📈 Built-in quality metrics (inertia, silhouette score)")
    print("5. 🔧 Easy to find optimal number of clusters")
    print("\n💡 Recommendation: Use K-Means for datasets with >100 time series")