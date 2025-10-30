import pandas as pd
import numpy as np
from dtaidistance import dtw
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
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
    def __init__(self, n_clusters=3, method='complete', metric='dtw', 
                 min_non_zero_ratio=0.1, min_variance=0.001):
        """
        Initialize the Robust Time Series Clustering model
        """
        self.n_clusters = n_clusters
        self.method = method
        self.metric = metric
        self.min_non_zero_ratio = min_non_zero_ratio
        self.min_variance = min_variance
        self.labels_ = None
        self.distance_matrix_ = None
        self.valid_columns_ = None
        self.removed_columns_ = None
        self.final_n_clusters_ = None
        self.normalized_data_ = None
        self.original_data_ = None
        self.similar_metrics_dict_ = None
        
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
                col_null_ratio = float(null_ratio[col])  # Convert to Python float
                col_non_zero_ratio = float(non_zero_ratio[col])  # Convert to Python float
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
        Compute distance matrix using DTW or Euclidean distance
        """
        # Convert DataFrame to numpy array (metrics as columns)
        ts_data = data.values.T
        
        print(f"Computing {self.metric} distance matrix for {ts_data.shape[0]} metrics...")
        
        if self.metric == 'dtw':
            try:
                self.distance_matrix_ = dtw.distance_matrix_fast(ts_data)
                self.distance_matrix_ = np.array(self.distance_matrix_)
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
    
    def fit(self, data):
        """
        Fit the clustering model to the data with robust data handling
        """
        # Clean and preprocess data
        cleaned_data = self._clean_data(data)
        self.original_data_ = cleaned_data
        
        print(f"Data shape after cleaning: {cleaned_data.shape}")
        
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
        
        # Perform hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=self.final_n_clusters_,
            metric='precomputed',
            linkage=self.method
        )
        
        self.labels_ = clustering.fit_predict(distance_matrix)
        
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
            if i < len(self.labels_):  # Safety check
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
        
        Returns:
        - dict: {metric_name: [similar_metric1, similar_metric2, ...]}
        """
        if self.similar_metrics_dict_ is None:
            raise ValueError("Model must be fitted first")
            
        return self.similar_metrics_dict_
    
    def get_cluster_assignments(self):
        """
        Get cluster assignments for each metric
        Returns: {metric_name: cluster_id}
        """
        if self.labels_ is None:
            raise ValueError("Model must be fitted first")
            
        assignments = {}
        for i, column in enumerate(self.valid_columns_):
            if i < len(self.labels_):
                assignments[column] = int(self.labels_[i])  # Convert to Python int
            else:
                assignments[column] = -1
        return assignments
    
    def get_metric_similarity_groups(self):
        """
        Alternative method that returns clusters as lists of similar metrics
        Returns: [[metric1, metric2, ...], [metric5, metric6, ...], ...]
        """
        if self.labels_ is None:
            raise ValueError("Model must be fitted first")
            
        cluster_groups = {}
        for i, (metric, cluster_id) in enumerate(zip(self.valid_columns_, self.labels_)):
            if i < len(self.labels_):
                cluster_id_int = int(cluster_id)  # Convert to Python int
                if cluster_id_int not in cluster_groups:
                    cluster_groups[cluster_id_int] = []
                cluster_groups[cluster_id_int].append(metric)
        
        return list(cluster_groups.values())
    
    def save_results_to_json(self, filename_prefix='clustering_results'):
        """
        Save all results to JSON files with proper type handling
        """
        if self.similar_metrics_dict_ is None:
            raise ValueError("Model must be fitted first")
        
        # Save similar metrics dictionary
        similar_metrics_file = f'{filename_prefix}_similar_metrics.json'
        with open(similar_metrics_file, 'w') as f:
            json.dump(self.similar_metrics_dict_, f, indent=2, cls=NumpyEncoder)
        print(f"💾 Saved similar metrics dictionary to '{similar_metrics_file}'")
        
        # Save cluster assignments
        assignments = self.get_cluster_assignments()
        assignments_file = f'{filename_prefix}_cluster_assignments.json'
        with open(assignments_file, 'w') as f:
            json.dump(assignments, f, indent=2, cls=NumpyEncoder)
        print(f"💾 Saved cluster assignments to '{assignments_file}'")
        
        # Save data quality report
        report = self.get_data_quality_report()
        report_file = f'{filename_prefix}_data_quality_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, cls=NumpyEncoder)
        print(f"💾 Saved data quality report to '{report_file}'")
        
        return {
            'similar_metrics': similar_metrics_file,
            'cluster_assignments': assignments_file,
            'data_quality_report': report_file
        }
    
    def visualize_clusters(self, figsize=(15, 10)):
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
        
        ax1.set_title('Time Series Clusters (Normalized)')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Normalized Value')
        ax1.legend()
        
        # Plot 2: Show similar metrics for a sample metric
        ax2 = axes[0, 1]
        if self.similar_metrics_dict_:
            # Find a metric that has similar counterparts
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
                
                for similar_metric in similar_metrics[:3]:  # Show first 3 similar metrics
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
        
        ax4.bar([str(c) for c in clusters], sizes)  # Convert to string for plotting
        ax4.set_title('Cluster Sizes')
        ax4.set_xlabel('Cluster ID')
        ax4.set_ylabel('Number of Metrics')
        for i, size in enumerate(sizes):
            ax4.text(i, size, str(size), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def get_data_quality_report(self):
        """
        Generate a report on data quality and filtering
        """
        if self.valid_columns_ is None:
            raise ValueError("Model must be fitted first")
            
        total_metrics = len(self.valid_columns_) + len(self.removed_columns_)
        return {
            'original_metrics_count': int(total_metrics),
            'valid_metrics_count': int(len(self.valid_columns_)),
            'removed_metrics_count': int(len(self.removed_columns_)),
            'final_n_clusters': int(self.final_n_clusters_),
            'retention_rate': float(len(self.valid_columns_) / total_metrics if total_metrics > 0 else 0)
        }

def generate_sample_data(n_timesteps=100, n_metrics=15):
    """
    Generate sample time series data with clear patterns for testing
    """
    np.random.seed(42)
    time_index = pd.date_range('2023-01-01', periods=n_timesteps, freq='D')
    
    data = {}
    
    # Create clear patterns that should form clusters
    patterns = [
        # Pattern 1: Sine waves
        lambda i: np.sin(np.linspace(0 + i*0.5, 4*np.pi + i*0.5, n_timesteps)) + np.random.normal(0, 0.1, n_timesteps),
        
        # Pattern 2: Linear trends  
        lambda i: np.linspace(0, 10 + i, n_timesteps) + np.random.normal(0, 0.2, n_timesteps),
        
        # Pattern 3: Seasonal patterns
        lambda i: np.sin(np.linspace(0, 8*np.pi, n_timesteps)) * (1 + i*0.1) + np.random.normal(0, 0.15, n_timesteps),
        
        # Pattern 4: Random walks
        lambda i: np.cumsum(np.random.normal(0, 0.1 + i*0.01, n_timesteps))
    ]
    
    for i in range(n_metrics):
        pattern_type = i % len(patterns)
        data[f'metric_pattern{pattern_type}_{i}'] = patterns[pattern_type](i)
    
    # Add some problematic data
    data['metric_all_zeros'] = np.zeros(n_timesteps)
    data['metric_mostly_nulls'] = np.where(np.random.random(n_timesteps) > 0.3, np.random.normal(0, 1, n_timesteps), np.nan)
    
    df = pd.DataFrame(data, index=time_index)
    return df

def main():
    """
    Main function demonstrating the similar metrics dictionary
    """
    print("=== Time Series Similar Metrics Finder ===")
    
    # Generate sample data
    data = generate_sample_data(n_timesteps=100, n_metrics=15)
    print(f"Generated data shape: {data.shape}")
    print(f"Sample metrics: {list(data.columns)[:8]}...")
    
    # Perform clustering
    ts_cluster = RobustTimeSeriesClustering(
        n_clusters=4,
        method='complete',
        metric='dtw',
        min_non_zero_ratio=0.1,
        min_variance=0.001
    )
    
    try:
        ts_cluster.fit(data)
        
        # Get data quality report
        report = ts_cluster.get_data_quality_report()
        print(f"\n📊 Data Quality Report:")
        print(f"   Original metrics: {report['original_metrics_count']}")
        print(f"   Valid metrics: {report['valid_metrics_count']}")
        print(f"   Final clusters: {report['final_n_clusters']}")
        print(f"   Retention rate: {report['retention_rate']:.1%}")
        
        # 🎯 MAIN RESULT: Get the similar metrics dictionary
        similar_metrics_dict = ts_cluster.get_similar_metrics_dict()
        
        print(f"\n🎯 SIMILAR METRICS DICTIONARY:")
        print(f"   Found {len(similar_metrics_dict)} metrics with similar patterns")
        
        # Display the results
        print(f"\n📋 SIMILARITY GROUPS (first 10 metrics):")
        displayed = 0
        for metric, similar_metrics in similar_metrics_dict.items():
            if displayed >= 10:
                break
            if similar_metrics:  # Only show metrics that have similar counterparts
                print(f"   {metric} -> Similar to: {similar_metrics}")
                displayed += 1
        
        # Show cluster assignments
        assignments = ts_cluster.get_cluster_assignments()
        print(f"\n🏷️  CLUSTER ASSIGNMENTS (first 10):")
        for i, (metric, cluster_id) in enumerate(assignments.items()):
            if i >= 10:
                break
            print(f"   {metric} -> Cluster {cluster_id}")
        
        # Alternative view: similarity groups
        similarity_groups = ts_cluster.get_metric_similarity_groups()
        print(f"\n👥 SIMILARITY GROUPS (clusters):")
        for i, group in enumerate(similarity_groups):
            print(f"   Group {i} ({len(group)} metrics): {group}")
        
        # Save results to JSON files
        saved_files = ts_cluster.save_results_to_json()
        
        # Generate summary statistics
        print(f"\n📈 SUMMARY STATISTICS:")
        metrics_with_similar = sum(1 for similar_list in similar_metrics_dict.values() if len(similar_list) > 0)
        avg_similar_metrics = np.mean([len(similar_list) for similar_list in similar_metrics_dict.values()])
        
        print(f"   Metrics with similar counterparts: {metrics_with_similar}/{len(similar_metrics_dict)}")
        print(f"   Average similar metrics per metric: {avg_similar_metrics:.1f}")
        
        # Count metrics per cluster
        cluster_counts = {}
        for cluster_id in assignments.values():
            cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
        
        print(f"   Metrics per cluster: {cluster_counts}")
        
        # Visualize results
        print(f"\n📊 Generating visualizations...")
        ts_cluster.visualize_clusters()
        
        return similar_metrics_dict
        
    except Exception as e:
        print(f"❌ Error during clustering: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result_dict = main()
    # if result_dict is not None:
    #     for k, v in result_dict.items():
    #         print(k, v)